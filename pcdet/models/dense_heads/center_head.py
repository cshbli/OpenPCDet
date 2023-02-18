import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ...ops.iou3d_nms.iou3d_nms_utils import rotate_nms_center_point
from ...utils.center_point_utils import Sequential, kaiming_init, \
    draw_heatmap_gaussian, gaussian_radius, multi_apply


class CenterHead(nn.Module):
    def __init__(
            self,
            model_cfg,
            init_bias=-2.19,
            share_conv_channel=64,
            num_hm_conv=2,
            dcn_head=False,
    ):
        super(CenterHead, self).__init__()

        self.model_cfg = model_cfg
        common_heads = dict()
        for heads in model_cfg.DENSE_HEAD.COMMON_HEADS:
            common_heads.update(heads)
        for k in common_heads.keys():
            common_heads[k] = tuple(common_heads[k])

        self.in_channels = sum(model_cfg.DENSE_HEAD.IN_CHANNELS)
        self.code_weights = model_cfg.DENSE_HEAD.CODE_WEIGHTS
        self.weight = model_cfg.DENSE_HEAD.LOC_LOSS_WEIGHT

        tasks = []
        for t_name in model_cfg.DENSE_HEAD.TASKS:
            tasks.append(dict(num_class=1, class_names=[t_name]))

        self.box_n_dim = 9 if 'vel' in common_heads else 7
        self.use_direction_classifier = False
        self.forward_ret_dict = []
        self.target_dict = dict()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.task_heads = tasks

        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(
                    SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )
            else:
                raise NotImplementedError

    def forward(self, data_dict):
        # print("data_dict.keys()", data_dict.keys())

        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)
        ret_list = []
        self.forward_ret_dict = []
        for task in self.tasks:
            self.forward_ret_dict.append(task(x))

        if self.training:
            self.target_dict = self._get_targets(gt_bboxes_3d=data_dict["gt_boxes"],
                                                 gt_labels_3d=data_dict["gt_names"])
        else:
            pass
        return

    @staticmethod
    def _sigmoid(x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        heatmaps, anno_boxes, inds, masks, cats = self.target_dict["heatmaps"], self.target_dict["anno_boxes"], \
                                                  self.target_dict["inds"], self.target_dict["masks"], \
                                                  self.target_dict["cats"]

        loss_dict = dict()
        cls_losses, loc_losses, rpn_losses = [], [], []
        for task_id, preds_dict in enumerate(self.forward_ret_dict):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])
            loss_heatmap = self.crit(preds_dict['hm'], heatmaps[task_id],
                                     inds[task_id], masks[task_id], cats[task_id])
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict['anno_box'] = torch.cat(
                (preds_dict['reg'], preds_dict['height'],
                 preds_dict['dim'], preds_dict['vel'],
                 preds_dict['rot']),
                dim=1)

            loss_bbox = self.crit_reg(preds_dict['anno_box'], masks[task_id], inds[task_id],
                                      target_box)
            loc_loss = (loss_bbox * loss_bbox.new_tensor(self.code_weights)).sum()
            rpn_loss = loss_heatmap + self.weight * loc_loss

            cls_losses.append(loss_heatmap)
            loc_losses.append(loc_loss)
            rpn_losses.append(rpn_loss)

        tb_dict = {
            'rpn_loss_cls': torch.mean(torch.stack(cls_losses)).item(),
            'rpn_loss_loc': torch.mean(torch.stack(loc_losses)).item(),
            'rpn_loss': torch.mean(torch.stack(rpn_losses)).item(),
        }
        return torch.mean(torch.stack(rpn_losses)), loss_dict

    @staticmethod
    def _reorganize_tensors(heatmaps):
        """
        heatmaps is a list with batch_num (tasks of tensors) -> batch_num x tasks x ??
        """
        res = []
        for task_idx in range(len(heatmaps[0])):
            tmp = []
            for batch_idx in range(len(heatmaps)):
                tmp.append(heatmaps[batch_idx][task_idx])
            # print(tmp[-1].shape)
            res.append(torch.stack(tmp))
        return res

    def _get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        # print("len bbox", len(gt_bboxes_3d))
        heatmaps, anno_boxes, inds, masks, cats = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d)
        heatmaps = self._reorganize_tensors(heatmaps)
        anno_boxes = self._reorganize_tensors(anno_boxes)
        inds = self._reorganize_tensors(inds)
        masks = self._reorganize_tensors(masks)
        cats = self._reorganize_tensors(cats)
        all_targets_dict = {
            "heatmaps": heatmaps,
            "anno_boxes": anno_boxes,
            "inds": inds,
            "masks": masks,
            "cats": cats
        }
        return all_targets_dict

    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device

        max_objs = self.model_cfg.TRAIN_CONFIG.MAX_OBJS * self.model_cfg.TRAIN_CONFIG.DENSE_REG
        grid_size = torch.tensor(self.model_cfg.TRAIN_CONFIG.GRID_SIZE)
        pc_range = torch.tensor(self.model_cfg.TRAIN_CONFIG.POINT_CLOUD_RANGE)
        voxel_size = torch.tensor(self.model_cfg.TRAIN_CONFIG.VOXEL_SIZE)

        feature_map_size = grid_size[:2] // self.model_cfg.TRAIN_CONFIG.OUT_SIZE_FACTOR

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0

        # print("shape", gt_labels_3d.shape, gt_bboxes_3d.shape)
        for idx, mask in enumerate(task_masks):
            # print("mask", mask, gt_bboxes_3d.shape)
            task_box = []
            task_class = []
            for m in mask:
                # zero are padded to the end of tensor when batch size > 1
                non_zero_mask = gt_bboxes_3d[m].sum(dim=1) != 0
                # print(non_zero_mask)
                # print(gt_bboxes_3d[m])
                task_box.append(gt_bboxes_3d[m][non_zero_mask])

                # print(gt_bboxes_3d[m], gt_labels_3d[m])
                # raise NotImplementedError
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m][non_zero_mask] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            # print("task_boxes", task_boxes[-1])
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks, cats = [], [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]),
                 int(feature_map_size[1]),
                 int(feature_map_size[0])
                 )
            )

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros(max_objs, dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros(max_objs, dtype=torch.uint8)
            cat = gt_labels_3d.new_zeros(max_objs, dtype=torch.int64)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.model_cfg.TRAIN_CONFIG.OUT_SIZE_FACTOR
                length = length / voxel_size[1] / self.model_cfg.TRAIN_CONFIG.OUT_SIZE_FACTOR
                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.model_cfg.TRAIN_CONFIG.GAUSSIAN_OVERLAP)
                    radius = max(self.model_cfg.TRAIN_CONFIG.MIN_RADIUS, int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.model_cfg.TRAIN_CONFIG.OUT_SIZE_FACTOR
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.model_cfg.TRAIN_CONFIG.OUT_SIZE_FACTOR

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    cat[new_idx] = cls_id
                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # set vx and vy to 0
                    vx, vy = task_boxes[idx][k][6], task_boxes[idx][k][6]

                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    # print(x, y, z, box_dim)
                    # if self.norm_bbox:
                    # vy and vy are replaced by rotations
                    box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0)
                    ])
                    # print("anno_box", anno_box[new_idx], width, length)

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
            cats.append(cat)
            # print("tmo", anno_box.shape, anno_box, cat.shape)
        # raise NotImplementedError

        return heatmaps, anno_boxes, inds, masks, cats

    @torch.no_grad()
    def predict(self):
        # get loss info
        rets = []
        metas = []
        test_cfg = self.model_cfg.TEST_CONFIG
        # self.forward_ret_dict
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=self.forward_ret_dict[0]['hm'].dtype,
                device=self.forward_ret_dict[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(self.forward_ret_dict):
            # convert N C H W to N H W C
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            # if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
            # else:
            #     meta_list = example["metadata"]
            #     if double_flip:
            #         meta_list = meta_list[: 4 * int(batch_size):4]

            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H * W, 2)
            batch_hei = batch_hei.reshape(batch, H * W, 1)

            batch_rot = batch_rot.reshape(batch, H * W, 1)
            batch_dim = batch_dim.reshape(batch, H * W, 3)
            batch_hm = batch_hm.reshape(batch, H * W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']

                batch_vel = batch_vel.reshape(batch, H * W, 2)
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else:
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)

            if test_cfg.get('per_class_nms', False):
                raise NotImplementedError
            else:
                rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range, task_id))

                # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = dict()
            for k in rets[0][i].keys():
                if k in ["box3d_lidar"]:
                    ret["pred_boxes"] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["scores"]:
                    ret["pred_scores"] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret["pred_labels"] = torch.cat([ret[i][k] + 1 for ret in rets])
            # drop velocity term
            ret["pred_boxes"] = ret["pred_boxes"][:, [0, 1, 2, 3, 4, 5, -1]]

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list

    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range, task_id):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) & \
                            (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]
            if test_cfg.get('circular_nms', False):
                raise NotImplementedError
            else:
                selected = rotate_nms_center_point(boxes_for_nms.float(), scores.float(),
                                                   thresh=test_cfg.nms.nms_iou_threshold,
                                                   pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                                   post_max_size=test_cfg.nms.nms_post_max_size)
            # print("nms result", len(box_preds), len(selected))
            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]
            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts


class FastFocalLoss(nn.Module):
    """
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def forward(self, out, target, ind, mask, cat):
        """
        Arguments:
            out: B x C x H x W
            target:B x C x H x W
            ind: B x M
            mask: B x M
            cat: (category id for peaks): B x M
        """
        mask = mask.float()
        gt = torch.pow(1 - target, 4)
        neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
        neg_loss = neg_loss.sum()

        pos_pred_pix = self._transpose_and_gather_feat(out, ind)  # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return - neg_loss
        return - (pos_loss + neg_loss) / num_pos

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    @staticmethod
    def _gather_feat(feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat


class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        """
        Regression loss for an output tensor
        Arguments:
            output (batch x dim x h x w)
            mask (batch x max_objects)
            ind (batch x max_objects)
            target (batch x max_objects x dim)
        """
        pred = self._transpose_and_gather_feat(output, ind)
        mask = mask.float().unsqueeze(2)

        loss = F.l1_loss(pred * mask, target * mask, reduction='none')
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)
        return loss

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    @staticmethod
    def _gather_feat(feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat


class SepHead(nn.Module):
    def __init__(
            self,
            in_channels,
            heads,
            head_conv=64,
            final_kernel=1,
            bn=False,
            init_bias=-2.19,
    ):
        super(SepHead, self).__init__()

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv - 1):
                fc.add(nn.Conv2d(in_channels, head_conv,
                                 kernel_size=final_kernel, stride=1,
                                 padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_conv, classes,
                             kernel_size=final_kernel, stride=1,
                             padding=final_kernel // 2, bias=True))

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict
