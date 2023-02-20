from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            # batch_dict dict_keys(['points', 'frame_id', 'gt_boxes', 'use_lead_xyz', 'voxels',
            # 'voxel_coords', 'voxel_num_points', 'image_shape', 'batch_size', 'pillar_features'])
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            # print("loss", loss)
            # print(tb_dict.keys())
            # print(disp_dict.keys())
            # loss tensor(2.2818, device='cuda:0', grad_fn=<AddBackward0>)
            # dict_keys(['loss_rpn', 'rpn_loss_cls', 'rpn_loss_loc', 'rpn_loss_dir', 'rpn_loss'])
            # dict_keys([])
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # pred_boxes tensor([], device='cuda:0', size=(0, 7))
            # pred_scores tensor([], device='cuda:0')
            # pred_labels tensor([], device='cuda:0', dtype=torch.int64)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
            disp_dict = {}

            loss_rpn, tb_dict = self.dense_head.get_loss()
            tb_dict = {
                'loss_rpn': loss_rpn.item(),
                **tb_dict
            }

            loss = loss_rpn
            return loss, tb_dict, disp_dict
