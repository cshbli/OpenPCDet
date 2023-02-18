import torch
from .detector3d_template import Detector3DTemplate
from .. import dense_heads


class PointPillarCenterNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    # def forward(self, batch_dict):
    #     for cur_module in self.module_list:
    #         batch_dict = cur_module(batch_dict)
    #
    #     if self.training:
    #         loss, tb_dict, disp_dict = self.get_training_loss()
    #
    #         ret_dict = {
    #             'loss': loss
    #         }
    #         return ret_dict, tb_dict, disp_dict
    #     else:
    #         pred_dicts, recall_dicts = self.post_processing(batch_dict)
    #         return pred_dicts, recall_dicts
    #
    # def get_training_loss(self):
    #     disp_dict = {}
    #
    #     loss_rpn, tb_dict = self.dense_head.get_loss()
    #     tb_dict = {
    #         'loss_rpn': loss_rpn.item(),
    #         **tb_dict
    #     }
    #
    #     loss = loss_rpn
    #     return loss, tb_dict, disp_dict

    def forward(self, batch_dict):
        # print("batch_dict", batch_dict["gt_boxes"])
        # if self.training:
        #     loss_dict = self.get_training_loss(batch_dict)
        #
        #     res = []
        #     tb_dict = dict()
        #     for k, v in loss_dict.items():
        #         if "combined" in k:
        #             res.append(v)
        #
        #         mean = torch.mean(v)
        #         tb_dict[k] = mean.item()
        #     ret_dict = {
        #         'loss': torch.stack(res)
        #     }
        #     print("cur_loss", torch.mean(ret_dict['loss']))
        #     return ret_dict, tb_dict, tb_dict
        # else:
        #     pred_dicts = self.dense_head.predict(batch_dict)
        #     return pred_dicts, {}
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts = self.dense_head.predict()
            return pred_dicts, {}


    def get_training_loss(self):
        # gt_labels = batch_dict["gt_names"]
        # gt_boxes = batch_dict["gt_boxes"]
        # print("frame", original_batch_dict["frame_id"])
        # loss_dict = self.dense_head.get_loss()
        # return loss_dict
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict



    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict

        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](model_cfg=self.model_cfg)

        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict
