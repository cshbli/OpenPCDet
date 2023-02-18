import copy
import logging
import os
from pathlib import Path

import numpy as np
import cv2
import pickle

import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.optimization import build_optimizer, build_scheduler
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti

from bstnnx_training.PyTorch.QAT.estimators import estimator


logger = logging.getLogger(__name__)


class PointPillarEstimator(estimator.Estimator):
    def __init__(
            self,
            cfg_file='/barn5/xinruizhang/code/OpenPCDet_robosense/tools/cfgs/robosense_models/pp_robosense_baseline_qat.yaml',
            cwd='/barn5/xinruizhang/code/OpenPCDet_robosense/tools',
            batch_size=8, dist=False, workers=0
    ):
        current_path = os.getcwd()
        self.batch_size = batch_size
        self.dist = dist
        os.chdir(cwd)
        config = copy.deepcopy(cfg)
        cfg_from_yaml_file(cfg_file, config)
        config.TAG = Path(cfg_file).stem
        config.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])
        self.config = config
        self.calib_set, self.calib_loader, self.calib_sampler = build_dataloader(
            dataset_cfg=config.DATA_CONFIG,
            class_names=config.CLASS_NAMES,
            batch_size=batch_size,
            dist=dist, workers=workers, logger=logger, training=True, calib=True
        )
        self.test_set, self.test_loader, self.sampler = build_dataloader(
            dataset_cfg=config.DATA_CONFIG,
            class_names=config.CLASS_NAMES,
            batch_size=batch_size,
            dist=dist, workers=workers, logger=logger, training=False
        )
        self.train_set, self.train_loader, self.train_sampler = build_dataloader(
            dataset_cfg=config.DATA_CONFIG,
            class_names=config.CLASS_NAMES,
            batch_size=batch_size,
            dist=dist, workers=workers,
            logger=logger,
            training=True
        )
        os.chdir(current_path)
        self.epochs = 20

    def get_checkpoint(
	self,
        ckpt='/barn3/xinruizhang/code/OpenPCDet/output/robosense_models/pp_robosense_baseline/robosense_baseline_08272021/ckpt/checkpoint_epoch_80.pth',
        total_epochs=80, reset=True
    ):
        model = build_network(model_cfg=self.config.MODEL, num_class=len(self.config.CLASS_NAMES), dataset=self.train_set)
        optimizer = build_optimizer(model, self.config.OPTIMIZATION)
        # model.load_params_from_file(ckpt, logger=logger, to_cpu=self.dist)
        if reset:
            model.load_params_from_file(ckpt, logger=logger, to_cpu=self.dist)
            start_epoch = 0
            it = 0
        else:
            it, start_epoch = model.load_params_with_optimizer(ckpt, to_cpu=self.dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer, total_iters_each_epoch=len(self.train_loader),
            last_epoch=last_epoch, total_epochs=self.config.OPTIMIZATION.NUM_EPOCHS, optim_cfg=self.config.OPTIMIZATION
        )
        self.epochs = self.config.OPTIMIZATION.NUM_EPOCHS
        return {
		'model': model,
		'optimizer': optimizer,
		'lr_scheduler': lr_scheduler,
                'lr_warmup_scheduler': lr_warmup_scheduler,
                'start_iter': it,
                'start_epoch': start_epoch,
                'total_epochs': self.config.OPTIMIZATION.NUM_EPOCHS
	}

    def clone_checkpoint(self, ckpt):
        model = ckpt['model']
        start_iter = ckpt['start_iter']
        start_epoch = ckpt['start_epoch']
        total_epochs = ckpt['total_epochs']
        optimizer = build_optimizer(model, self.config.OPTIMIZATION)
        last_epoch = start_epoch + 1
        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer, total_iters_each_epoch=len(self.train_loader),
            last_epoch=last_epoch, total_epochs=total_epochs, optim_cfg=self.config.OPTIMIZATION
        )
        return {
                'model': model,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'lr_warmup_scheduler': lr_warmup_scheduler,
                'start_iter': start_iter,
                'start_epoch': start_epoch,
                'total_epochs': total_epochs
        }


    def load_data_to_device(self, batch_dict, device):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
                continue
            batch_dict[key] = torch.from_numpy(val).float().to(device)

    def train_model_official_schedule(self, ckpt, display_step=100, device='cpu', epochs=None):
        model, optimizer, lr_scheduler, start_epoch, start_iter = ckpt['model'], ckpt['optimizer'], ckpt['lr_scheduler'], ckpt['start_epoch'], ckpt['start_iter']
        model.to(device)
        model.verify_quantization_(verify_quantization=False, force=False)
        model.update_range_()
        model.update_attr('bn_training', True)
        model.train()
        steps = 0
        accumulated_iter = start_iter
        dataloader = self.train_loader
        freeze_bn = True
        freeze_scales = True
        if not epochs:
            epochs = self.epochs
        else:
            epochs = epochs
        for cur_epoch in range(start_epoch, start_epoch + epochs):
            if cur_epoch > 2 and freeze_bn:
                model.update_attr('bn_training', False)
                model.train()
                freeze_bn = False
            if cur_epoch > 3 and freeze_scales:
                model.update_range_(False)
                model.verify_quantization_(force=True)
                model.train()
                freeze_scales = False

            ckpt['start_epoch'] = cur_epoch
            losses = []
            for batch_dict in dataloader:
                try:
                    cur_lr = float(optimizer.lr)
                except:
                    cur_lr = optimizer.param_groups[0]['lr']
                ckpt['start_iter'] = accumulated_iter
                lr_scheduler.step(accumulated_iter)
                losses_in_steps = []
                optimizer.zero_grad()
                self.load_data_to_device(batch_dict, device)
                with torch.no_grad():
                    batch_dict = model.preprocess(batch_dict)
                (
                    batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']
                ) = model(batch_dict['spatial_features'])
                ret_dict, tb_dict, disp_dict = model.postprocess(batch_dict)
                loss = ret_dict['loss'].mean()
                if hasattr(model, 'update_global_step'):
                    model.update_global_step()
                else:
                    model.module.update_global_step()
                loss.backward()
                clip_grad_norm_(model.parameters(), self.config.OPTIMIZATION.GRAD_NORM_CLIP)
                optimizer.step()
                loss = loss.detach()
                # losses.append(loss)
                losses_in_steps.append(loss)
                steps += 1
                accumulated_iter += 1
                if steps % display_step == 0:
                    loss_in_steps = torch.stack(losses_in_steps).mean()
                    print('Step ' + str(accumulated_iter) + ", Minibatch Loss=" +
                            "{:.4f}".format(loss) + ", Average Loss since last display=" +
                            "{:.4f}".format(loss_in_steps)
                    )
                    print('cur_lr: {}'.format(cur_lr))
                    losses.extend(losses_in_steps)
                    losses_in_steps = []
            losses.extend(losses_in_steps)
            losses_in_steps = []
            loss_mean = torch.stack(losses).mean()
            print('Epoch %s Average Loss=%s' % (cur_epoch, "{:.4f}".format(loss_mean)))
        print("Training is Done")

    def train_model(self, ckpt, display_step=100, device='cpu', epochs=None):
        model, optimizer, lr_scheduler, start_epoch, start_iter = ckpt['model'], ckpt['optimizer'], ckpt['lr_scheduler'], ckpt['start_epoch'], ckpt['start_iter']
        model.to(device)
        model.train()
        steps = 0
        accumulated_iter = start_iter
        dataloader = self.train_loader
        if not epochs:
            epochs = self.epochs
        else:
            epochs = epochs
        for cur_epoch in range(start_epoch, start_epoch + epochs):
            ckpt['start_epoch'] = cur_epoch
            losses = []
            for batch_dict in dataloader:
                try:
                    cur_lr = float(optimizer.lr)
                except:
                    cur_lr = optimizer.param_groups[0]['lr']
                ckpt['start_iter'] = accumulated_iter
                lr_scheduler.step(accumulated_iter)
                losses_in_steps = []
                optimizer.zero_grad()
                self.load_data_to_device(batch_dict, device)
                with torch.no_grad():
                    batch_dict = model.preprocess(batch_dict)
                (
                    batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']
                ) = model(batch_dict['spatial_features'])
                ret_dict, tb_dict, disp_dict = model.postprocess(batch_dict)
                loss = ret_dict['loss'].mean()
                if hasattr(model, 'update_global_step'):
                    model.update_global_step()
                else:
                    model.module.update_global_step()
                loss.backward()
                clip_grad_norm_(model.parameters(), self.config.OPTIMIZATION.GRAD_NORM_CLIP)
                optimizer.step()
                loss = loss.detach()
                # losses.append(loss)
                losses_in_steps.append(loss)
                steps += 1
                accumulated_iter += 1
                if steps % display_step == 0:
                    loss_in_steps = torch.stack(losses_in_steps).mean()
                    print('Step ' + str(accumulated_iter) + ", Minibatch Loss=" +
                            "{:.4f}".format(loss) + ", Average Loss since last display=" +
                            "{:.4f}".format(loss_in_steps)
                    )
                    print('cur_lr: {}'.format(cur_lr))
                    losses.extend(losses_in_steps)
                    losses_in_steps = []
            losses.extend(losses_in_steps)
            losses_in_steps = []
            loss_mean = torch.stack(losses).mean()
            print('Epoch %s Average Loss=%s' % (cur_epoch, "{:.4f}".format(loss_mean)))
        print("Training is Done")

    def eval_model(
            self, model,
            output_path='pytorch/poitpillar/output',
            device="cpu"
    ):
        model.to(device)
        model.eval()
        label = None
        dataloader = self.test_loader
        label = "Testing"
        dataset = dataloader.dataset
        class_names = dataset.class_names
        det_annos = []
        for batch_dict in dataloader:
            self.load_data_to_device(batch_dict, device)
            with torch.no_grad():
                batch_dict = model.preprocess(batch_dict)
                (
                    batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']
                ) = model(batch_dict['spatial_features'])
                pred_dicts, ret_dict = model.postprocess(batch_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=Path(output_path)
            )
            det_annos += annos
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric=self.config.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=output_path
        )
        pkl_file = os.path.join(output_path, 'result.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(det_annos, f)
        print('%s result_str: %s' % (label, result_str))
        print('%s result_dict: %s' % (label, result_dict))

    def calib_model(
            self, model,
            output_path='pytorch/poitpillar/output',
            device="cpu"
    ):
        model.to(device)
        model.eval()
        label = None
        dataloader = self.calib_loader
        label = "Training"
        dataset = dataloader.dataset
        class_names = dataset.class_names
        det_annos = []
        for batch_dict in dataloader:
            self.load_data_to_device(batch_dict, device)
            with torch.no_grad():
                batch_dict = model.preprocess(batch_dict)
                (
                    batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']
                ) = model(batch_dict['spatial_features'])
                pred_dicts, ret_dict = model.postprocess(batch_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=Path(output_path)
            )
            det_annos += annos
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric=self.config.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=output_path
        )
        print('%s result_str: %s' % (label, result_str))
        print('%s result_dict: %s' % (label, result_dict))

    def fit_model(
        self, model,
        batch_steps=0,
        device="cpu"
    ):
        model.to(device)
        model.eval()
        dataloader = self.test_loader
        dataset = dataloader.dataset
        class_names = dataset.class_names
        batch_dicts = []
        for batch_dict in dataloader:
            self.load_data_to_device(batch_dict, device)
            with torch.no_grad():
                batch_dict = model.preprocess(batch_dict)
                (
                    batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']
                ) = model(batch_dict['spatial_features'])
            batch_dicts.append(batch_dict)
            if batch_steps >= 0 and len(batch_dicts) >= batch_steps:
                break
        return batch_dicts

    def get_predicts(self, model, batch_dicts, device="cpu"):
        model.to(device)
        model.eval()
        det_annos = []
        dataloader = self.test_loader
        dataset = dataloader.dataset
        class_names = dataset.class_names
        for batch_dict in batch_dicts:
            self.load_data_to_device(batch_dict, device)
            with torch.no_grad():
                pred_dicts, ret_dict = model.postprocess(batch_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=None
            )
            det_annos += annos
        return det_annos

    def get_image_from_dataset(self, dataset, idx):
        img_file = dataset.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return cv2.imread(str(img_file), cv2.IMREAD_COLOR)

    def get_label_from_datasest(self, dataset, idx):
        return dataset.get_label(idx)

    def get_label_from_pred(self, path, idx):
        label_file = Path(path) / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def show_image_with_boxes(self, img, objects, score_thresh=-1):
        """ Show image with 2D bounding boxes """
        img1 = np.copy(img)  # for 2d bbox
        for obj in objects:
            box2d = obj.box2d
            topleft = (int(box2d[0]), int(box2d[1]))
            bottomright = (int(box2d[2]), int(box2d[3]))
            if obj.cls_type == "DontCare":
                continue
            if obj.score < score_thresh:
                continue
            if obj.cls_type == "Car":
                cv2.rectangle(
                    img1,
                    topleft,
                    bottomright,
                    (0, 255, 0),
                    2,
                )
            if obj.cls_type == "Pedestrian":
                cv2.rectangle(
                    img1,
                    topleft,
                    bottomright,
                    (255, 255, 0),
                    2,
                )
            if obj.cls_type == "Cyclist":
                cv2.rectangle(
                    img1,
                    topleft,
                    bottomright,
                    (0, 255, 255),
                    2,
                )
        return img1