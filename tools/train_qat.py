import os
import copy
import onnx
import json
import time
import datetime
import shutil
import numpy as np
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import glob
import argparse
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from pathlib import Path
from pcdet.utils import common_utils

import bstnnx_training
import bstnnx_training.PyTorch.QAT.core as quantizer
from qat_config import quantize_model, quantize_bias

from pcdet.datasets import build_dataloader
from pointpillar_estimator import PointPillarEstimator
from pcdet.optimization import build_optimizer, build_scheduler


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file', default=None)
    parser.add_argument('--ckpt', type=str, help='ckpt', default=None)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=12)
    parser.add_argument('--epochs', type=int, help='epochs', default=1)
    parser.add_argument('--output_dir', type=str, help='output_dir', default='pytorch/pointpillar')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    return parser.parse_args()


def create_dir_if_need(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        print('directory exists')
        # shutil.rmtree(output_dir)
        # os.makedirs(output_dir)
    ## convert to absolute_path
    output_dir = os.path.abspath(output_dir)
    return output_dir


# For manually fusing models, fuse Conv+BN+ReLU. It is not being used, as we don't fuse ReLU.
def fuse_model(model):
    model.backbone_2d.encoder = torch.quantization.fuse_modules(model.backbone_2d.encoder, [['0', '1', '2'], ['3', '4', '5']])
    model.backbone_2d.blocks[0] = torch.quantization.fuse_modules(model.backbone_2d.blocks[0], [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8'], ['9', '10', '11']])
    model.backbone_2d.blocks[1] = torch.quantization.fuse_modules(model.backbone_2d.blocks[1], [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8'], ['9', '10', '11'], ['12', '13', '14'], ['15', '16', '17']])
    model.backbone_2d.deblocks[0] = torch.quantization.fuse_modules(model.backbone_2d.deblocks[0], ['1', '2', '3'])
    model.backbone_2d.deblocks[1] = torch.quantization.fuse_modules(model.backbone_2d.deblocks[1], ['1', '2', '3'])


def main():

    # 0. Config preparation
    args = get_arguments()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    output_dir = create_dir_if_need(args.output_dir)
    print(f"output_dir = {output_dir}")

    ckpt_save_dir = create_dir_if_need(output_dir + '/ckpt/')
    print(f"ckpt_save_dir = {ckpt_save_dir}")

    pred_save_dir = create_dir_if_need(output_dir + '/preds/')
    print(f"pred_save_dir = {pred_save_dir}") 

    log_file = output_dir + ('/log_pred_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file)

    # Model preparation
    estimator = PointPillarEstimator(cfg_file=args.config, batch_size=args.batch_size, cwd=os.getcwd())
    ckpt = estimator.get_checkpoint(ckpt=args.ckpt, reset=True)
    os.makedirs(args.output_dir, exist_ok=True)
    model = ckpt['model']

    # use CPU on input_tensor as our backend for parsing GraphTopology forced model to be on CPU
    # random_data = np.random.rand(8, 128, 600, 560).astype("float32")
    # We should use batch_size=1, otherwise, there may have some out of memory `killed` errors while `fuse_modules`
    random_data = np.random.rand(1, 128, 600, 560).astype("float32")
    sample_data = torch.from_numpy(random_data).to(device)
    model.eval()
    fused_model = quantizer.fuse_modules(model, auto_detect=True, input_tensor=sample_data.cpu(), debug_mode=True)
    prepared_model = quantize_model(fused_model, device, backend="bst", sample_data=sample_data)

    # Train model
    display_step = 100
    epochs = args.epochs

    # model, optimizer, lr_scheduler, start_epoch, start_iter = ckpt['model'], ckpt['optimizer'], ckpt['lr_scheduler'], ckpt['start_epoch'], ckpt['start_iter']
    start_iter = 0
    start_epoch = 0
    
    # resume training 
    if args.pretrained_model is not None:
        model = prepared_model
        model.load_state_dict(torch.load(args.pretrained_model), strict=False)
    
        ckpt_list = glob.glob(os.path.join(ckpt_save_dir, '*.pth'))
        curr_ckpt = os.path.abspath(args.pretrained_model)
        if len(ckpt_list) > 0 and curr_ckpt in ckpt_list:
            ckpt_list.sort(key=os.path.getmtime)
            start_epoch = int(ckpt_list[-1].split('/')[-1][17:-4])
    print(f"start epoch is {start_epoch}")
    prepared_model.to(device)
    prepared_model.train()
    optimizer = build_optimizer(prepared_model, estimator.config.OPTIMIZATION)
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(estimator.train_loader),
        last_epoch=1, total_epochs=estimator.config.OPTIMIZATION.NUM_EPOCHS, optim_cfg=estimator.config.OPTIMIZATION
    )
    model = prepared_model
    model.to(device)
    model.train()
    steps = 0
    accumulated_iter = start_iter
    dataloader = estimator.train_loader
    result_str_list = []
    ckpt_save_interval=1
    max_ckpt_save_num = 100
    for cur_epoch in range(start_epoch, epochs):
        ckpt['start_epoch'] = cur_epoch
        losses = []
        for batch_dict in tqdm(dataloader):
            try:
                cur_lr = float(optimizer.lr)
            except:
                cur_lr = optimizer.param_groups[0]['lr']
            ckpt['start_iter'] = accumulated_iter
            lr_scheduler.step(accumulated_iter)
            losses_in_steps = []
            optimizer.zero_grad()
            estimator.load_data_to_device(batch_dict, device)
            # with torch.no_grad():
            batch_dict = model.preprocess(batch_dict)
            (
                batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']
            ) = model(batch_dict['spatial_features'])
            ret_dict, tb_dict, disp_dict = model.postprocess(batch_dict)
            loss = ret_dict['loss'].mean()
            loss.backward()
            if hasattr(model, 'update_global_step'):
                model.update_global_step()
            else:
                model.module.update_global_step()
            clip_grad_norm_(model.parameters(), estimator.config.OPTIMIZATION.GRAD_NORM_CLIP)
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

        # save trained model
        trained_epoch = cur_epoch + 1
        if trained_epoch % ckpt_save_interval == 0:
            ckpt_list = glob.glob(ckpt_save_dir + 'checkpoint_epoch_*.pth')
            ckpt_list.sort(key=os.path.getmtime)
            if ckpt_list.__len__() >= max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                    os.remove(ckpt_list[cur_file_idx])

            ckpt_path = os.path.join(ckpt_save_dir, f'checkpoint_epoch_{trained_epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)   

        losses.extend(losses_in_steps)
        losses_in_steps = []
        loss_mean = torch.stack(losses).mean()
        print('Epoch %s Average Loss=%s' % (cur_epoch, "{:.4f}".format(loss_mean)))

        if cur_epoch == start_epoch:
            print("Aligning bst hardware...")
            model.apply(torch.quantization.disable_observer)
            quantizer.align_bst_hardware(model, sample_data, debug_mode=True)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        if cur_epoch >= start_epoch:
            quantize_bias(model)

    print("Training is Done")

    print("Start evaluation...")
    prepared_model = model
    prepared_model.to(device)
    prepared_model.eval()
    label = None
    dataloader = estimator.test_loader
    label = "Testing"
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    # pred_list = []
    for batch_dict in tqdm(dataloader):
        estimator.load_data_to_device(batch_dict, device)
        with torch.no_grad():
            batch_dict = prepared_model.preprocess(batch_dict)
            (
                batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']
            ) = prepared_model(batch_dict['spatial_features'])
            pred_dicts, ret_dict = prepared_model.postprocess(batch_dict)
            # pred_list.append(pred_dicts)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=Path(pred_save_dir)
        )
        det_annos += annos
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=estimator.config.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=pred_save_dir
    )

    # import pickle
    # f_pred_dicts = open('./train_pred_list.pkl', 'wb')
    # pickle.dump(pred_list, f_pred_dicts)
    # f_pred_dicts.close()

    # print('%s result_str: %s' % (label, result_str))
    # print('%s result_dict: %s' % (label, result_dict))
    result_str_list.append(result_str)

    logger.info('********************** Evaluation results  **********************\n')
    logger.info(result_str)

    prepared_model.to(device)
    prepared_model.train()

    prepared_model.to(device)
    prepared_model.eval()
    rand_in = np.random.rand(1, 1, 128, 600, 560).astype("float32")
    print("Exporting onnx...")
    sample_in = tuple(torch.from_numpy(x) for x in rand_in)
    onnx_model_path, quant_param_json_path = quantizer.export_onnx(prepared_model, sample_in, result_dir=output_dir, debug_mode=True)
    prepared_model.to(device)
    prepared_model.train()
    print("Done")


if __name__ == '__main__':
    main()
