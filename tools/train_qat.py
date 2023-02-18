import os
import copy
import onnx
import json
import time
import shutil
import numpy as np
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

import argparse
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from pathlib import Path

import bstnnx_training
import bstnnx_training.PyTorch.QAT.core as quantizer
import torch.quantization as quantizer_torch
from bstnnx_training.PyTorch.QAT.utils import utility, serialization
from bstnnx_training.utils.version import EXPORT_ONNX_OPSET_VERSION

from pcdet.datasets import build_dataloader
from pointpillar_estimator import PointPillarEstimator
from pcdet.optimization import build_optimizer, build_scheduler


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file', default='/barn5/xinruizhang/code/OpenPCDet_robosense/tools/cfgs/robosense_models/pp_robosense_baseline_qat_0.yaml')
    parser.add_argument('--ckpt', type=str, help='ckpt', default='/barn3/xinruizhang/code/OpenPCDet/output/robosense_models/pp_robosense_baseline/robosense_baseline_08272021/ckpt/checkpoint_epoch_80.pth')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=12)
    parser.add_argument('--epochs', type=int, help='epochs', default=None)
    parser.add_argument('--output_dir', type=str, help='output_dir', default='pytorch/pointpillar')
    parser.add_argument('--cwd', type=str, help='cwd', default='/barn5/xinruizhang/code/OpenPCDet_robosense/tools')
    return parser.parse_args()


def main():

    # 0. Config preparation
    args = get_arguments()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_name = 'pointpillars_torch'
    input_dir = "./"
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
            raise RuntimeError(f"{input_dir} does not exists")
    print(f"input_dir = {input_dir}")

    def create_dir_if_need(output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        ## convert to absolute_path
        output_dir = os.path.abspath(output_dir)
        return output_dir

    output_dir = "../results/job1"
    output_dir = create_dir_if_need(output_dir)
    print(f"output_dir = {output_dir}")

    tmp_dir = "../tmp_1"
    tmp_dir = create_dir_if_need(tmp_dir)
    print(f"tmp_dir = {tmp_dir}")

    # Model preparation
    estimator = PointPillarEstimator(cfg_file=args.config, batch_size=args.batch_size, cwd=args.cwd)
    ckpt = estimator.get_checkpoint(ckpt=args.ckpt, reset=True)
    os.makedirs(args.output_dir, exist_ok=True)
    model = ckpt['model']

    # use CPU on input_tensor as our backend for parsing GraphTopology forced model to be on CPU
    random_data = np.random.rand(8, 128, 600, 560).astype("float32")
    sample_data = torch.from_numpy(random_data).to(device)
    model.eval()
    fused_model = quantizer.fuse_modules(model, auto_detect=True, input_tensor=sample_data.cpu(), debug_mode=True)
    prepared_model = quantize_model(fused_model, device, backend="bst", sample_data=sample_data)

    # Train model
    display_step = 100
    epochs = 25

    # model, optimizer, lr_scheduler, start_epoch, start_iter = ckpt['model'], ckpt['optimizer'], ckpt['lr_scheduler'], ckpt['start_epoch'], ckpt['start_iter']
    start_iter = 0
    start_epoch = 0

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
    for cur_epoch in range(start_epoch, start_epoch + epochs):
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

        losses.extend(losses_in_steps)
        losses_in_steps = []
        loss_mean = torch.stack(losses).mean()
        print('Epoch %s Average Loss=%s' % (cur_epoch, "{:.4f}".format(loss_mean)))

        if cur_epoch == start_epoch:
            print("Aligning bst hardware...")
            model.apply(torch.quantization.disable_observer)
            quantizer.align_bst_hardware(model, sample_data, debug_mode=True)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

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

    for batch_dict in tqdm(dataloader):
        estimator.load_data_to_device(batch_dict, device)
        with torch.no_grad():
            batch_dict = prepared_model.preprocess(batch_dict)
            (
                batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']
            ) = prepared_model(batch_dict['spatial_features'])
            pred_dicts, ret_dict = prepared_model.postprocess(batch_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=Path(tmp_dir)
        )
        det_annos += annos
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=estimator.config.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=tmp_dir
    )
    print('%s result_str: %s' % (label, result_str))
    print('%s result_dict: %s' % (label, result_dict))
    result_str_list.append(result_str)
    prepared_model.to(device)
    prepared_model.train()

    prepared_model.to(device)
    prepared_model.eval()
    rand_in = np.random.rand(1, 1, 128, 600, 560).astype("float32")
    print("Exporting onnx...")
    sample_in = tuple(torch.from_numpy(x) for x in rand_in)
    onnx_model_path, quant_param_json_path = quantizer.export_onnx(prepared_model, sample_in, result_dir=tmp_dir)
    prepared_model.to(device)
    prepared_model.train()
    print("Done")


def quantize_model(model, device, backend='default', sample_data=None):
    model.to(device)
    model.train()
    if backend == 'default':
        activation_quant = quantizer_torch.fake_quantize.FakeQuantize.with_args(
            observer=quantizer_torch.observer.default_observer.with_args(dtype=torch.qint8),
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        weight_quant = quantizer_torch.fake_quantize.FakeQuantize.with_args(
            observer=quantizer_torch.observer.default_observer.with_args(dtype=torch.qint8),
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        # assign qconfig to model
        model.qconfig = quantizer_torch.QConfig(activation=activation_quant, weight=weight_quant)
        # prepare qat model using qconfig settings
        prepared_model = quantizer_torch.prepare_qat(model, inplace=False)
    elif backend == 'bst':
        bst_activation_quant = quantizer.fake_quantize.FakeQuantize.with_args(
            observer=quantizer.observer.MinMaxObserver.with_args(dtype=torch.qint8),
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        bst_weight_quant = quantizer.fake_quantize.FakeQuantize.with_args(
            observer=quantizer.observer.MinMaxObserver.with_args(dtype=torch.qint8),
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        # 1) [bst_alignment] get b0 pre-bind qconfig adjusting Conv's activation quant scheme
        b0_pre_bind_qconfig = quantizer.pre_bind(model, input_tensor=sample_data.to('cpu'), debug_mode=True, observer_scheme_dict={"weight_scheme": "MinMaxObserver", "activation_scheme": "MinMaxObserver"})
        # 2) assign qconfig to model
        model.qconfig = quantizer.QConfig(activation=bst_activation_quant, weight=bst_weight_quant,
                                                    qconfig_dict=b0_pre_bind_qconfig)
        # 3) prepare qat model using qconfig settings
        prepared_model = quantizer.prepare_qat(model, inplace=False)
        # 4) [bst_alignment] link model observers
        prepared_model = quantizer.link_modules(prepared_model, auto_detect=True, input_tensor=sample_data.to('cpu'), inplace=False, debug_mode=True)

    prepared_model.eval()
    return prepared_model

if __name__ == '__main__':
    main()
