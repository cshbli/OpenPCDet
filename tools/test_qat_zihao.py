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
import pickle

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
from pcdet.utils import common_utils

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
    parser.add_argument('--ckpt_qat', type=str, help='ckpt for qat', default='ckpt_100_frames.pth')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=12)
    parser.add_argument('--epochs', type=int, help='epochs', default=None)
    parser.add_argument('--output_dir', type=str, help='output_dir', default='pytorch/pointpillar')
    parser.add_argument('--cwd', type=str, help='cwd', default='/barn5/xinruizhang/code/OpenPCDet_robosense/tools')
    parser.add_argument('--no_percentile', dest='no_percentile', action='store_true')
    parser.add_argument('--update_scales', dest='update_scales', action='store_true')
    parser.add_argument('--update_bn', dest='update_bn', action='store_true')
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

def compare_eval_res():
    df_eval = open('eval_pred_list.pkl', 'rb')
    eval_preds = pickle.load(df_eval)

    df_train = open('train_pred_list.pkl', 'rb')
    train_preds = pickle.load(df_train)
    for i in range(len(train_preds)):
        for j in range(len(train_preds[i])):
            np.testing.assert_allclose(eval_preds[i][j]['pred_boxes'].to('cpu'), 
                                       train_preds[i][j]['pred_boxes'].to('cpu'), rtol=1e-5, atol=0)
            print('--------pred_boxes is done---------')
            np.testing.assert_allclose(eval_preds[i][j]['pred_scores'].to('cpu'), 
                                       train_preds[i][j]['pred_scores'].to('cpu'), rtol=1e-5, atol=0)
            print('--------pred_scores is done---------')
            np.testing.assert_allclose(eval_preds[i][j]['pred_labels'].to('cpu'), 
                                       train_preds[i][j]['pred_labels'].to('cpu'), rtol=1e-5, atol=0)
            print('--------pred_labels is done---------')

def main():
    # 0. Config preparation
    args = get_arguments()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    output_dir = create_dir_if_need(args.output_dir)
    print(f"output_dir = {output_dir}")

    pred_save_dir = create_dir_if_need(output_dir + '/preds/')
    print(f"pred_save_dir = {pred_save_dir}") 

    log_file = output_dir + ('/log_pred_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file)

    # Model preparation
    estimator = PointPillarEstimator(cfg_file=args.config, batch_size=args.batch_size, cwd=args.cwd)
    ckpt = estimator.get_checkpoint(ckpt=args.ckpt, reset=True)
    model = ckpt['model']

    # use CPU on input_tensor as our backend for parsing GraphTopology forced model to be on CPU
    random_data = np.random.rand(4, 128, 600, 560).astype("float32")
    sample_data = torch.from_numpy(random_data).to(device)
    model.eval()
    fused_model = quantizer.fuse_modules(model, auto_detect=True, input_tensor=sample_data.cpu(), debug_mode=True)
    prepared_model = quantize_model(fused_model, device, backend="bst", sample_data=sample_data)

    model = prepared_model
    model.load_state_dict(torch.load(args.ckpt_qat))
    model.apply(torch.quantization.disable_observer)
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    print("Start evaluation...")
    prepared_model = model
    prepared_model.to(device)
    prepared_model.eval()

    result_str_list = []
    det_annos = []
    # pred_list = []

    label = None
    #dataloader = estimator.calib_loader
    dataloader = estimator.test_loader
    label = "Testing"
    dataset = dataloader.dataset
    class_names = dataset.class_names

    for batch_dict in tqdm(dataloader):
        estimator.load_data_to_device(batch_dict, device)
        with torch.no_grad():
            batch_dict = prepared_model.preprocess(batch_dict)
            (
                batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']
            ) = prepared_model(batch_dict['spatial_features'])
            pred_dicts, ret_dict = prepared_model.postprocess(batch_dict)
            # for i in range(len(pred_dicts)):
            #     pred_dicts[i]['frame_id'] = batch_dict['frame_id'][i]
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

    # f_pred_dicts = open('./eval_pred_list.pkl', 'wb')
    # pickle.dump(pred_list, f_pred_dicts)
    # f_pred_dicts.close()

    result_str_list.append(result_str)

    logger.info(result_str)

    # export onnx model and json 
    prepared_model.to(device)
    prepared_model.train()

    prepared_model.to(device)
    prepared_model.eval()
    rand_in = np.random.rand(1, 1, 128, 600, 560).astype("float32")
    print("Exporting onnx...")
    sample_in = tuple(torch.from_numpy(x) for x in rand_in)
    onnx_model_path, quant_param_json_path = quantizer.export_onnx(prepared_model, sample_in, result_dir=output_dir, debug_mode=False)
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
            # observer=quantizer.observer.MinMaxObserver.with_args(dtype=torch.qint8), 
            observer=quantizer.observer.MinMaxObserver.with_args(dtype=torch.qint8,quantile=0.95), 
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        bst_weight_quant = quantizer.fake_quantize.FakeQuantize.with_args(
            # observer=quantizer.observer.MinMaxObserver.with_args(dtype=torch.qint8), 
            observer=quantizer.observer.MinMaxObserver.with_args(dtype=torch.qint8,quantile=0.95), 
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        # 1) [bst_alignment] get b0 pre-bind qconfig adjusting Conv's activation quant scheme
        b0_pre_bind_qconfig = quantizer.pre_bind(model, input_tensor=sample_data.to('cpu'), debug_mode=False, observer_scheme_dict={"weight_scheme": "MinMaxObserver", "activation_scheme": "MinMaxObserver"})
        # 2) assign qconfig to model
        model.qconfig = quantizer.QConfig(activation=bst_activation_quant, weight=bst_weight_quant,
                                                    qconfig_dict=b0_pre_bind_qconfig)
        # 3) prepare qat model using qconfig settings
        prepared_model = quantizer.prepare_qat(model, inplace=False)    
        # 4) [bst_alignment] link model observers
        prepared_model = quantizer.link_modules(prepared_model, auto_detect=True, input_tensor=sample_data.to('cpu'), inplace=False, debug_mode=False)
    
    prepared_model.eval()
    return prepared_model

if __name__ == '__main__':
    main()
    # compare_eval_res()

