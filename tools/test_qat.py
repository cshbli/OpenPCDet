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

from qat_config import quantize_model, quantize_bias

from pcdet.datasets import build_dataloader
from pointpillar_estimator import PointPillarEstimator
from pcdet.optimization import build_optimizer, build_scheduler

# Only necessary if we want to get intermediate layer output
# from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file', default=None)
    parser.add_argument('--ckpt', type=str, help='ckpt', default=None)
    parser.add_argument('--ckpt_qat', type=str, help='ckpt for qat', default='ckpt_100_frames.pth')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=12)
    parser.add_argument('--epochs', type=int, help='epochs', default=None)
    parser.add_argument('--output_dir', type=str, help='output_dir', default='pytorch/pointpillar')
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
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    output_dir = create_dir_if_need(args.output_dir)
    print(f"output_dir = {output_dir}")

    pred_save_dir = output_dir + '/preds/'
    pred_save_dir = create_dir_if_need(pred_save_dir)
    print(f"pred_save_dir = {pred_save_dir}") 

    log_file = output_dir + ('/log_pred_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file)

    # Model preparation
    estimator = PointPillarEstimator(cfg_file=args.config, batch_size=args.batch_size, cwd=os.getcwd())
    ckpt = estimator.get_checkpoint(ckpt=args.ckpt, reset=True)
    model = ckpt['model']

    # use CPU on input_tensor as our backend for parsing GraphTopology forced model to be on CPU
    # random_data = np.random.rand(8, 128, 600, 560).astype("float32")
    random_data = np.random.rand(1, 128, 600, 560).astype("float32")
    sample_data = torch.from_numpy(random_data).to(device)
    model.eval()
    fused_model = quantizer.fuse_modules(model, auto_detect=True, input_tensor=sample_data.cpu(), debug_mode=True)
    prepared_model = quantize_model(fused_model, device, backend="bst", sample_data=sample_data)

    model = prepared_model
    model.load_state_dict(torch.load(args.ckpt_qat))

    model.apply(torch.quantization.disable_observer)
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    quantize_bias(model)

    print("Start evaluation...")
    prepared_model = model
    prepared_model.to(device)
    prepared_model.eval()

    result_str_list = []
    det_annos = []
    pred_list = []

    label = None
    dataloader = estimator.test_loader
    label = "Testing"
    dataset = dataloader.dataset
    class_names = dataset.class_names

    # pred_list = []
    for batch_dict in tqdm(dataloader):
        estimator.load_data_to_device(batch_dict, device)
        with torch.no_grad():            
            batch_dict = prepared_model.preprocess(batch_dict)
            return_layers = {
                'backbone_2d.quant': 'input_quant',
            }            
            (batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']) = prepared_model(batch_dict['spatial_features'])
            # For getting intermediate outputs from Torch model
            # mid_getter = MidGetter(prepared_model, return_layers=return_layers, keep_output=True)
            # mid_outputs, (batch_dict['cls_preds'], batch_dict['box_preds'], batch_dict['dir_cls_preds']) = mid_getter(batch_dict['spatial_features'])
            # np.save("input", batch_dict['spatial_features'].cpu().numpy())
            # np.save("input_quant", mid_outputs['input_quant'].cpu().numpy())
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

    # print('%s result_str: %s' % (label, result_str))
    # print('%s result_dict: %s' % (label, result_dict))
    result_str_list.append(result_str)

    logger.info('********************** Evaluation results  **********************\n')
    logger.info(result_str)

    # export onnx model and json
    prepared_model.to(device)
    prepared_model.eval()
    rand_in = np.random.rand(1, 1, 128, 600, 560).astype("float32")
    print("Exporting onnx...")
    sample_in = tuple(torch.from_numpy(x) for x in rand_in)
    # onnx_model_path, quant_param_json_path = quantizer.export_onnx(prepared_model, sample_in, result_dir=output_dir)
    onnx_model_path, quant_param_json_path = quantizer.export_onnx(prepared_model, sample_in, result_dir=output_dir, debug_mode=True)
    prepared_model.to(device)
    prepared_model.train()
    print("Done")


if __name__ == '__main__':
    main()
    # compare_eval_res()

