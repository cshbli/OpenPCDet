import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--pkl', type=str, default=None, help='pkl_file_path')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    np.random.seed(1024)

    return args, cfg

def main():
    args, cfg = parse_config()

    log_file = "./debug.log"
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    pkl_path = args.pkl

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=4, logger=logger, training=False
    )
    eval_utils.eval_from_file(cfg, pkl_path, test_loader, logger)

if __name__ == '__main__':
    main()
