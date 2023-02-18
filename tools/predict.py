import argparse
import glob, os
from pathlib import Path
import pickle  
import numpy as np
import torch
import shutil

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
            zeros = np.zeros(len(points)).reshape(len(points), 1)  # jisheng
            points = np.concatenate((points, zeros), axis=1)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
            zeros = np.zeros(len(points)).reshape(len(points), 1)  # jisheng
            points = np.concatenate((points, zeros), axis=1)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--pkl_path', type=str, default='demo_data',
                        help='specify path for prediction pickles')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def save_pred():
    args, cfg = parse_config()
    cfg.ROOT_DIR = '/jsc/ONCE_dataset_pcdet/data'

    data_root = args.data_path
    dirs = os.listdir(data_root)

    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    for i, dir in enumerate(dirs):
        print('Processing %d scenes' % (i + 1))
        print('--------------------------------------------------------------')
        scene_addr = os.path.join(data_root, dir)
        bin_addr = os.path.join(scene_addr, 'lidar_roof')

        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(bin_addr), ext=args.ext, logger=logger
        )
        logger.info(f'Total number of samples: \t{len(demo_dataset)}')

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        with torch.no_grad():
            for idx, data_dict in enumerate(demo_dataset):
                logger.info(f'Visualized sample index: \t{idx + 1}')
                data_dict = demo_dataset.collate_batch([data_dict])
                data_dict_cpu = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)

                pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'].cpu()
                pred_dicts[0]['pred_scores'] = pred_dicts[0]['pred_scores'].cpu()
                pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'].cpu()

                pkl_dir = os.path.join(scene_addr, 'pred_pkls/')
                if not os.path.exists(pkl_dir):
                    os.makedirs(pkl_dir)

                strs = demo_dataset.sample_file_list[idx].split('/')
                pred_dict_dir = pkl_dir + strs[-1][:-4] + '.pkl'

                f_pred_dicts = open(pred_dict_dir, 'wb')
                pickle.dump(pred_dicts, f_pred_dicts)
                f_pred_dicts.close()
        print('Completed %d scenes' % (i + 1))
        print('--------------------------------------------------------------')

    logger.info('Demo done.')

def copy_pcds(bin_root, save_path):
    bins = glob.glob(os.path.join(bin_root, '*.bin'))
    bins.sort()
    for idx, src in enumerate(bins):
        if idx < 4950: continue
        strs = src.split('/')
        det = os.path.join(save_path, strs[-1])
        shutil.copy(src, det)


def predict():
    # predictions
    args, cfg = parse_config()

    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            print('Processing %d frames', idx + 1)
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            data_dict_cpu = demo_dataset.collate_batch([data_dict])  
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # pred_dicts: ['pred_boxes', 'pred_scores', 'pred_labels']
            # print('The size of pred boxes is', pred_dicts[0]['pred_boxes'].size())

            pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'].cpu()
            pred_dicts[0]['pred_scores'] = pred_dicts[0]['pred_scores'].cpu()
            pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'].cpu()

            # path to save prediciton pickles
            strs = str(demo_dataset.sample_file_list[idx]).split('/')
            pred_dict_dir = args.pkl_path + strs[-1][:-4] + '.pkl'

            f_pred_dicts = open(pred_dict_dir, 'wb')
            pickle.dump(pred_dicts, f_pred_dicts)
            f_pred_dicts.close()

    logger.info('Demo done.')

    # copy testing bins
    # bin_root = '/barn4/jishengchen/data/leishen/training/lslidar_2/'
    # save_path = '/barn4/jishengchen/data/leishen/training/test/'
    # copy_pcds(bin_root, save_path)

def main():
    print('to be continued')


if __name__ == '__main__':
    predict()
    # main()

