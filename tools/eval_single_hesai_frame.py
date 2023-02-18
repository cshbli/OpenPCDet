import argparse
import datetime
import os
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch
from pypcd import pypcd

import pcdet.utils.calibration_hesai as calibration_hesai
import tools.visual_utils.visualize_utils as visualize_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.models import build_network
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, box_utils


# from tools.visual_utils.draw_3d_bbox_bev import SingleBevBboxDrawer
# from tools.visual_utils.draw_3d_bbox_image import SingleImg3DBboxDrawer


class LoadSingleHesaiScene:
    def __init__(self, anno_folder_path, sequence_idx, cfg):
        self.anno_folder_path = anno_folder_path
        self.sample_idx = sequence_idx
        self.class_names = cfg.CLASS_NAMES
        self.dataset_cfg = cfg.DATA_CONFIG
        self.point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            cfg.DATA_CONFIG.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            cfg.DATA_CONFIG.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=False
        )
        self.calib = None
        self.lidar_paths = []
        for cur, _dirs, files in os.walk(anno_folder_path):
            for file in files:
                if not file.endswith("pcd") and not file.endswith("bin"):
                    continue
                self.lidar_paths.append(os.path.join(cur, file))

        self.lidar_paths.sort()

    def get_lidar(self):
        lidar_file = Path(self.lidar_paths[self.sample_idx])
        assert lidar_file.exists(), lidar_file + "not exist"

        if self.lidar_paths[self.sample_idx].endswith("pcd"):
            return self._read_pcd(str(lidar_file))
        elif self.lidar_paths[self.sample_idx].endswith("bin"):
            return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        else:
            raise NotImplementedError

    @staticmethod
    def get_image_shape():
        return [1920, 1080]

    def get_calib(self, front_to_rear_axle=3.838, rear_axle_height=0.387):
        if self.calib is not None:
            return self.calib

        camera_car_calib_file = Path(os.path.join(self.anno_folder_path, "camera_car.yaml"))
        lidar_car_calib_file = Path(os.path.join(self.anno_folder_path, "camera_lidar.yaml"))
        assert camera_car_calib_file.exists(), str(camera_car_calib_file) + " not found"
        assert lidar_car_calib_file.exists(), str(lidar_car_calib_file) + " not found"
        self.calib = calibration_hesai.Calibration(camera_car_calib_file, lidar_car_calib_file,
                                                   front_to_rear_axle, rear_axle_height)
        return self.calib

    @staticmethod
    def _read_pcd(pcd_path):
        pc = pypcd.PointCloud.from_path(pcd_path)
        pcd_data = []
        for data in pc.pc_data:
            x, y, z, r = data[0], data[1], data[2], data[3]
            pcd_data.append([x, y, z, r])
        pcd_data = np.array(pcd_data)
        return pcd_data

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)

        return data_dict

    def process_single_scene(self, has_label):
        points = self.get_lidar()
        calib = self.get_calib()
        input_dict = {
            'points': points,
            'frame_id': self.sample_idx,
            'calib': calib,
        }

        data_dict = self.prepare_data(input_dict)
        data_dict['image_shape'] = self.get_image_shape()
        data_dict['frame_idx'] = int(self.sample_idx)

        data_dict['points'] = np.pad(data_dict['points'], ((0, 0), (1, 0)), 'constant', constant_values=0)
        data_dict['voxel_coords'] = np.pad(data_dict['voxel_coords'], ((0, 0), (1, 0)), 'constant',
                                           constant_values=0)
        data_dict['image_shape'] = np.reshape(data_dict['image_shape'], (1, 2))

        for k in data_dict.keys():
            if not isinstance(data_dict[k], np.ndarray):
                data_dict[k] = np.array([data_dict[k]])
        data_dict['batch_size'] = 1
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]

            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            # print("shape", box_dict['pred_boxes'].shape)
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="/home/xunfeizhou/Documents/hesai_demo/pp_hesai_demo_0.yaml", required=False,
                        help='specify the config for evaluation')
    parser.add_argument('--ckpt', type=str, default="/home/xunfeizhou/Documents/hesai_demo/checkpoint_epoch_120.pth", required=False,
                        help='checkpoint to start from')
    parser.add_argument('--anno_folder', type=str, default="/media/xunfeizhou/DATA/data/hesai_testing_data_parking_lot_07082021",
                        required=False, help='annotation folder contains the raw data')
    parser.add_argument('--seq_id', type=int, default=1, required=False, help='sequence id of the raw data file')
    parser.add_argument('--sup_score', type=float, default=0.5,
                        help='detection confidence lower than it would not be visualized')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    np.random.seed(1024)

    return args, cfg


def eval_single_scene(cfg, ckpt_path, anno_folder_path, sequence):
    log_file_name = 'log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file_path = os.path.join(anno_folder_path, log_file_name)

    single_scene_loader = LoadSingleHesaiScene(anno_folder_path=anno_folder_path, sequence_idx=sequence, cfg=cfg)

    data_dict = single_scene_loader.process_single_scene(has_label=False)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=1, logger=None, training=False
    )
    logger = common_utils.create_logger(log_file_path, rank=cfg.LOCAL_RANK)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=False)

    model.cuda()
    model.eval()
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, ret_dict = model(data_dict)
    dt_anno = single_scene_loader.generate_prediction_dicts(
        data_dict, pred_dicts, cfg.CLASS_NAMES)
    return dt_anno


def visualize(anno_folder_path, sequence, dt_anno, score_threshold, cfg):
    hesai_scene = LoadSingleHesaiScene(anno_folder_path=anno_folder_path, sequence_idx=sequence, cfg=cfg)
    pts_cloud = hesai_scene.get_lidar()
    pick = [0, 1, 2, 3]  # index for x, y, z, i
    pts_3d = pts_cloud[:, pick].astype(float)
    visualize_utils.draw_scenes(points=pts_3d[:, 0:3], gt_boxes=dt_anno[0]["boxes_lidar"], gt_labels=dt_anno[0]["name"],
                                plane_coeff=None, scores=dt_anno[0]["score"], score_threshold=score_threshold)
    mlab.show(stop=True)


def main():
    args, cfg = parse_config()
    dt_anno = eval_single_scene(cfg=cfg, ckpt_path=args.ckpt, anno_folder_path=args.anno_folder, sequence=args.seq_id)
    visualize(anno_folder_path=args.anno_folder, sequence=args.seq_id, dt_anno=dt_anno, score_threshold=args.sup_score,
              cfg=cfg)


if __name__ == '__main__':
    main()
