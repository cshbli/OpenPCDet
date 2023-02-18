import datetime
import os
import pickle
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from thop import profile
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.models import build_network
from pcdet.models import load_data_to_gpu
from pcdet.utils import calibration_kitti
from pcdet.utils import common_utils, box_utils
from skimage import io
from tools.visual_utils.draw_3d_bbox_bev import SingleBevBboxDrawer
from tools.visual_utils.draw_3d_bbox_image import SingleImg3DBboxDrawer


class LoadSingleKittiScene:
    def __init__(self, anno_folder_path, sequence_idx, cfg):
        self.anno_folder_path = anno_folder_path
        self.sample_idx = str(sequence_idx).zfill(6)
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

    def get_lidar(self):
        lidar_file = Path(os.path.join(self.anno_folder_path, self.sample_idx + ".bin"))
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self):
        img_file = Path(os.path.join(self.anno_folder_path, self.sample_idx + ".png"))
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_calib(self):
        calib_file = Path(os.path.join(self.anno_folder_path, self.sample_idx + ".txt"))
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    # @staticmethod
    # def get_fov_flag(pts_rect, img_shape, calib):
    #     """
    #     Args:
    #         pts_rect:
    #         img_shape:
    #         calib:
    #
    #     Returns:
    #
    #     """
    #     pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    #     val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    #     val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    #     val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    #     pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    #
    #     return pts_valid_flag

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

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def process_single_scene(self, has_label):
        points = self.get_lidar()
        calib = self.get_calib()

        img_shape = self.get_image_shape()
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]
        # left fill 0 as batch size
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
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for evaluation')
    parser.add_argument('--ckpt', type=str, default=None, required=True, help='checkpoint to start from')
    parser.add_argument('--anno_folder', type=str, default=None, required=True, help='annotation folder contains the raw data')
    parser.add_argument('--seq_id', type=int, default=None, required=True, help='sequence id of the raw data file')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    np.random.seed(1024)

    return args, cfg


def eval_single_scene(cfg, ckpt_path, anno_folder_path, sequence):

    log_file_name = 'log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file_path = os.path.join(anno_folder_path, log_file_name)

    single_scene_loader = LoadSingleKittiScene(anno_folder_path=anno_folder_path, sequence_idx=sequence, cfg=cfg)

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
        macs, params = profile(model, inputs=(data_dict,))
        print("macs, params", macs, params)
        pred_dicts, ret_dict = model(data_dict)
        # for i in range(100):
        #     start_time = time.time()
        #     pred_dicts, ret_dict = model(data_dict)
        #     torch.cuda.synchronize()
        #     time_taken = time.time() - start_time
        #     print("Run-Time: %.4f s" % time_taken)
    dt_anno = single_scene_loader.generate_prediction_dicts(
        data_dict, pred_dicts, cfg.CLASS_NAMES,
        output_path=None)
    with open("./dt_anno.pkl", 'wb') as handle:
        pickle.dump(dt_anno, handle)


def visualize(sequence, anno_folder_path):
    sequence = str(sequence).zfill(6)
    with open("./dt_anno.pkl", 'rb') as handle:
        dt_anno = pickle.load(handle)

    object_type = [0, 1, 2]
    suppressed_score = 0.5
    cam_bbox_drawer = SingleImg3DBboxDrawer(single_frame_gt_annotation=None, single_frame_dt_annotation=dt_anno[0],
                                            frame_id=sequence, img_folder_path=anno_folder_path,
                                            calib_folder_path=anno_folder_path)
    cam_bbox_drawer.draw_dt_3d_bbox(object_type=object_type, suppressed_score=suppressed_score, highlighted_id=-1)

    bev_bbox_drawer = SingleBevBboxDrawer(single_frame_gt_annotation=None, single_frame_dt_annotation=dt_anno[0],
                                          frame_id=sequence,
                                          lidar_folder_path=anno_folder_path,
                                          calib_folder_path=anno_folder_path)
    bev_bbox_drawer.generate_bev_base_image()
    bev_bbox_drawer.draw_dt_3d_bbox(object_type=object_type, suppressed_score=suppressed_score, highlighted_id=-1)

    bev_img = cv2.resize(bev_bbox_drawer.bev_img, (0, 0), fx=1.5, fy=1.5)
    height = bev_img.shape[0]
    width = height / cam_bbox_drawer.img.shape[0] * cam_bbox_drawer.img.shape[1]
    width = int(width)
    cam_img = cv2.resize(cam_bbox_drawer.img, dsize=(width, height))

    vis = np.concatenate((bev_img, cam_img), axis=1)
    vis = cv2.resize(vis, (0, 0), fx=0.7, fy=0.7)

    cv2.imshow("frame", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args, cfg = parse_config()
    eval_single_scene(cfg=cfg, ckpt_path=args.ckpt, anno_folder_path=args.anno_folder, sequence=args.seq_id)
    # visualize(args.seq_id, args.anno_folder)


if __name__ == '__main__':
    main()
    # python eval_single_anno.py --cfg_file ../output/single_anno/pp_cp_official.yaml --ckpt ../output/single_anno/checkpoint_epoch_80.pth --anno_folder ../output/single_anno --seq_id 6

