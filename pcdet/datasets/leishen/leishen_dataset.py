import copy
import pickle
import csv
from pathlib import Path
import numpy as np
from skimage import io
from pypcd import pypcd
from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import object3d_leishen, calibration_leishen, box_utils, common_utils

# modified from robosense dataset
class LeishenDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.leishen_infos = []
        self.calib = None
        self.include_leishen_data(self.mode)

    def include_leishen_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Leishen dataset')
        leishen_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                # import random
                # infos = random.sample(infos, 1)
                # infos = infos[3624:3684]
                leishen_infos.extend(infos)

        self.leishen_infos.extend(leishen_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Leishen dataset: %d' % (len(leishen_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        print("working on " + str(self.root_split_path))

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        print("total file number is ", len(self.sample_id_list))

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'lslidar_2' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_metadata(self, idx):
        lidar_file = self.root_split_path / 'lslidar_2' / ('%s.bin' % idx)
        assert lidar_file.exists(), str(lidar_file) + " not found"
        return str(lidar_file)

    def get_image_shape(self, idx):
        # img_idx 1-2413: 1280*720; img_idx 2414-5000: 1920*540
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        # print(idx, str(label_file))
        return object3d_leishen.get_objects_from_label(label_file)

    def get_calib(self):
        self.calib = calibration_leishen.Calibration()
        return self.calib

    def get_road_plane(self, idx):
        return None

    # no test data from leishen
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.leishen_infos[0].keys():
            return None, {}
        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()
            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])
            return ap_result_str, ap_dict

        from ...datasets.leishen.leishen_eval import get_leishen_eval_result_kitti_style
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.leishen_infos]

        # add following transformation to use lidar coordinate system for evaluation
        from ..kitti import kitti_utils
        map_name_to_kitti = {
            'Car': 'Car',
            'Truck': 'Truck',
            'Bus': 'Bus',
            'Non_motor_vehicles': 'Non_motor_vehicles',
            'Pedestrians': 'Pedestrians'
        }
        kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
        kitti_utils.transform_annotations_to_kitti_format(
            eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
            info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
        )

        ap_result_str, ap_dict = get_leishen_eval_result_kitti_style(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        raise NotImplementedError

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        import pickle

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('leishen_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        print("work on", info_path)
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

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
            # print(pred_boxes_img)
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

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.leishen_infos) * self.total_epochs

        return len(self.leishen_infos)

    def __getitem__(self, index):
        np.set_printoptions(suppress=True)
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.leishen_infos)

        info = copy.deepcopy(self.leishen_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = self.get_calib()

        img_shape = info['image']['image_shape']
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='Unknown')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
            })

            # print("gt", input_dict["gt_boxes"])
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        data_dict['frame_idx'] = int(sample_idx)
        return data_dict

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info

            calib = self.get_calib()
            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.linalg.inv(calib.c2v_ext)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            info['calib'] = calib_info

            lidar_file_path = self.get_metadata(sample_idx)
            info['lidar_file_path'] = lidar_file_path

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = dict()
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['score'] = np.array([-1.0 for _ in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'Unknown'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                l = np.array([obj.l for obj in obj_list]).reshape(num_gt, 1)
                h = np.array([obj.h for obj in obj_list]).reshape(num_gt, 1)
                w = np.array([obj.w for obj in obj_list]).reshape(num_gt, 1)
                rots = np.array([obj.ry for obj in obj_list]).reshape(num_gt, 1)

                loc_lidar = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                # gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots], axis=1)
                gt_boxes_lidar = np.concatenate([loc_lidar, w, l, h, rots], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                gt_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(gt_boxes_lidar, calib)
                annotations['dimensions'] = gt_boxes_camera[:, 3:6]
                annotations['location'] = gt_boxes_camera[:, 0:3]
                annotations['rotation_y'] = gt_boxes_camera[:, 6]

                gt_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                    gt_boxes_camera, calib, image_shape=self.get_image_shape(sample_idx)
                )
                annotations['bbox'] = gt_boxes_img
                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def dump_single_scene_anno(self, sample_idx, has_label=True, count_inside_pts=True):
        """
        used for debug
        """
        print('%s sample_idx: %s' % (self.split, sample_idx))
        info = {}
        pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
        info['point_cloud'] = pc_info

        image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
        info['image'] = image_info
        # TODO(Xunfei): no need to repeat read calib
        calib = self.get_calib()

        P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
        R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
        R0_4x4[3, 3] = 1.
        R0_4x4[:3, :3] = calib.R0
        V2C_4x4 = np.linalg.inv(calib.c2v_ext)
        calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
        info['calib'] = calib_info

        lidar_file_path = self.get_metadata(sample_idx)
        info['lidar_file_path'] = lidar_file_path
        if has_label:
            obj_list = self.get_label(sample_idx)
            annotations = dict()
            # name is maintained
            annotations['name'] = np.array([obj.cls_type for obj in obj_list])
            annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
            annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
            annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
            annotations['score'] = np.array([-1.0 for _ in obj_list])
            annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

            num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
            num_gt = len(annotations['name'])
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index'] = np.array(index, dtype=np.int32)

            loc_lidar = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
            l = np.array([obj.l for obj in obj_list]).reshape(num_gt, 1)
            h = np.array([obj.h for obj in obj_list]).reshape(num_gt, 1)
            w = np.array([obj.w for obj in obj_list]).reshape(num_gt, 1)
            rots = np.array([obj.ry for obj in obj_list]).reshape(num_gt, 1)
            gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots], axis=1)
            annotations['gt_boxes_lidar'] = gt_boxes_lidar
            gt_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(gt_boxes_lidar, calib)
            gt_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                gt_boxes_camera, calib, image_shape=self.get_image_shape(sample_idx)
            )
            annotations['bbox'] = gt_boxes_img
            annotations['dimensions'] = gt_boxes_camera[:, 3:6]
            annotations['location'] = gt_boxes_camera[:, 0:3]
            annotations['rotation_y'] = gt_boxes_camera[:, 6]
            info['annos'] = annotations

            if count_inside_pts:
                points = self.get_lidar(sample_idx)
                corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                for k in range(num_objects):
                    flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                    num_points_in_gt[k] = flag.sum()
                annotations['num_points_in_gt'] = num_points_in_gt

        return info


def dump_annos():
    import yaml
    from pathlib import Path
    from easydict import EasyDict

    dataset_cfg = EasyDict(yaml.safe_load(open(
        "./tools/cfgs/dataset_configs/leishen_dataset.yaml")))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset = LeishenDataset(dataset_cfg=dataset_cfg, class_names=['Car', 'Pedestrian'],
                           root_path=ROOT_DIR / 'data' / 'leishen', training=False)
    dataset.set_split("train")
    gt_infos = []
    for idx in range(4):
        x = dataset.dump_single_scene_anno(idx * 100)
        gt_infos.append(x)

    pickle.dump(gt_infos, open("./leishen_infos.pkl", mode='wb'))


def create_leishen_infos(dataset_cfg, class_names, data_path, save_path, workers=8):
    dataset = LeishenDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'
    train_filename = save_path / ('leishen_infos_%s.pkl' % train_split)
    val_filename = save_path / ('leishen_infos_%s.pkl' % val_split)

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    leishen_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(leishen_infos_train, f)
    print('leishen_ info train file is saved to %s' % train_filename)

    dataset.set_split(train_split)
    leishen_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(leishen_infos_val, f)
    print('leishen_ info val file is saved to %s' % val_filename)

    # with open(trainval_filename, 'wb') as f:
    #     pickle.dump(leishen_infos_train + leishen_infos_val, f)
    # print('leishen_ info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # leishen_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(leishen_infos_test, f)
    # print('leishen_ info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_leishen_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_leishen_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Truck', 'Bus', 'Non_motor_vehicles', 'Pedestrians'],
            data_path=ROOT_DIR / 'data' / 'leishen',
            save_path=ROOT_DIR / 'data' / 'leishen'
        )


