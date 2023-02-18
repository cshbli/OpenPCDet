import numpy as np
import csv
import copy
import pickle

from pypcd import pypcd
from skimage import io

from ..dataset import DatasetTemplate

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import object3d_hesai, calibration_hesai, box_utils, common_utils


class HesaiDataset(DatasetTemplate):
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
        # training or testing
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.pcd_paths, self.img_paths = [], []

        split_dir = self.root_path / 'ImageSets' / (self.split + '.csv')
        with open(split_dir) as split_csv:
            csv_reader = csv.reader(split_csv, delimiter=',', skipinitialspace=True)
            for row in csv_reader:
                self.pcd_paths.append(row[0])
                self.img_paths.append(row[1])
        # concurrency disrupted the sequence        
        self.pcd_paths.sort()
        self.img_paths.sort()

        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.hesai_infos = []
        self.calib = None
        self.include_hesai_data(self.mode)

    def include_hesai_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading HESAI dataset')
        hesai_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                hesai_infos.extend(infos)

        self.hesai_infos.extend(hesai_infos)

        if self.logger is not None:
            self.logger.info('Total samples for HESAI dataset: %d' % (len(hesai_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        print("working on " + str(self.root_split_path))
        
        # re-init file paths
        self.pcd_paths, self.img_paths = [], []
        split_dir = self.root_path / 'ImageSets' / (self.split + '.csv')
        with open(split_dir) as split_csv:
            csv_reader = csv.reader(split_csv, delimiter=',', skipinitialspace=True)
            for row in csv_reader:
                self.pcd_paths.append(row[0])
                self.img_paths.append(row[1])
        # concurrency disrupted the sequence        
        self.pcd_paths.sort()
        self.img_paths.sort()
        print("total file number is ", len(self.pcd_paths))


        # split_dir = self.root_path / 'ImageSets' / (self.split + '.csv')
        # csv split is organized by lidar_path, img_path, they are pre-populated and associated, e.g.,
        # cap_data_1605167682/pcd/hesai_1605167654.059942.pcd, cap_data_1605167682/cam/camera_1_1605167654.20884562.jpeg
        # with open(split_dir) as split_csv:
        #     csv_reader = csv.reader(split_csv, delimiter=',', skipinitialspace=True)
        #     for row in csv_reader:
        #         self.pcd_paths.append(row[0])
        #         self.img_paths.append(row[1])

        self.sample_id_list = [i for i in range(len(self.pcd_paths))]

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / self.pcd_paths[idx]
        assert lidar_file.exists(), str(lidar_file) + " not found"
        return self._read_pcd(lidar_file)

    def get_metadata(self, idx):
        lidar_file = self.root_split_path / self.pcd_paths[idx]
        assert lidar_file.exists(), str(lidar_file) + " not found"
        return str(lidar_file)

    def get_image_shape(self, idx):
        img_file = self.root_split_path / self.img_paths[idx]
        assert img_file.exists(), str(img_file) + " not found"
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_path = self.pcd_paths[idx]
        label_path = label_path[:-3] + "txt"
        label_path = label_path.replace("/pcd", "/annotation")
        label_file = self.root_split_path / label_path
        # print(idx, str(label_file))
        assert label_file.exists(), str(label_file) + " not found"
        return object3d_hesai.get_objects_from_label(label_file)

    def get_calib(self, front_to_rear_axle=3.838, rear_axle_height=0.387):
        if self.calib is not None:
            return self.calib
        camera_car_calib_file = self.root_path / "camera_car.yaml"
        lidar_car_calib_file = self.root_path / "camera_lidar.yaml"
        assert camera_car_calib_file.exists(), str(camera_car_calib_file) + " not found"
        assert lidar_car_calib_file.exists(), str(lidar_car_calib_file) + " not found"
        # raise NotImplementedError
        # not fully implemented yet
        self.calib = calibration_hesai.Calibration(camera_car_calib_file, lidar_car_calib_file,
                                             front_to_rear_axle, rear_axle_height)
        return self.calib

    def get_road_plane(self, idx):
        road_plane_path = self.pcd_paths[idx]
        road_plane_path = road_plane_path[:-3] + "txt"
        road_plane_path = road_plane_path.replace("/pcd", "/planes")
        road_plane_file = self.root_split_path / road_plane_path
        assert road_plane_file.exists(), str(road_plane_file) + " not found"

        with open(road_plane_file, 'r') as f:
            lines = f.readlines()
        plane = []
        for line in lines:
            plane.append(float(line))
        plane = np.array(plane)
        assert plane.shape[0] == 4, road_plane_path + " road plane shape is not correct"
        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.hesai_infos[0].keys():
            return None, {}

        from ...datasets.hesai.hesai_eval import get_hesai_eval_result_kitti_style
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.hesai_infos]
        ap_result_str, ap_dict = get_hesai_eval_result_kitti_style(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    @staticmethod
    def _read_pcd(pcd_path):
        pc = pypcd.PointCloud.from_path(pcd_path)
        pcd_data = []
        for data in pc.pc_data:
            x, y, z, r = data[0], data[1], data[2], data[3]
            pcd_data.append([x, y, z, r])
        pcd_data = np.array(pcd_data)
        return pcd_data

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

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        import pickle

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('hesai_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

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
            return len(self.hesai_infos) * self.total_epochs

        return len(self.hesai_infos)

    def __getitem__(self, index):
        # index = 4
        np.set_printoptions(suppress=True)
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.hesai_infos)

        info = copy.deepcopy(self.hesai_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            a = points.shape[0]
            points = points[fov_flag]
            b = points.shape[0]
            # with open("sample.txt", mode='a') as file:
            #     print(self.pcd_paths[index], b, a, file=file)
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            # annos is a single frame

            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            # x y z l h w
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
            # TODO(Xunfei): no need to repeat read calib
            calib = self.get_calib()

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            # V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            V2C_4x4 = np.linalg.inv(calib.c2v_ext)
            # print("V2C", V2C_4x4)
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
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([-1.0 for _ in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
                
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    # fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    # pts_fov = points[fov_flag]
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
        used to dump annos, use it as follows:

        total = 4
        gt_infos = []
        for idx in range(total):
            x = dataset.process_single_scene_test(idx)
            gt_infos.append(x)

        import pickle

        with open("gt_infos.pkl", "wb") as output_file:
            pickle.dump(gt_infos, output_file)

        Args:
            sample_idx:
            has_label:
            count_inside_pts:

        Returns:

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
        V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
        calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

        info['calib'] = calib_info

        if has_label:
            obj_list = self.get_label(sample_idx)
            annotations = {}
            annotations['name'] = np.array([obj.cls_type for obj in obj_list])
            annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
            annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
            annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
            annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
            annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
            annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
            annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
            annotations['score'] = np.array([-1.0 for _ in obj_list])
            annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

            num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
            num_gt = len(annotations['name'])
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index'] = np.array(index, dtype=np.int32)

            loc = annotations['location'][:num_objects]
            dims = annotations['dimensions'][:num_objects]
            rots = annotations['rotation_y'][:num_objects]
            loc_lidar = calib.rect_to_lidar(loc)
            l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
            loc_lidar[:, 2] += h[:, 0] / 2
            gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
            annotations['gt_boxes_lidar'] = gt_boxes_lidar

            info['annos'] = annotations

            if count_inside_pts:
                points = self.get_lidar(sample_idx)
                calib = self.get_calib(sample_idx)
                pts_rect = calib.lidar_to_rect(points[:, 0:3])

                fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                pts_fov = points[fov_flag]
                
                corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                for k in range(num_objects):
                    flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                    num_points_in_gt[k] = flag.sum()
                annotations['num_points_in_gt'] = num_points_in_gt

        return info


def create_hesai_annos(dataset_cfg, class_names, data_path, save_path, workers=8):
    dataset = HesaiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('hesai_infos_%s.pkl' % train_split)
    val_filename = save_path / ('hesai_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'hesai_infos_trainval.pkl'
    test_filename = save_path / 'hesai_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    # dataset.set_split(train_split)
    # hesai_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    # with open(train_filename, 'wb') as f:
    #     pickle.dump(hesai_infos_train, f)
    # print('hesai info train file is saved to %s' % train_filename)

    # dataset.set_split(val_split)
    # hesai_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    # with open(val_filename, 'wb') as f:
    #     pickle.dump(hesai_infos_val, f)
    # print('hesai info val file is saved to %s' % val_filename)

    # with open(trainval_filename, 'wb') as f:
    #     pickle.dump(hesai_infos_train + hesai_infos_val, f)
    # print('hesai info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # hesai_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(hesai_infos_test, f)
    # print('hesai info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


def dump_annos():
    import yaml
    from pathlib import Path
    from easydict import EasyDict

    dataset_cfg = EasyDict(yaml.safe_load(open(
        "./tools/cfgs/dataset_configs/hesai_dataset.yaml")))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset = HesaiDataset(dataset_cfg=dataset_cfg, class_names=['Car', 'Pedestrian'],
                           root_path=ROOT_DIR / 'data' / 'hesai', training=False)
    dataset.set_split("train")
    gt_infos = []
    for idx in range(4):
        x = dataset.dump_single_scene_anno(idx)
        gt_infos.append(x)

    import pickle

    with open("gt_infos.pkl", "wb") as output_file:
        pickle.dump(gt_infos, output_file)


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_hesai_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
    
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_hesai_annos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'TwoWheeler', 'TrailerTruck', 'Bus'],
            data_path=ROOT_DIR / 'data' / 'hesai',
            save_path=ROOT_DIR / 'data' / 'hesai'
        )

        # type_to_id = {'Car': 1, 'Pedestrian': 2, 'Bicycle': 3, 'Motorcycle': 4,
        #               'Truck': 5, 'Trailer': 6, 'Bus': 7}
    # dump_annos()



