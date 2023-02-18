import os
import math
import cv2
import pickle
import numpy as np
from .draw_3d_bbox_image import Object3D, _read_imageset_file
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common as kitti

TOP_Y_MIN = -30
TOP_Y_MAX = +30
TOP_X_MIN = 0
TOP_X_MAX = 100
TOP_Z_MIN = -3.5
TOP_Z_MAX = 0.6

TOP_X_DIVISION = 0.2
TOP_Y_DIVISION = 0.2
TOP_Z_DIVISION = 0.3


class LidarUtils:
    def __init__(self, lidar_file_path):
        self.data = None
        self.file_path = lidar_file_path
        pass

    def get_bev_data(self):
        self.read_lidar_file()
        bev = self.lidar_data_2_bev()
        return bev

    def read_lidar_file(self):
        dtype = np.float32
        n_vec = 4
        self.data = np.fromfile(self.file_path, dtype=dtype)
        self.data = self.data.reshape((-1, n_vec))
        return

    def lidar_data_2_bev(self):
        """
        convert raw velodyne lidar data to bird's eye view
        """
        # truncate the data within range
        idx = np.where(self.data[:, 0] > TOP_X_MIN)
        self.data = self.data[idx]
        idx = np.where(self.data[:, 0] < TOP_X_MAX)
        self.data = self.data[idx]

        idx = np.where(self.data[:, 1] > TOP_Y_MIN)
        self.data = self.data[idx]
        idx = np.where(self.data[:, 1] < TOP_Y_MAX)
        self.data = self.data[idx]

        idx = np.where(self.data[:, 2] > TOP_Z_MIN)
        self.data = self.data[idx]
        idx = np.where(self.data[:, 2] < TOP_Z_MAX)
        self.data = self.data[idx]

        pxs = self.data[:, 0]
        pys = self.data[:, 1]
        pzs = self.data[:, 2]
        prs = self.data[:, 3]
        qxs = ((pxs - TOP_X_MIN) // TOP_X_DIVISION).astype(np.int32)
        qys = ((pys - TOP_Y_MIN) // TOP_Y_DIVISION).astype(np.int32)
        # qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
        qzs = (pzs - TOP_Z_MIN) / TOP_Z_DIVISION
        quantized = np.dstack((qxs, qys, qzs, prs)).squeeze()

        x0, xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
        y0, yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
        z0, zn = 0, int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)
        height = xn - x0
        width = yn - y0
        channel = zn - z0 + 2

        bev = np.zeros(shape=(height, width, channel), dtype=np.float32)

        for x in range(xn):
            ix = np.where(quantized[:, 0] == x)
            quantized_x = quantized[ix]
            if len(quantized_x) == 0:
                continue
            yy = -x

            for y in range(yn):
                iy = np.where(quantized_x[:, 1] == y)
                quantized_xy = quantized_x[iy]
                count = len(quantized_xy)
                if count == 0:
                    continue
                xx = -y

                bev[yy, xx, zn + 1] = min(1, np.log(count + 1) / math.log(32))
                max_height_point = np.argmax(quantized_xy[:, 2])
                bev[yy, xx, zn] = quantized_xy[max_height_point, 3]

                for z in range(zn):
                    iz = np.where(
                        (quantized_xy[:, 2] >= z) & (quantized_xy[:, 2] <= z + 1)
                    )
                    quantized_xyz = quantized_xy[iz]
                    if len(quantized_xyz) == 0:
                        continue
                    zz = z

                    # height per slice
                    max_height = max(0, np.max(quantized_xyz[:, 2]) - z)
                    bev[yy, xx, zz] = max_height
        return bev


class SingleBevBboxDrawer:
    def __init__(self, single_frame_gt_annotation, single_frame_dt_annotation, frame_id,
                 lidar_folder_path="/media/xunfeizhou/DATA/KITTI/data_object_velodyne/testing/velodyne",
                 calib_folder_path="/home/xunfeizhou/Documents/kitti/training/calib"):
        lidar_file_name = str(frame_id).zfill(6) + ".bin"
        calib_file_name = str(frame_id).zfill(6) + ".txt"
        lidar_file_path = os.path.join(lidar_folder_path, lidar_file_name)

        lidar_util = LidarUtils(lidar_file_path)
        self.bev_data = lidar_util.get_bev_data()

        calib_path = os.path.join(calib_folder_path, calib_file_name)
        for line in open(calib_path):
            # Projection matrix from rect camera coord to image2 coord
            if 'P2:' in line:
                cam_to_img = line.strip().split(' ')
                cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
                self.proj_mat = np.reshape(cam_to_img, (3, 4))
            # Rotation from reference camera coord to rect camera coord
            if "R0_rect" in line or "R_rect" in line:
                ref_to_rec = line.strip().split(' ')
                ref_to_rec = np.asarray([float(number) for number in ref_to_rec[1:]])
                self.ref_to_rec_mat = np.reshape(ref_to_rec, [3, 3])

            # Rigid transform from Velodyne coord to reference camera coord
            if "Tr_velo_to_cam" in line or "Tr_velo_cam" in line:
                velo_to_ref = line.strip().split(' ')
                velo_to_ref = np.asarray([float(number) for number in velo_to_ref[1:]])
                self.velo_to_ref_mat = np.reshape(velo_to_ref, [3, 4])
                self.ref_to_velo_mat = self.inverse_rigid_trans(self.velo_to_ref_mat)

        self.gt_anno = single_frame_gt_annotation
        self.dt_anno = single_frame_dt_annotation
        self.bev_img = None

    def generate_bev_base_image(self):
        top_image = np.sum(self.bev_data, axis=2)
        top_image = top_image - np.min(top_image)
        divisor = np.max(top_image) - np.min(top_image)
        top_image = top_image / divisor * 255
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
        self.bev_img = top_image
        return

    def draw_gt_3d_bbox(self, color=(255, 0, 0)):
        if self.bev_img is None:
            print("call generate_bev_base_image function first to generate base image")
            return

        cnt = self.gt_anno['name'].shape[0]
        for obj_id in range(cnt):
            obj = Object3D(self.gt_anno, obj_id)
            if obj.type != "DontCare":
                # print(obj.type)
                _, box3d_pts_3d = self.compute_box_3d(obj)
                boxes3d = self.project_rect_to_velo(box3d_pts_3d)

                x0 = boxes3d[0, 0]
                y0 = boxes3d[0, 1]
                x1 = boxes3d[1, 0]
                y1 = boxes3d[1, 1]
                x2 = boxes3d[2, 0]
                y2 = boxes3d[2, 1]
                x3 = boxes3d[3, 0]
                y3 = boxes3d[3, 1]
                u0, v0 = self.lidar_to_bev_coords(x0, y0)
                u1, v1 = self.lidar_to_bev_coords(x1, y1)
                u2, v2 = self.lidar_to_bev_coords(x2, y2)
                u3, v3 = self.lidar_to_bev_coords(x3, y3)

                self.bev_img = cv2.line(self.bev_img, (u0, v0), (u1, v1), color, 2, cv2.LINE_AA)
                self.bev_img = cv2.line(self.bev_img, (u1, v1), (u2, v2), color, 2, cv2.LINE_AA)
                self.bev_img = cv2.line(self.bev_img, (u2, v2), (u3, v3), color, 2, cv2.LINE_AA)
                self.bev_img = cv2.line(self.bev_img, (u3, v3), (u0, v0), color, 2, cv2.LINE_AA)
                self.draw_label(obj, u0, v0, color)
        return

    def draw_dt_3d_bbox(self, color=(0, 0, 255), object_type=[0, 1, 2], suppressed_score=0.0,
                        highlighted_id=-1):
        """

        Args:
            color: color of the detection bbox
            object_type: list of int 0 'Car', 1: 'Pedestrian', 2: "Cyclist"
            suppressed_score: score lower than this value will not be shown
            highlighted_id: this bbox will be highlight
        Returns:

        """
        types = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        if self.bev_img is None:
            print("call generate_bev_base_image function first to generate base image")
            return

        cnt = self.dt_anno['name'].shape[0]
        for obj_id in range(cnt):
            obj = Object3D(self.dt_anno, obj_id)
            if obj.type in types and types[obj.type] in object_type and obj.score >= suppressed_score:
                _, box3d_pts_3d = self.compute_box_3d(obj)
                boxes3d = self.project_rect_to_velo(box3d_pts_3d)
                x0 = boxes3d[0, 0]
                y0 = boxes3d[0, 1]
                x1 = boxes3d[1, 0]
                y1 = boxes3d[1, 1]
                x2 = boxes3d[2, 0]
                y2 = boxes3d[2, 1]
                x3 = boxes3d[3, 0]
                y3 = boxes3d[3, 1]
                u0, v0 = self.lidar_to_bev_coords(x0, y0)
                u1, v1 = self.lidar_to_bev_coords(x1, y1)
                u2, v2 = self.lidar_to_bev_coords(x2, y2)
                u3, v3 = self.lidar_to_bev_coords(x3, y3)

                is_highlight = False
                if 0 <= highlighted_id == obj_id:
                    is_highlight = True
                if not is_highlight:
                    self.bev_img = cv2.line(self.bev_img, (u0, v0), (u1, v1), color, 1, cv2.LINE_AA)
                    self.bev_img = cv2.line(self.bev_img, (u1, v1), (u2, v2), color, 1, cv2.LINE_AA)
                    self.bev_img = cv2.line(self.bev_img, (u2, v2), (u3, v3), color, 1, cv2.LINE_AA)
                    self.bev_img = cv2.line(self.bev_img, (u3, v3), (u0, v0), color, 1, cv2.LINE_AA)
                else:
                    self.bev_img = cv2.line(self.bev_img, (u0, v0), (u1, v1), [255, 0, 0], 1, cv2.LINE_AA)
                    self.bev_img = cv2.line(self.bev_img, (u1, v1), (u2, v2), [255, 0, 0], 1, cv2.LINE_AA)
                    self.bev_img = cv2.line(self.bev_img, (u2, v2), (u3, v3), [255, 0, 0], 1, cv2.LINE_AA)
                    self.bev_img = cv2.line(self.bev_img, (u3, v3), (u0, v0), [255, 0, 0], 1, cv2.LINE_AA)
                self.draw_label(obj, u0, v0, color)

        return

    def draw_label(self, obj, u0, v0, color):
        score = round(obj.score, 2)
        if score == 0:
            score = "GT"
        else:
            score = str(score)
        txt = str(obj.type)[:3] + " " + score
        font = cv2.FONT_HERSHEY_SIMPLEX
        pos = (u0, v0)
        font_scale = 0.5
        thickness = 1
        self.bev_img = cv2.putText(self.bev_img, txt, pos, font,
                                   font_scale, color, thickness, cv2.LINE_AA)
        return

    def compute_box_3d(self, obj):
        """ Takes an object and a projection matrix (P) and projects the 3d
            bounding box into the image plane.
            Returns:
                corners_2d: (8,2) array in left image coord.
                corners_3d: (8,3) array in in rect camera coord.
        """
        # compute rotational matrix around yaw axis
        rot_mat = self.roty(obj.ry)

        # 3d bounding box dimensions
        l = obj.l
        w = obj.w
        h = obj.h

        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(rot_mat, np.vstack([x_corners, y_corners, z_corners]))
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
        corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
        corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
        # print 'cornsers_3d: ', corners_3d
        # only draw 3d bounding box for objs in front of the camera
        if np.any(corners_3d[2, :] < 0.1):
            corners_2d = None
            return corners_2d, np.transpose(corners_3d)

        # project the 3d bounding box into the image plane
        corners_2d = self.project_to_image(np.transpose(corners_3d), self.proj_mat)
        # print 'corners_2d: ', corners_2d
        return corners_2d, np.transpose(corners_3d)

    @staticmethod
    def project_to_image(pts_3d, proj_mat):
        """ Project 3d points to image plane.
        Usage: pts_2d = projectToImage(pts_3d, P)
          input: pts_3d:        nx3 matrix
                 proj_mat:      3x4 projection matrix
          output: pts_2d: nx2 matrix
          proj_mat(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
          => normalize projected_pts_2d(2xn)
          <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
              => normalize projected_pts_2d(nx2)
        """
        n = pts_3d.shape[0]
        pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
        # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
        pts_2d = np.dot(pts_3d_extend, np.transpose(proj_mat))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    @staticmethod
    def roty(t):
        """ Rotation about the y-axis. """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def project_rect_to_velo(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_ref_to_rect(self, pts_3d_ref):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(self.ref_to_rec_mat, np.transpose(pts_3d_ref)))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.ref_to_velo_mat))

    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.ref_to_rec_mat), np.transpose(pts_3d_rect)))

    @staticmethod
    def cart2hom(pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    @staticmethod
    def inverse_rigid_trans(input_trans):
        """ Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        """
        inv_tr = np.zeros_like(input_trans)  # 3x4
        inv_tr[0:3, 0:3] = np.transpose(input_trans[0:3, 0:3])
        inv_tr[0:3, 3] = np.dot(-np.transpose(input_trans[0:3, 0:3]), input_trans[0:3, 3])
        return inv_tr

    @staticmethod
    def lidar_to_bev_coords(x, y):
        xn = int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
        yn = int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
        xx = yn - int((y - TOP_Y_MIN) // TOP_Y_DIVISION)
        yy = xn - int((x - TOP_X_MIN) // TOP_X_DIVISION)

        return xx, yy


def main():
    gt_path = "/home/xunfeizhou/Documents/data_object_label_2/training/label_2"
    gt_split_file = "/home/xunfeizhou/PycharmProjects/OpenPCDet/data/kitti/ImageSets/val.txt"
    val_image_ids = _read_imageset_file(gt_split_file)
    gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

    with open('../../inference_results/result.pkl', 'rb') as f:
        dt_data = pickle.load(f)

    frame_seq = 0
    bbox_drawer = SingleBevBboxDrawer(gt_annos[frame_seq], dt_data[frame_seq],
                                      val_image_ids[frame_seq])
    bbox_drawer.generate_bev_base_image()
    bbox_drawer.draw_gt_3d_bbox()
    # bbox_drawer.draw_dt_3d_bbox()
    img = cv2.resize(bbox_drawer.bev_img, (0, 0), fx=1.5, fy=1.5)
    cv2.imshow("BEV", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
