from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common as kitti
import os
import cv2
import pickle
import numpy as np


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


class Object3D:
    def __init__(self, single_frame_annotation, index):
        # extract label, truncation, occlusion
        self.type = single_frame_annotation['name'][index]  # 'Car', 'Pedestrian', ...
        self.truncation = single_frame_annotation['truncated'][index]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            single_frame_annotation['occluded'][index]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = single_frame_annotation['alpha'][index]  # object observation angle [-pi..pi]

        # extract 3d bounding box information
        self.h = single_frame_annotation['dimensions'][index, 1]  # box height
        self.w = single_frame_annotation['dimensions'][index, 2]  # box width
        self.l = single_frame_annotation['dimensions'][index, 0]  # box length (in meters)
        self.t = single_frame_annotation['location'][index]  # location (x,y,z) in camera coord.
        self.ry = single_frame_annotation['rotation_y'][
            index]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.score = single_frame_annotation['score'][index]


class SingleImg3DBboxDrawer:
    def __init__(self, single_frame_gt_annotation, single_frame_dt_annotation, frame_id,
                 img_folder_path="/home/xunfeizhou/Documents/kitti/training/image_2",
                 calib_folder_path="/home/xunfeizhou/Documents/kitti/training/calib"):
        # open image and calib file two times if drawing both gt and dt bbox
        img_file_name = str(frame_id).zfill(6) + ".png"
        calib_file_name = str(frame_id).zfill(6) + ".txt"
        img_path = os.path.join(img_folder_path, img_file_name)
        calib_path = os.path.join(calib_folder_path, calib_file_name)
        self.img = cv2.imread(img_path)
        self.gt_anno = single_frame_gt_annotation
        self.dt_anno = single_frame_dt_annotation

        for line in open(calib_path):
            if 'P2:' in line:
                cam_to_img = line.strip().split(' ')
                cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
                self.proj_mat = np.reshape(cam_to_img, (3, 4))

    def draw_gt_3d_bbox(self):
        """
        draw 3d bbox of ground truth data
        """
        cnt = self.gt_anno['name'].shape[0]
        for obj_id in range(cnt):
            obj = Object3D(self.gt_anno, obj_id)
            # print("ground truth location is ", obj.t)
            if obj.type != "DontCare":
                box3d_pts_2d, _ = self.compute_box_3d(obj)
                self.img = self.draw_projected_box3d(self.img, box3d_pts_2d, color=(255, 0, 0))
                self.img = self.draw_label(self.img, obj, box3d_pts_2d, color=(255, 0, 0))
        return

    def draw_dt_3d_bbox(self, object_type=None, suppressed_score=0.0, highlighted_id=-1):
        """
        draw 3d bbox of detection results
        Args:
            object_type:list of int 0 'Car', 1: 'Pedestrian', 2: "Cyclist"
            suppressed_score:
            highlighted_id: this bbox will be highlight
        """
        if object_type is None:
            object_type = [0, 1, 2]
        types = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        cnt = self.dt_anno['name'].shape[0]

        for obj_id in range(cnt):
            obj = Object3D(self.dt_anno, obj_id)
            # print("detection location is ", obj.t)
            if obj.type in types and types[obj.type] in object_type and obj.score >= suppressed_score:
                box3d_pts_2d, _ = self.compute_box_3d(obj)
                is_highlight = False
                if 0 <= highlighted_id == obj_id:
                    is_highlight = True
                if not is_highlight:
                    self.img = self.draw_projected_box3d(self.img, box3d_pts_2d, color=(0, 0, 0))
                else:
                    self.img = self.draw_projected_box3d(self.img, box3d_pts_2d, color=(0, 0, 255))
                self.img = self.draw_label(self.img, obj, box3d_pts_2d, color=(0, 0, 0))
        return

    @staticmethod
    def draw_label(img, obj, box3d_pts_2d, color=(255, 0, 0)):
        """
        put category and confidence on top of 3d bbox
        """
        try:
            score = round(obj.score, 2)
            if score == 0:
                score = "GT"
            else:
                score = str(score)
            txt = str(obj.type)[:3] + " " + score

            font = cv2.FONT_HERSHEY_SIMPLEX
            pos = (int(box3d_pts_2d[0, 0]), int(box3d_pts_2d[0, 1]))
            font_scale = 0.5
            thickness = 1
            img = cv2.putText(img, txt, pos, font,
                              font_scale, color, thickness, cv2.LINE_AA)
        except TypeError:
            pass
        return img

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

    @staticmethod
    def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
        """ Draw 3d bounding box in image
            qs: (8,3) array of vertices for the 3d box in following order:
                1 -------- 0
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7
        """
        try:
            qs = qs.astype(np.int32)
            for k in range(0, 4):
                # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
                i, j = k, (k + 1) % 4
                # use LINE_AA for opencv3
                # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
                cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
                i, j = k + 4, (k + 1) % 4 + 4
                cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

                i, j = k, k + 4
                cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        except AttributeError:
            print("AttributeError in draw_projected_box3d function")
            print(qs)
            pass
        return image

    def get_2d_bbox(self):
        cnt = self.dt_anno['name'].shape[0]
        res = []
        for obj_id in range(cnt):
            obj = Object3D(self.dt_anno, obj_id)
            box3d_pts_2d, _ = self.compute_box_3d(obj)

            try:
                x0, y0, x1, y1 = 1e6, 1e6, 0, 0
                qs = box3d_pts_2d.astype(np.int32)
                for i in range(8):
                    x0 = min(x0, qs[i, 0])
                    x1 = max(x1, qs[i, 0])
                    y0 = min(y0, qs[i, 1])
                    y1 = max(y1, qs[i, 1])

            except AttributeError:
                x0, y0, x1, y1 = 0, 0, 0, 0

            res.append([x0, y0, x1, y1])

        return res



def main():
    gt_path = "/home/xunfeizhou/Documents/data_object_label_2/training/label_2"
    gt_split_file = "/home/xunfeizhou/PycharmProjects/OpenPCDet/data/kitti/ImageSets/val.txt"
    val_image_ids = _read_imageset_file(gt_split_file)
    gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

    with open('../../inference_results/result.pkl', 'rb') as f:
        dt_data = pickle.load(f)

    frame_seq = 0
    bbox_drawer = SingleImg3DBboxDrawer(gt_annos[frame_seq], dt_data[frame_seq],
                                        val_image_ids[frame_seq])
    bbox_drawer.draw_gt_3d_bbox()
    # bbox_drawer.draw_dt_3d_bbox()
    cv2.imshow("frame", bbox_drawer.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()