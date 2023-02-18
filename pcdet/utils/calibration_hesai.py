import numpy as np
import cv2


class Calibration(object):
    def __init__(self, camera_yaml, lidar_yaml, front_to_rear_axle=0.0, rear_axle_height=0.0):
        """

        Args:
            camera_yaml: path to the yaml file contain camera calibration params
            lidar_yaml: path to the yaml file contain lidar calibration params
            front_to_rear_axle: float
            rear_axle_height: float
        """

        # load camera
        yaml_file = cv2.FileStorage(str(camera_yaml), cv2.FILE_STORAGE_READ)
        size_node = yaml_file.getNode("ImageSize")
        # resolution = (int(size_node.at(0).real()), int(size_node.at(1).real()))
        # distortion = np.squeeze(yaml_file.getNode("DistCoeff").mat(), 0)
        intrinsic = yaml_file.getNode("CameraMat").mat()
        # camera2car = yaml_file.getNode("CameraExtrinsicMat").mat()
        self.P2 = np.zeros((3, 4))
        self.P2[:, :3] = intrinsic
        #self.P2[2:, 3:] = 4.981016000000e-03
        #self.P2[0:, 3:] = 4.575831000000e+01
        #self.P2[1:, 3:] = -3.454157000000e-01
        # 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 4.575831000000e+01
        # 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 -3.454157000000e-01
        # 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.981016000000e-03
        yaml_file.release()
        # load lidar
        yaml_file = cv2.FileStorage(str(lidar_yaml), cv2.FILE_STORAGE_READ)
        # camera to lidar
        self.c2v_ext = yaml_file.getNode("CameraExtrinsicMat").mat()  # 4 by 4
        # # self.c2v = self.c2v_ext[:3, :]  # 3 by 4
        # print(self.c2v.shape)
        c2v_hom_t = np.linalg.inv(self.c2v_ext)

        R0_ext = np.zeros((4, 4), dtype=float)
        np.fill_diagonal(R0_ext, 1)
        v2c_ext = np.dot(c2v_hom_t, np.linalg.inv(R0_ext.T))
        self.R0 = R0_ext[:3, :3]  # 3 by 3
        # self.v2c_ext = np.linalg.inv(self.c2v_ext)  # 4 by 4
        self.V2C = v2c_ext[:3, :]  # 3 by 4
        # print(self.v2c)
        # print(self.c2v_ext)
        yaml_file.release()

    @staticmethod
    def cart_to_hom(pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        camera coordinates to lidar coordinates
        :param pts_rect: (N, 3)
        :return pts_lidar: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)

        pts_lidar = np.dot(pts_rect_hom, self.c2v_ext.T)
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)  # n by 4
        # print("pts_lidar_hom", pts_lidar_hom.shape)
        tmp = np.linalg.inv(self.c2v_ext)
        pts_rect = np.dot(pts_lidar_hom, self.V2C.T)
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return np.dot(pts_lidar_hom, tmp.T)[:, 0:3]

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)   # n by 4
        # print("pts_rect", pts_rect_hom.shape)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)  # n by 3
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth


if __name__ == '__main__':
    front_to_rear_axle=3.838
    rear_axle_height=0.387
    calib = Calibration("/media/xunfeizhou/DATA/LIdar_20000/hesai/camera_car.yaml", "/media/xunfeizhou/DATA/LIdar_20000/hesai/camera_lidar.yaml",
                                               front_to_rear_axle, rear_axle_height)
    positions = np.array([[1, 2, 3]]).astype(float)
    x = calib.rect_to_lidar(positions)
    y = calib.lidar_to_rect(x)
    print(x, y)
    print(calib.P2)
