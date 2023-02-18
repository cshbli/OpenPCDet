import numpy as np


class Calibration:
    def __init__(self):
        """
        fake calib
        Args:
        """

        intrinsic = [1.9563846147609399e+03, 0., 1.0104833255775920e+03,
                     0., 1.9555614831451201e+03, 5.8116172920355007e+02,
                     0., 0., 1.]
        intrinsic = np.asarray(intrinsic)
        intrinsic = np.reshape(intrinsic, (3, 3))

        self.P2 = np.zeros((3, 4))
        self.P2[:, :3] = intrinsic
        self.P2[2:, 3:] = 0
        self.P2[0:, 3:] = 0
        self.P2[1:, 3:] = 0

        self.c2v_ext = [6.1298591260681287e-17, -3.6643708706555707e-02, 9.9932839377865634e-01, 4.5946389967293955e-01,
                        -9.9978068347484539e-01, 2.0928354823873292e-02, 7.6740793381617370e-04, 3.8477373536537067e-02,
                        -2.0942419883356950e-02, -9.9910922454784445e-01, -3.6635672135693416e-02,
                        -4.0439141512038163e-01,
                        0., 0., 0., 1.]
        self.c2v_ext = np.asarray(self.c2v_ext)
        self.c2v_ext = np.reshape(self.c2v_ext, (4, 4))
        c2v_hom_t = np.linalg.inv(self.c2v_ext)

        R0_ext = np.zeros((4, 4), dtype=float)
        np.fill_diagonal(R0_ext, 1)
        v2c_ext = np.dot(c2v_hom_t, np.linalg.inv(R0_ext.T))
        self.R0 = R0_ext[:3, :3]  # 3 by 3
        self.V2C = v2c_ext[:3, :]  # 3 by 4

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
        pts_rect_hom = self.cart_to_hom(pts_rect)  # n by 4
        # print("pts_rect", pts_rect_hom.shape)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)  # n by 3
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth


if __name__ == '__main__':
    calib = Calibration()
    positions = np.array([[1, 2, 3]]).astype(float)
    x = calib.rect_to_lidar(positions)
    y = calib.lidar_to_rect(x)
    print(x, y)
    print(calib.P2)
