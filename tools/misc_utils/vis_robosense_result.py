import mayavi.mlab as mlab
import pickle
import tools.misc_utils.vis_3d_utils as v
from pypcd import pypcd
import numpy as np
import os

def _read_pcd(pcd_path):
    pc = pypcd.PointCloud.from_path(pcd_path)
    pcd_data = []
    for data in pc.pc_data:
        x, y, z, r = data[0], data[1], data[2], data[3]
        pcd_data.append([x, y, z, r])
    pcd_data = np.array(pcd_data)
    return pcd_data

tmp = "/media/xunfeizhou/DATA/robosense/training"
pcd_path = "Crossroads/xiaolukou/pcd/xiaolukou_4195.pcd"
pcd_path = os.path.join(tmp, pcd_path)

pts = _read_pcd(pcd_path)
pts = pts[~np.isnan(pts).any(axis=1)]

path = "/home/xunfeizhou/Documents/result.pkl"
data = pickle.load(open(path, mode="rb"))
v.draw_scenes(points=pts, gt_boxes=data[0]["boxes_lidar"][:20, :], gt_labels=None)
mlab.show(stop=True)
