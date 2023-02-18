import numpy as np
import json

id_to_merged_id = {"vehicle": "Vehicle",
                   "pedestrian": "Pedestrian",
                   "bicycle": "MotorBicycle",
                   "motorcycle": "MotorBicycle",
                   "tricycle": "Vehicle",
                   "big_vehicle": "Big_vehicle",
                   "huge_vehicle": "Huge_vehicle",
                   "cone": "Cone",
                   "unknown": "Cone"
                   }


def get_objects_from_label(label_file):
    json_data = json.load(open(label_file))
    objects = [Object3d(anno) for anno in json_data["labels"]]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Vehicle': 1, 'Pedestrian': 2, 'MotorBicycle': 3,
                  'Big_vehicle': 4, 'Huge_vehicle': 5, "Cone": 6}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, anno):
        self.cls_type = anno["type"]

        if self.cls_type not in id_to_merged_id.keys():
            print(self.cls_type, "not recognized")
            raise NotImplemented
        else:
            self.cls_type = id_to_merged_id[self.cls_type]

        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = 0.0  # no such data in anno
        #(NOTE):KITTI Occlusion level per class. 0: fully visible, 1: occluded less than 50%, 2: occluded more than 50%, 3: unknown.
        #robosense vis level 1~4 means 25%, 50%, 75%, 100% visibility
        self.occlusion = int(anno["visibility"])
        self.alpha = 0.0  # no such data in anno
        # left, top, right, bottom
        self.box2d = None
        self.h = anno["size"]["z"]
        self.w = anno["size"]["y"]
        self.l = anno["size"]["x"]

        self.loc = np.array((anno["center"]["x"], anno["center"]["y"], anno["center"]["z"]), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)

        self.ry = float(anno["rotation"]["yaw"])
        self.rotations = anno["rotation"]
        self.level = self.get_kitti_obj_level()

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                    % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                       self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str

    def get_kitti_obj_level(self):
        if self.occlusion >=3 and self.dis_to_cam/self.h <= 25:
            return 0  # Easy
        elif self.occlusion >=1 and  self.dis_to_cam/self.h <=36:
            return 1  # Moderate
        return 2 # hard



