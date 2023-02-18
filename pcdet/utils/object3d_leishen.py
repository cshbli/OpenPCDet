import numpy as np

id_to_cap_id = {   "car": "Car",
                   "truck": "Truck",
                   "bus": "Bus",
                   "non_motor_vehicles": "Non_motor_vehicles",
                   "pedestrians": "Pedestrians"
                }

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Truck': 2, 'Bus': 3,
                  'Non_motor_vehicles': 4, 'Pedestrians': 5}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]

        if self.cls_type not in id_to_cap_id.keys():
            print(self.cls_type, "not recognized")
            raise NotImplemented
        else:
            self.cls_type = id_to_cap_id[self.cls_type]

        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])
        self.alpha = 0.0  # no such data in anno
        # left, top, right, bottom
        self.box2d = None
        self.w = float(label[8])
        self.l = float(label[9])
        self.h = float(label[10])

        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = -float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
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



