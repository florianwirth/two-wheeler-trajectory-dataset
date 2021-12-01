import numpy as np
import glob, os
import math
from random import getrandbits
import matplotlib
from scipy import interpolate

from scipy.signal import savgol_filter


matplotlib.use("TkAgg")


def make_pose_from_rot_pos(rot_mat, position):
    """calculate pose from rotation matrix and position vector"""
    pose = np.zeros([4, 4])
    pose[:3, :3] = rot_mat
    pose[:3, 3] = position
    pose[3, 3] = 1
    return pose


def calculate_corner_points(pose, scale):
    """"calculate translation of edge point relative to object coordinate system,
    multiply with pose in world space afterwards"""
    translation_of_corner_point = np.identity(4)
    corner_points = np.zeros([2, 2, 2, 3])
    for length in range(2):
        for height in range(2):
            for width in range(2):
                translation_of_corner_point[:3, 3] = [np.power(-1, 1 - length) * scale[0]/2,
                                                      np.power(-1, 1 - height) * scale[1]/2,
                                                      np.power(-1, 1 - width) * scale[2]/2]
                translation_of_corner_point = np.matmul(pose, translation_of_corner_point)
                corner_points[length, width, height, :] = translation_of_corner_point[:3, 3]
    return corner_points


class TxtEntry:
    """one line of .txt equals one obj in one scene in one sequence"""
    def __init__(self, tl, seq_no, scene_no):
        """INIT"""
        """Object valid?"""
        self.valid = False
        """Unique id for this object in this sequence"""
        self.id = 0
        """Unique number for this sequence"""
        self.seq_no = 0
        """Unique number for this scene in the current sequence"""
        self.scene_no = 0
        """Property entries:    0->class: 0=pedestrian. 1=two wheeler, 2=car, 3=truck/bus/tram
                                1-> priority: 0=I have priority, 1=opponent has priority, 2=indifferent
                                2-> direction: 0=my direction, 1=opposite direction, 2=indifferent
                                3-> is parking: 0=is parking, 1=is not parking
                                4-> lane w.r.t to ego lane: 
                                    0=2 to left, 1=1 to left, 2=mine, 3=1 to right, 4=2 to right"""
        self.property_label = np.zeros([5])
        self.quality = 0

        self.appearance = np.zeros([3])

        """position relative to the sensor"""
        self.position = np.zeros([3])
        """euclidean distance from sensor to object"""
        self.distance_to_ego = 0
        """angle from sensor to object, front of ego vehicle is 0"""
        self.angle_to_ego = 0
        """orientation represented as a quaternion"""
        self.orient_q = np.zeros([4])
        """orientation in rodriguez representation"""
        self.rodriguez_rot = np.zeros([4])
        """orientation as rotation matrix"""
        self.rot_mat = np.zeros([3, 3])
        """pose of object"""
        self.pose = np.zeros([4, 4])
        """scale of object: length, height, width"""
        self.scale = np.zeros([3])
        """volume of bounding box"""
        self.volume = 0.
        """corner points of bounding box (8), 
        defined as 3 signs representing the directions the axis are pointing to"""
        self.corner_points = np.zeros([2, 2, 2, 3])
        """6 planes, defining the hull of the box: 
        first index defines axis, second index defines orientation of axis"""
        self.faces = np.zeros([3, 2, 4])
        """12 edges: first index defines axis parallel to edge, 
        second and third define the orientations for the other two axis """
        self.edges = np.zeros([3, 2, 2, 6])

        # check for validity
        if ((tl[10:16] == np.array([0, 0, 0, 0, 0, 0])).any() |
                (tl[13:17] == np.array([1, 0, 0, 0])).any() |
                (tl[13:17] == np.array([1, 1, 1, 1])).any() |
                (tl[17:20] == np.array([0, 0, 0])).any() |
                (tl[17:20] == np.array([10, 10, 10])).any()):
            self.valid = False
        else:
            self.valid = True

            self.id = int(tl[0])
            self.seq_no = seq_no
            self.scene_no = scene_no

            self.property_label = np.array([int(tl[1]), int(tl[2]), int(tl[3]), int(tl[4]), int(tl[5])])
            self.quality = int(tl[6])
            self.appearance = np.array([float(tl[7]), float(tl[8]), float(tl[9])])

            self.position = np.array([float(tl[10]), float(tl[11]), float(tl[12])])
            self.distance_to_ego = np.linalg.norm(np.array([float(tl[10]), float(tl[12])]))
            self.angle_to_ego = np.arctan(float(tl[12]) / float(tl[10]))
            self.orient_q = np.array([float(tl[13]), float(tl[14]), float(tl[15]), float(tl[16])])
            self.rot_mat = self.q_to_rot(self.orient_q)
            self.rodriguez_rot = self.rodriguez(self.orient_q)

            self.scale = np.array([np.abs(float(tl[17])), np.abs(float(tl[18])), np.abs(float(tl[19]))])
            if self.scale.min() < 0.:
                ValueError("scale smaller than zero")
            self.volume = self.scale[0] * self.scale[1] * self.scale[2]

            self.pose = make_pose_from_rot_pos(self.rot_mat, self.position)
            self.rotation_matrix = self.pose[:3, :3]
            self.corner_points = calculate_corner_points(self.pose, self.scale)
            self.planes_of_object()
            self.edges_of_object()
            self.roll = self.roll_angle()
            self.ground_xz = self.ground_xz()
            self.yaw = self.yaw_angle()


    @staticmethod
    def rodriguez(q):
        """calc rotation angle and rotation axis"""
        if np.abs(np.linalg.norm(q) - 1) > 0.001:
            raise ValueError("Quaternion is not a unit quaternion!")

        axis = np.array([q[0] / np.sin(q[3] / 2),
                         q[1] / np.sin(q[3] / 2),
                         q[2] / np.sin(q[3] / 2)])

        axis = (axis / np.linalg.norm(axis)) * np.array([1, 1, -1])

        angle = - 2 * np.arccos(q[3])

        return np.concatenate([axis, [angle]])

    @staticmethod
    def get_third_axis(axis_a, axis_b):
        """return unused axis for rotation"""
        if axis_a == axis_b:
            rand_bool = bool(getrandbits(1))
            if axis_a == 0:
                if rand_bool:
                    return 1
                else:
                    return 2
            elif axis_a == 1:
                if rand_bool:
                    return 0
                else:
                    return 2
            elif axis_a == 2:
                if rand_bool:
                    return 0
                else:
                    return 1
        if (0 != axis_a) & (0 != axis_b):
            return 0
        elif (1 != axis_a) & (1 != axis_b):
            return 1
        elif (2 != axis_a) & (2 != axis_b):
            return 2
        else:
            raise ValueError("Input not in correct range. Only 0, 1, 2 allowed.")

    @staticmethod
    def turn_around_axis(scale, pose, axis):
        if axis == 0:
            rot = np.array([[1, 0, 0, 0],
                            [0, 0, -1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
            temp = scale[1]
            scale[1] = scale[2]
            scale[2] = temp
        elif axis == 1:
            rot = np.array([[0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 0, 1]])
            temp = scale[0]
            scale[0] = scale[2]
            scale[2] = temp
        elif axis == 2:
            rot = np.array([[0, -1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
            temp = scale[1]
            scale[1] = scale[0]
            scale[0] = temp
        else:
            raise ValueError("Axis has invalid value!")
        return np.matmul(pose, rot), scale

    def roll_angle(self):
        """roll angle. right is positive"""

        roll = np.array([self.edges[1, 0, 0, 3],
                         self.edges[1, 0, 0, 4],
                         self.edges[1, 0, 0, 5]])

        return -(math.pi / 2. - np.arccos(np.dot(roll, np.array([0, 1, 0]))))

    def q_to_rot(self, q):
        """Transformation of quaternion to rotation matrix"""

        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]

        r = np.array([[1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                      [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
                      [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]])
        return r

    def yaw_angle(self):
        """yaw angle. right is positive"""

        head = np.array([self.edges[2, 1, 1, 3],
                         self.edges[2, 1, 1, 4],
                         self.edges[2, 1, 1, 5]])

        if np.arccos(np.dot(head, np.array([0, 0, -1]))) > math.pi/2:
            return np.arccos(np.dot(head, np.array([-1, 0, 0])))
        else:
            return -np.arccos(np.dot(head, np.array([-1, 0, 0])))

    def ground_xz(self):
        """middle of lower surface projected on ground plane"""
        x = np.mean(self.corner_points[:, :, 0, 0])
        z = np.mean(self.corner_points[:, :, 0, 2])
        return np.array([x, z])

    def verify_orientation(self, min_dimension, max_dimension):
        return self.scale.argmin() == min_dimension & self.scale.argmax() == max_dimension

    def unify_orientation(self):
        """transform in a unified way orientation: y upwards/downwards, x to the front/back,
        pedestrian: long dimension: y, short dimension: z,
        two wheeler: long dimension: x, short dimension: z,
        vehicle: long dimension: x, short dimension: y,
        truck: long dimension: x, short: z"""
        if self.property_label[0] == 0:
            """pedestrian"""
            min_axis = 1
            max_axis = 2
        elif self.property_label[0] == 1:
            """two wheeler"""
            min_axis = 2
            max_axis = 0
        elif self.property_label[0] == 2:
            """car"""
            min_axis = 1
            max_axis = 0
        elif self.property_label[0] == 3:
            """truck"""
            min_axis = 2
            max_axis = 0
        else:
            raise ValueError("Invalid property_label[0]")

        pose = self.pose
        scale = self.scale
        if scale.argmin() != min_axis:
            turn_axis = self.get_third_axis(min_axis, scale.argmin())
            pose, scale = self.turn_around_axis(scale, pose, turn_axis)
            if scale.argmin() == min_axis:
                pose = self.pose
                scale = self.scale
        if scale.argmax() != max_axis:
            pose, scale = self.turn_around_axis(scale, pose, min_axis)
            if (scale.argmax() == max_axis) | (scale.argmin() == min_axis):
                self.pose = pose
                self.scale = scale
            else:
                """Debug output"""
                raise ValueError("Function isn't able to rotate COS in the demanded way!")

    def make_corner_points(self):
        """computes the global position of the corner points of the object
        0 defines the negatively defined side of the object (rear, right, bottom)
        1 defines the positively defined side of the object (front, left, top)"""
        for length in range(2):
            for width in range(2):
                for height in range(2):
                    local_position = np.array([np.power(-1, 1 - length) * self.scale[0] / 2,
                                               np.power(-1, 1 - width) * self.scale[1] / 2,
                                               np.power(-1, 1 - height) * self.scale[2] / 2])
                    local_transform = np.array(np.concatenate([self.rotation_matrix, local_position]),
                                               [0, 0, 0, 1])
                    global_transform = np.matmul(self.pose, local_transform)
                    self.corner_points[length, width, height, :] = global_transform[:3, 3]

    @staticmethod
    def point_above_face(p, a):
        """Checks if p is above face a, above means n is pointing in p's direction"""
        if (not a.shape == [4, 1]) & (not p.shape == [3, 1]):
            ValueError("Invalid input in 'point_above_face'.")
        n = a[:3]
        d = a[3]
        r = n
        s = p

        x_a = (d - s[0] * n[0] - s[1] * n[1] - s[2] * n[2]) / (r[0] * n[0] + r[1] * n[1] + r[2] * n[2])

        return x_a < 0

    def give_n_d(self, points):
        """calculated normal vector n and pistance d, given 4 corner points of an object and sign of normal vector
         n is in the middle of the face, pointing outside, distance to center should be one component in 'scale'"""

        c = (points[1, 0, :] + points[0, 1, :]) / 2
        n = np.cross(points[1, 1, :] - points[1, 0, :], points[1, 1, :] - points[0, 1, :])
        n = n / (np.sqrt(np.dot(n, n)))
        d = np.dot(n, c)
        if self.point_above_face(self.position, np.concatenate([n, [d]])):
            """if n is pointing to the inside of the box, turn the normal vector around"""
            n = -n
            d = np.dot(n, c)

        return np.concatenate([n, [d]])

    def planes_of_object(self):
        """calculates normal vector n of face and its distance d to the origin. 3 defines the number if axis,
        2, whether it's the upper or lower face, and 4 is the concat length of n + d"""

        self.faces[0, 1, :] = self.give_n_d(self.corner_points[1, :, :, :])
        self.faces[0, 0, :] = self.give_n_d(self.corner_points[0, :, :, :])

        self.faces[1, 1, :] = self.give_n_d(self.corner_points[:, 1, :, :])
        self.faces[1, 0, :] = self.give_n_d(self.corner_points[:, 0, :, :])

        self.faces[2, 1, :] = self.give_n_d(self.corner_points[:, :, 1, :])
        self.faces[2, 0, :] = self.give_n_d(self.corner_points[:, :, 0, :])

    def edges_of_object(self):
        """calculates support vector s and directional vector r, concat it to array of length 6
        s is in the middle of the edge and r points in the positive direction
        it takes one axis and 2 sides of the object to explicitly define one edge.
        'axis' states the coordinate axis parallel to the edges currently parametrized
        'side_' 1 & 2 are the remaining axis. 0 means we use the negative plane of this axis,
        1 means we use the positive plane of this axis"""

        for axis in range(3):
            for side_1 in range(2):
                for side_2 in range(2):
                    if axis == 0:
                        self.edges[axis, side_1, side_2, :3] = \
                            (self.corner_points[side_1, side_2, 1, :] + self.corner_points[side_1, side_2, 0, :]) / 2
                        r = \
                            self.corner_points[side_1, side_2, 1, :] - self.corner_points[side_1, side_2, 0, :]
                        self.edges[axis, side_1, side_2, 3:] = r / np.linalg.norm(r)
                    elif axis == 1:
                        self.edges[axis, side_1, side_2, :3] = \
                            (self.corner_points[1, side_1, side_2, :] + self.corner_points[0, side_1, side_2, :]) / 2
                        r = \
                            self.corner_points[1, side_1, side_2, :] - self.corner_points[0, side_1, side_2, :]
                        self.edges[axis, side_1, side_2, 3:] = r / np.linalg.norm(r)
                    elif axis == 2:
                        self.edges[axis, side_1, side_2, :3] = \
                            (self.corner_points[side_1, 1, side_2, :] + self.corner_points[side_1, 0, side_2, :]) / 2
                        r = \
                            self.corner_points[side_1, 1, side_2, :] - self.corner_points[side_1, 0, side_2, :]
                        self.edges[axis, side_1, side_2, 3:] = r / np.linalg.norm(r)
                    else:
                        ValueError("Axis larger than 3 which is not possible!")


class LabelContainer:
    seq = []
    seq_valid = []

    def __init__(self, path_to_results):
        """Initialize label container"""
        self.seq = []
        self.seq_valid = []
        os.chdir(path_to_results)

        for root, dirs, files in sorted(os.walk("labels", topdown=True)):
            if files.__len__() != 0:
                print(root)
                txt_list = sorted(glob.glob(root + "/*.txt"))
                sequence = []
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # plt.grid()
                for path_to_txt in txt_list:
                    scene = []
                    #print(path_to_txt)
                    text_file = open(path_to_txt, "r")
                    lines = text_file.readlines()
                    for line in lines:
                        split_line = line.split(" ")
                        values = np.zeros([split_line.__len__()])
                        i = 0
                        for string in split_line:
                            values[i] = float(string)
                            i += 1
                        txt_entry = TxtEntry(values, sequence.__len__() - 1, scene.__len__() - 1)

                        scene.append(txt_entry)
                    sequence.append(scene)

                self.seq.append(sequence)
                # plt.show()

        os.chdir(os.path.join(os.path.dirname(__file__), '.'))

        os.chdir(path_to_results)
        for root, dirs, files in sorted(os.walk("validation", topdown=True)):
            if files.__len__() != 0:
                #print(root)
                txt_list = sorted(glob.glob(root + "/*.txt"))
                sequence = []
                for path_to_txt in txt_list:
                    scene = []
                    #print(path_to_txt)
                    text_file = open(path_to_txt, "r")
                    lines = text_file.readlines()
                    for line in lines:
                        split_line = line.split(" ")
                        values = np.zeros([split_line.__len__()])
                        i = 0
                        for string in split_line:
                            values[i] = float(string)
                            i += 1
                        txt_entry = TxtEntry(values, sequence.__len__() - 1, scene.__len__() - 1)
                        scene.append(txt_entry)
                    sequence.append(scene)
                    self.seq_valid.append(sequence)

        os.chdir(os.path.join(os.path.dirname(__file__), '.'))

    def simplify(self, track_idx, obj_idx):
        """only give relevant quantities for two wheelers"""
        roll = []
        yaw_label = []
        x = []
        y = []
        v = []
        euler = np.zeros([self.seq[track_idx].__len__(), 3])
        for i, _ in enumerate(self.seq[track_idx]):
            roll.append(self.seq[track_idx][i][obj_idx].roll)
            yaw_label.append(self.seq[track_idx][i][obj_idx].yaw)
            x.append(self.seq[track_idx][i][obj_idx].ground_xz[1])
            y.append(self.seq[track_idx][i][obj_idx].ground_xz[0])


        x_interp = np.array(x)
        y_interp = np.array(y)
        t = np.linspace(0,  (x_interp.shape[0] - x_interp.shape[0] % 2) / 2, x_interp.shape[0])
        f_x = interpolate.interp1d(t, x_interp, kind="cubic")
        f_y = interpolate.interp1d(t, y_interp, kind="cubic")

        freq = 1000
        t_new = np.linspace(0., (x_interp.shape[0] - x_interp.shape[0] % 2) / 2, int((x_interp.shape[0] - x_interp.shape[0] % 2) / 2 * freq + 1))
        x_new = f_x(t_new)
        y_new = f_y(t_new)
        d = np.sqrt(np.power(np.gradient(x_new, edge_order=2), 2) + np.power(np.gradient(y_new, edge_order=2), 2))

        d_x_new = np.gradient(x_new)
        d_y_new = np.gradient(y_new)

        yaw = np.angle(d_y_new - d_x_new * 1j)

        yaw_new = []

        for i, e in enumerate(yaw):
            if i == 0:
                yaw_new.append(e)
                continue
            elif e - yaw_new[i - 1] < -math.pi:
                yaw_new.append(yaw_new[i - 1] + (2 * math.pi + e - yaw_new[i - 1]))
            elif e - yaw_new[i - 1] > math.pi:
                yaw_new.append(yaw_new[i - 1] - (2 * math.pi - e + yaw_new[i - 1]))
            else:
                yaw_new.append(yaw_new[i - 1] + (e - yaw_new[i - 1]))
        for i, e in enumerate(yaw_new):
            if i == 0:
                yaw_new[i] = e
                continue
            elif e - yaw_new[i - 1] < -math.pi:
                yaw_new[i] = (yaw_new[i - 1] + (2 * math.pi + e - yaw_new[i - 1]))
            elif e - yaw_new[i - 1] > math.pi:
                yaw_new[i] = (yaw_new[i - 1] - (2 * math.pi - e + yaw_new[i - 1]))
            else:
                yaw_new[i] = (yaw_new[i - 1] + (e - yaw_new[i - 1]))
        for i, e in enumerate(yaw_new):
            if i == 0:
                yaw_new[i] = e
                continue
            elif e - yaw_new[i - 1] < -math.pi:
                yaw_new[i] = (yaw_new[i - 1] + (2 * math.pi + e - yaw_new[i - 1]))
            elif e - yaw_new[i - 1] > math.pi:
                yaw_new[i] = (yaw_new[i - 1] - (2 * math.pi - e + yaw_new[i - 1]))
            else:
                yaw_new[i] = (yaw_new[i - 1] + (e - yaw_new[i - 1]))

        yaw = savgol_filter(np.array(yaw_new), 2501, 2)

        f_v_interp = interpolate.interp1d(t_new, d * freq * 3.6, kind="cubic")

        v = f_v_interp(t)
        yaw_ = np.gradient(yaw[::501], edge_order=2) * 2.
        yaw__ = np.gradient(yaw_, edge_order=2) * 2.

        roll_ = np.gradient(roll, edge_order=2) * 2.
        roll__ = np.gradient(roll_, edge_order=2) * 2.

        return {"yaw__": yaw__,
                "yaw_": yaw_,
                "yaw": yaw[::501],
                "a": np.gradient(v),
                "v": v,
                "x": x,
                "y": y,
                "roll__": roll__,
                "roll_": roll_,
                "roll": roll}


if __name__ == "__main__":
    sut_labels = LabelContainer("bike_paper")
