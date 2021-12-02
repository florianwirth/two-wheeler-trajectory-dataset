import os
from os import listdir

import dataLoader
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import cm
from matplotlib.collections import LineCollection


class st2w:
    """
    Database class for single-track-two-wheeler dataset to load and visualize the dataset
    """
    class_names = ['bike', 'motorbike', 'scooter']
    variable_names = [
        "yaw__",
        "yaw_",
        "yaw",
        "a",
        "v",
        "x",
        "y",
        "roll__",
        "roll_",
        "roll"]

    def __init__(self, dataroot: str = '../data/', usePointClouds=False):
        self.dataroot = dataroot
        self.labelContainer = dataLoader.LabelContainer(self.dataroot)
        self.labels = self.get_st2w_dict([self.labelContainer])
        if usePointClouds:
            self.pcds = self.get_pointcloud_directories()
        self.bboxes = self.getBboxes()
        print()

    def getBboxes(self):
        bboxes = {}
        for i, classname in enumerate(st2w.class_names):
            bboxes[classname] = []
            for j, _ in enumerate(self.labelContainer.seq[i]):
                corner_points = self.labelContainer.seq[i][j][0].corner_points
                bbox = np.zeros((8, 3))
                bbox[0] = corner_points[0, 0, 0]
                bbox[1] = corner_points[0, 1, 0]
                bbox[2] = corner_points[1, 1, 0]
                bbox[3] = corner_points[1, 0, 0]
                bbox[4] = corner_points[0, 0, 1]
                bbox[5] = corner_points[0, 1, 1]
                bbox[6] = corner_points[1, 1, 1]
                bbox[7] = corner_points[1, 0, 1]
                for k, b in enumerate(bbox):
                    temp = b[1]
                    bbox[k][1] = b[2]
                    bbox[k][2] = temp
                bboxes[classname].append(bbox)
        return bboxes

    def getLabel(self, classname: str, step: int = None):
        result = {}
        if step is None:
            result = self.labels[classname]
        else:
            for variable_name in st2w.variable_names:
                result[variable_name] = self.labels[classname][variable_name][step]
        return result

    @staticmethod
    def get_st2w_dict(container):
        """
        :param container: Label Container
        :return:  st2w dict
        """
        Dict = {}
        for i, classname in enumerate(st2w.class_names):
            Dict.update({classname: container[0].simplify(i, 0)})
        return Dict

    def get_pointcloud_directories(self):
        """
        method for generating the directory path of the point clouds
        :param category: vehicle category ('bike', 'motorbike', 'scooter')
        :return: list of directories
        """
        pcds = {}
        for classname in st2w.class_names:
            label_dir = os.path.join(self.dataroot, 'labels/')
            pcds[classname] = []
            if classname is "bike":
                point_cloud_dir = os.path.join(self.dataroot, 'pcls/RecordedDataBike/pcd/')
                label_dir += '0'
            elif classname is 'motorbike':
                point_cloud_dir = os.path.join(self.dataroot, 'pcls/RecordedDataMotorbike/pcd/')
                label_dir += '1'
            else:
                point_cloud_dir = os.path.join(self.dataroot, 'pcls/RecordedDataScooter/pcd/')
                label_dir += '2'
            label_idxs = []
            for f in listdir(label_dir):
                temp = f.replace('.txt', '')
                temp = temp.replace('label_', '')
                print(temp)
                label_idxs.append(temp)

            for f in listdir(point_cloud_dir):
                temp = f.replace('.pcd', '')
                if label_idxs.__contains__(temp):
                    pcds[classname].append(point_cloud_dir + f)
        print("Ende")

        return pcds

    class Visualizer:
        @staticmethod
        def plot_trajectory_colored(x, y, value, label, colormap, figure_name, save_path=None, min_max=None):
            """
            plot a given trajectory with a attribute colored
            :param x: x-values of trajectory
            :param y: y-values of trajectory
            :param value: value for coloring
            :param label: label for colorbar
            :param colormap: colormap
            :param save_path: path for saving the plot
            :param min_max: [min, max] of value for color
            """
            if min_max is None:
                min_max = [min(value), max(value)]
            norm = plt.Normalize(min_max[0], min_max[1])

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            color_map = cm.get_cmap(colormap, 256)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=color_map, norm=norm)
            lc.set_array(value)
            lc.set_linewidth(1)
            lc.set_joinstyle('round')
            lc.set_capstyle('round')
            line = ax.add_collection(lc)
            ax.axis('equal')
            ax.set_xlabel('x in m')
            ax.set_ylabel('y in m')
            ax.set_title(figure_name)
            cbar = fig.colorbar(line, ax=ax)
            cbar.ax.set_ylabel(label)
            if save_path is not None:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()

        @staticmethod
        def plot_trajectory_velocity(labels, figure_name='Two-wheeler', save_path=None):
            st2w.Visualizer.plot_trajectory_colored(labels['x'],
                                                    labels['y'],
                                                    labels['v'] / 3.6,
                                                    "Velocity in m/s",
                                                    'viridis',
                                                    figure_name,
                                                    save_path)

        @staticmethod
        def plot_trajectory_rollangle(labels, figure_name='Two-wheeler', save_path=None):
            st2w.Visualizer.plot_trajectory_colored(labels['x'],
                                                    labels['y'],
                                                    np.rad2deg(labels['roll']),
                                                    "Roll angle in degree",
                                                    'rainbow',
                                                    figure_name,
                                                    save_path)

        @staticmethod
        def view_pointcloud(pcd_path, bbox):
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd = np.asarray(pcd.points)
            for i, p in enumerate(pcd):
                pcd[i][1] *= -1
            st2w.Visualizer.display_frame_statistics(pcd, bbox)

        axes_limits = [
            [-40, 40],  # [-40, 50],  # X axis range
            [-40, 40],  # [-40, 40],  # Y axis range
            [-6, 6]  # Z axis range
        ]
        axes_str = ['X', 'Y', 'Z']

        @staticmethod
        def draw_box(pyplot_axis, vertices, color='black'):
            """
            Draws a bounding 3D box in a pyplot axis.

            Parameters
            ----------
            pyplot_axis : Pyplot axis to draw in.
            vertices    : Array 8 box vertices containing x, y, z coordinates.
            color       : Drawing color. Defaults to `black`.
            """
            # vertices = vertices[axes, :]
            connections = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
                [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
                [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
            ]
            for connection in connections:
                pyplot_axis.plot(vertices[connection, 0], vertices[connection, 1], vertices[connection, 2], c=color,
                                 lw=2)

        @staticmethod
        def display_frame_statistics(pcd, bbox, points=0.2):
            """
            Displays statistics for a single frame. Draws camera data, 3D plot of the lidar point cloud data and point cloud
            projections to various planes.

            Parameters
            ----------
            pcd         : pointcloud to visualize.
            bbox  : bbox of object
            points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.
            """
            point_size = 0.001 * (1. / points)

            def draw_point_cloud(ax, pcd, title, axes=None, xlim3d=None, ylim3d=None, zlim3d=None):
                """
                Convenient method for drawing various point cloud projections as a part of frame statistics.
                """
                if axes is None:
                    axes = [0, 1, 2]
                print(pcd)
                pcd_temp = []
                for p in pcd:
                    if (st2w.Visualizer.axes_limits[0][1] >= p[0] >= st2w.Visualizer.axes_limits[0][0]) and (
                            st2w.Visualizer.axes_limits[1][1] >= p[1] >= st2w.Visualizer.axes_limits[1][0]) and (
                            st2w.Visualizer.axes_limits[2][1] >= p[2] >= st2w.Visualizer.axes_limits[2][0]):
                        pcd_temp.append(p)
                pcd_temp = np.array(pcd_temp)
                ax.scatter(pcd_temp[:, 0], pcd_temp[:, 1], pcd_temp[:, 2], s=point_size, c='black', cmap='gray')
                ax.set_title(title)
                ax.set_xlabel('{} axis'.format(st2w.Visualizer.axes_str[axes[0]]))
                ax.set_ylabel('{} axis'.format(st2w.Visualizer.axes_str[axes[1]]))
                if len(axes) > 2:
                    ax.set_xlim3d(*st2w.Visualizer.axes_limits[axes[0]])
                    ax.set_ylim3d(*st2w.Visualizer.axes_limits[axes[1]])
                    ax.set_zlim3d(*st2w.Visualizer.axes_limits[axes[2]])
                    ax.set_zlabel('{} axis'.format(st2w.Visualizer.axes_str[axes[2]]))
                else:
                    ax.set_xlim(*st2w.Visualizer.axes_limits[axes[0]])
                    ax.set_ylim(*st2w.Visualizer.axes_limits[axes[1]])
                # User specified limits
                if xlim3d != None:
                    ax.set_xlim3d(xlim3d)
                if ylim3d != None:
                    ax.set_ylim3d(ylim3d)
                if zlim3d != None:
                    ax.set_zlim3d(zlim3d)

                st2w.Visualizer.draw_box(ax, bbox, color='blue')

            # Draw point cloud data as 3D plot
            f2 = plt.figure(figsize=(15, 8))
            ax2 = f2.add_subplot(111, projection='3d')
            draw_point_cloud(ax2, pcd, 'Velodyne scan')
            plt.show()
