# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D

from math import radians, cos, sin
import numpy as np

import seaborn as sns

# Code from: https://stackoverflow.com/questions/49097667/how-can-i-rotate-a-3d-array-nxnxn-by-x-degrees-around-x-y-and-z-axes

def rotate(volume, deg_angle, axis):
        d = len(volume)
        h = len(volume[0])
        w = len(volume[0][0])
        min_new_x = 0
        max_new_x = 0
        min_new_y = 0
        max_new_y = 0
        min_new_z = 0
        max_new_z = 0
        new_coords = []
        angle = radians(deg_angle)

        for z in range(d):
            for y in range(h):
                for x in range(w):

                    new_x = None
                    new_y = None
                    new_z = None

                    if axis == "x":
                        new_x = int(round(x))
                        new_y = int(round(y*cos(angle) - z*sin(angle)))
                        new_z = int(round(y*sin(angle) + z*cos(angle)))
                    elif axis == "y":
                        new_x = int(round(z*sin(angle) + x*cos(angle)))
                        new_y = int(round(y))
                        new_z = int(round(z*cos(angle) - x*sin(angle)))
                    elif axis == "z":
                        new_x = int(round(x*cos(angle) - y*sin(angle)))
                        new_y = int(round(x*sin(angle) + y*cos(angle)))
                        new_z = int(round(z))

                    val = volume.item((z, y, x))
                    new_coords.append((val, new_x, new_y, new_z))
                    if new_x < min_new_x: min_new_x = new_x
                    if new_x > max_new_x: max_new_x = new_x
                    if new_y < min_new_y: min_new_y = new_y
                    if new_y > max_new_y: max_new_y = new_y
                    if new_z < min_new_z: min_new_z = new_z
                    if new_z > max_new_z: max_new_z = new_z

        new_x_offset = abs(min_new_x)
        new_y_offset = abs(min_new_y)
        new_z_offset = abs(min_new_z)

        new_width = abs(min_new_x - max_new_x)
        new_height = abs(min_new_y - max_new_y)
        new_depth = abs(min_new_z - max_new_z)

        rotated = np.empty((new_depth + 1, new_height + 1, new_width + 1))
        rotated.fill(0)
        for coord in new_coords:
            val = coord[0]
            x = coord[1]
            y = coord[2]
            z = coord[3]

            if rotated[new_z_offset + z][new_y_offset + y][new_x_offset + x] == 0:
                rotated[new_z_offset + z][new_y_offset + y][new_x_offset + x] = val

        return rotated

def get_volume_views(volume, title=""):

    volume = volume.squeeze().__ge__(0.5)
    volume = rotate(volume, 90, "x")
    volume = rotate(volume, 90, "y")
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.voxels(volume, edgecolor="k")
    ax.voxels(volume)
    # ax.view_init(30, -60)

    if title!="":
        plt.title(title)

    plt.axis('off')
    plt.grid(b=None)
    return fig

