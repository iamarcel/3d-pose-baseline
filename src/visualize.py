import os
import json

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as img
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import viz
import data_utils


def get_3d_lines(frame):
    I     = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
    J     = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points

    lines = []
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [frame[I[i], j], frame[J[i], j]] ) for j in range(3)]
        lines.append((x, z, -y))

    return lines


def draw_3d_pose(frame, axes=[], ax=None):
    LR    = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    lcolor="#3498db"
    rcolor="#e74c3c"

    lines = get_3d_lines(frame)

    if axes is None or len(axes) == 0:
        for i, line in enumerate(lines):
            axes.append(ax.plot(line[0], line[1], line[2], lw=2, c=lcolor if LR[i] else rcolor)[0])
    else:
        for i, line in enumerate(lines):
            axes[i].set_data(line[0], line[1])
            axes[i].set_3d_properties(line[2])

    return axes



def plot_skeleton(points, points_2d, image_paths):
    if (points.shape[1] != 32 or points.shape[2] != 3):
        raise ValueError("Expected points.shape to be (?, 32, 3), got " + str(points.shape))

    fig = plt.figure()
    ax_2d = fig.add_subplot(131)
    ax = fig.add_subplot(132, projection='3d')
    ax.view_init(18, -70)
    ax.set_aspect(1)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_zlim(-500, 500)
    ax_img = fig.add_subplot(133)
    ax_img.axis('off')

    points_2d = data_utils.process_stacked_hourglass(points_2d)
    viz.show2Dpose(np.reshape(points_2d[0][0], (64,1)), ax_2d)
    axes_3d = draw_3d_pose(points[0], ax=ax)
    ax_img = plt.imshow(img.imread(image_paths[0]), animated=True)

    def update(frame_enumerated):
        img_i, frame = frame_enumerated
        ax_img.set_data(img.imread(image_paths[img_i]))

        ax_2d.clear()
        viz.show2Dpose(np.reshape(points_2d[0][img_i], (64,1)), ax_2d)
        ax_2d.invert_yaxis()

        ax.clear()
        viz.show3Dpose(frame, ax)
        # draw_3d_pose(frame, axes=axes_3d)

    ani = FuncAnimation(fig, update, frames=enumerate(points))

    plt.show()

def image_path(id, i, root, ext):
    return os.path.join(root, id + "-" + str(i) + ext)

def preview_first_clip():
    config = {}
    with open('config.json') as config_file:
        config = json.load(config_file)
    image_root = config['image_root']
    image_extension = config['image_extension']
    clip = config['clips'][0]
    images = [image_path(clip['id'], i + 1, image_root, image_extension) for i in range(clip['end'] - clip['start'])]

    points_2d = np.array(clip['points_2d'])

    plot_skeleton(np.array(clip['points_3d']),
                  points_2d,
                  images)

preview_first_clip()
