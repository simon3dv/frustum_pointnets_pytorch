from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import nuscenes2kitti_util as utils
import _pickle as pickle
from nuscenes2kitti_object import *
import argparse
import ipdb
import shutil


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds

def vis_label():
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d

    split = 'v1.0-mini'
    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI'),split=split)
    type2color = {'Pedestrian': 0,
                  'Car': 1,
                  'Cyclist': 2}
    sensor_list = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']
    print('Sensor_list:',sensor_list)
    linewidth = 2
    colors = ((0, 0, 255), (255, 0, 0), (155, 155, 155))
    print('linewidth={}'.format(linewidth))
    '''
    -v1.0-mini
        -calib
        -image_CAM_FRONT
        -image_CAM_...
        ...
        -label_CAM_FRONT
        -label_CAM_...
        ...
        -calib
        _LIDAR_TOP
        
        -vis
            -vis2d_CAM_FRONT
            -vis2d_CAM_...
            ...
            -vis3d_CAM_FRONT
            -vis3d_CAM_...
            ...
    '''
    for present_sensor in sensor_list:
        save2ddir = os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI',split,'vis','vis2d_'+present_sensor)
        save3ddir = os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI',split,'vis','vis3d_'+present_sensor)
        if os.path.isdir(save2ddir) == True:
            print('previous save2ddir found. deleting...')
            shutil.rmtree(save2ddir)
        os.makedirs(save2ddir)
        if os.path.isdir(save3ddir) == True:
            print('previous save3ddir found. deleting...')
            shutil.rmtree(save3ddir)
        os.makedirs(save3ddir)

        print('Saving images with 2d boxes to {}...'.format(save2ddir))
        print('Saving images with 3d boxes to {}...'.format(save3ddir))
        for data_idx in tqdm(range(dataset.num_samples)):
            # Load data from dataset
            objects = dataset.get_label_objects(present_sensor,data_idx)
            # objects[0].print_object()
            img = dataset.get_image(present_sensor,data_idx)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #print(('Image shape: ', img.shape))
            pc_velo = dataset.get_lidar(data_idx)[:,0:3]
            calib = dataset.get_calibration(data_idx)
            ''' Show image with 2D bounding boxes '''
            img1 = np.copy(img)  # for 2d bbox
            img2 = np.copy(img)  # for 3d bbox
            for obj in objects:
                if obj.type == 'DontCare': continue
                #if obj.type not in type2color.keys(): continue
                #c = type2color[obj.type]
                c = 0
                cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                              (int(obj.xmax), int(obj.ymax)), colors[c][::-1], 2)

                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib)

                # img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)
                def draw_rect(selected_corners, color):
                    prev = selected_corners[-1]
                    for corner in selected_corners:
                        cv2.line(img2,
                                 (int(prev[0]), int(prev[1])),
                                 (int(corner[0]), int(corner[1])),
                                 color, linewidth)
                        prev = corner

                corners_2d = box3d_pts_2d
                # Draw the sides
                for i in range(4):
                    cv2.line(img2,
                             (int(corners_2d.T[i][0]), int(corners_2d.T[i][1])),
                             (int(corners_2d.T[i + 4][0]), int(corners_2d.T[i + 4][1])),
                             colors[c][::-1], linewidth)

                # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
                draw_rect(corners_2d.T[:4], colors[c][::-1])
                draw_rect(corners_2d.T[4:], colors[c][::-1])

                # Draw line indicating the front
                center_bottom_forward = np.mean(corners_2d.T[0:2], axis=0)
                center_bottom = np.mean(corners_2d.T[[0, 1, 2, 3]], axis=0)
                # center_bottom_forward = np.mean(corners_2d.T[2:4], axis=0)
                # center_bottom = np.mean(corners_2d.T[[2, 3, 7, 6]], axis=0)
                cv2.line(img2,
                         (int(center_bottom[0]), int(center_bottom[1])),
                         (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                         colors[c][::-1], linewidth)

            cv2.imwrite(os.path.join(save2ddir, str(data_idx).zfill(6) + '.jpg'),img1)
            cv2.imwrite(os.path.join(save3ddir, str(data_idx).zfill(6) + '.jpg'),img2)

def demo():
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI'))
    data_idx = 11

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)  # objects = [Object3d(line) for line in lines]
    objects[0].print_object()

    calib = dataset.get_calibration(data_idx)  # utils.Calibration(calib_filename)
    box2d = objects[0].box2d
    xmin, ymin, xmax, ymax = box2d
    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    uvdepth = np.zeros((1, 3))
    uvdepth[0, 0:2] = box2d_center
    uvdepth[0, 2] = 20  # some random depth
    #box2d_center_rect = calib.project_image_to_rect(uvdepth)
    #frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
    #                                box2d_center_rect[0, 0])
    #print('frustum_angle:', frustum_angle)
    img = dataset.get_image(data_idx)  # (370, 1224, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(('Image shape: ', img.shape))
    #pc_velo = dataset.get_lidar(data_idx)[:, 0:3]  # (115384, 3)
    #calib = dataset.get_calibration(data_idx)  # utils.Calibration(calib_filename)

    ## Draw lidar in rect camera coord
    # print(' -------- LiDAR points in rect camera coordination --------')
    # pc_rect = calib.project_velo_to_rect(pc_velo)
    # fig = draw_lidar_simple(pc_rect)
    # raw_input()
    # Draw 2d and 3d boxes on image
    print(' -------- 2D bounding boxes in images --------')
    show_image_with_boxes(img, objects, calib)
    #raw_input()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--vis_label', action='store_true', help='Run vis_label.')
    args = parser.parse_args()

    if args.demo:
        demo()
        exit()
    if args.vis_label:
        vis_label()
        exit()
