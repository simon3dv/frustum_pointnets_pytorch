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
import matplotlib.pyplot as plt
import glob
import time

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

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line widthf
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
        Draw 3d bounding box in image
    Tips:
        KITTI
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        nuScenes
            1 -------- 0
           /|         /|
          5 -------- 4 .
          | |        | |
          . 2 -------- 3
          |/         |/
          6 -------- 7

    '''
    import mayavi.mlab as mlab
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def vis_label(split='v1.0-mini',sensor_list=['CAM_FRONT'],type_whitelist=['Car']):
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple  # , draw_gt_boxes3d
    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI'), split=split)
    type2color = {}
    for i,x in enumerate(type_whitelist):
        type2color[x] = i
    print('type_whitlist:', type_whitelist)
    print('Sensor_list:', sensor_list)
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
        save2ddir = os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI', split, 'vis_label', 'vis2d_' + present_sensor)
        save3ddir = os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI', split, 'vis_label', 'vis3d_' + present_sensor)
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
            objects = dataset.get_label_objects(present_sensor, data_idx)
            # objects[0].print_object()
            img = dataset.get_image(present_sensor, data_idx)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(('Image shape: ', img.shape))
            pc_velo = dataset.get_lidar(data_idx)[:, 0:3]
            calib = dataset.get_calibration(data_idx)
            ''' Show image with 2D bounding boxes '''
            img1 = np.copy(img)  # for 2d bbox
            img2 = np.copy(img)  # for 3d bbox
            for obj in objects:
                if obj.type == 'DontCare': continue
                # if obj.type not in type2color.keys(): continue
                # c = type2color[obj.type]
                c = 0
                cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                              (int(obj.xmax), int(obj.ymax)), colors[c][::-1], 2)

                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, getattr(calib, present_sensor))

                # img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)
                def draw_rect(selected_corners, color):
                    prev = selected_corners[-1]
                    for corner in selected_corners:
                        cv2.line(img2,
                                 (int(prev[0]), int(prev[1])),
                                 (int(corner[0]), int(corner[1])),
                                 color, linewidth)
                        prev = corner

                corners_2d = box3d_pts_2d  # (8,2)
                # Draw the sides
                for i in range(4):
                    cv2.line(img2,
                             (int(corners_2d[i][0]), int(corners_2d[i][1])),
                             (int(corners_2d[i + 4][0]), int(corners_2d[i + 4][1])),
                             colors[c][::-1], linewidth)

                # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
                draw_rect(corners_2d[:4], colors[c][::-1])
                draw_rect(corners_2d[4:], colors[c][::-1])

                # Draw line indicating the front
                center_bottom_forward = np.mean(corners_2d[0:2], axis=0)
                center_bottom = np.mean(corners_2d[[0, 1, 2, 3]], axis=0)
                # center_bottom_forward = np.mean(corners_2d.T[2:4], axis=0)
                # center_bottom = np.mean(corners_2d.T[[2, 3, 7, 6]], axis=0)
                cv2.line(img2,
                         (int(center_bottom[0]), int(center_bottom[1])),
                         (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                         colors[c][::-1], linewidth)

            cv2.imwrite(os.path.join(save2ddir, str(data_idx).zfill(6) + '.jpg'), img1)
            cv2.imwrite(os.path.join(save3ddir, str(data_idx).zfill(6) + '.jpg'), img2)

def vis_pred(split='training', sensor_list = ['CAM_FRONT'], type_whitelist=['Car'], vis_pred_path=None):
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple  # , draw_gt_boxes3d
    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI'), split=split)
    type2color = {}
    for i,x in enumerate(type_whitelist):
        type2color[x] = i
    print('type_whitlist:', type_whitelist)
    print('Sensor_list:', sensor_list)
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
        save2ddir = os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI', split, 'vis_pred', 'vis2d_' + present_sensor)
        save3ddir = os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI', split, 'vis_pred', 'vis3d_' + present_sensor)
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

        filename_list = glob.glob(os.path.join(vis_pred_path,"*.txt"))
        for label_filename in tqdm(filename_list):
            # Load data from dataset
            data_idx = int(label_filename[-10:-4])
            objects = utils.read_label(label_filename)
            # objects[0].print_object()
            img = dataset.get_image(present_sensor, data_idx)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(('Image shape: ', img.shape))
            pc_velo = dataset.get_lidar(data_idx)[:, 0:3]
            calib = dataset.get_calibration(data_idx)
            ''' Show image with 2D bounding boxes '''
            img1 = np.copy(img)  # for 2d bbox
            img2 = np.copy(img)  # for 3d bbox
            for obj in objects:
                if obj.type == 'DontCare': continue
                # if obj.type not in type2color.keys(): continue
                # c = type2color[obj.type]
                c = 0
                cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                              (int(obj.xmax), int(obj.ymax)), colors[c][::-1], 2)

                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, getattr(calib, present_sensor))

                # img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)
                def draw_rect(selected_corners, color):
                    prev = selected_corners[-1]
                    for corner in selected_corners:
                        cv2.line(img2,
                                 (int(prev[0]), int(prev[1])),
                                 (int(corner[0]), int(corner[1])),
                                 color, linewidth)
                        prev = corner

                corners_2d = box3d_pts_2d  # (8,2)
                # Draw the sides
                for i in range(4):
                    cv2.line(img2,
                             (int(corners_2d[i][0]), int(corners_2d[i][1])),
                             (int(corners_2d[i + 4][0]), int(corners_2d[i + 4][1])),
                             colors[c][::-1], linewidth)

                # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
                draw_rect(corners_2d[:4], colors[c][::-1])
                draw_rect(corners_2d[4:], colors[c][::-1])

                # Draw line indicating the front
                center_bottom_forward = np.mean(corners_2d[0:2], axis=0)
                center_bottom = np.mean(corners_2d[[0, 1, 2, 3]], axis=0)
                # center_bottom_forward = np.mean(corners_2d.T[2:4], axis=0)
                # center_bottom = np.mean(corners_2d.T[[2, 3, 7, 6]], axis=0)
                cv2.line(img2,
                         (int(center_bottom[0]), int(center_bottom[1])),
                         (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                         colors[c][::-1], linewidth)

            cv2.imwrite(os.path.join(save2ddir, str(data_idx).zfill(6) + '.jpg'), img1)
            cv2.imwrite(os.path.join(save3ddir, str(data_idx).zfill(6) + '.jpg'), img2)

def demo_object(data_idx=11,object_idx=0):
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    def draw_3d_object(pc, color=None):
        ''' Draw lidar points. simplest set up. '''
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
        if color is None: color = (pc[:, 2] - np.min(pc[:,2])) / (np.max(pc[: , 2])-np.min(pc[:, 2]))
        # draw points
        #nodes = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], colormap='gnuplot', scale_factor=0.04,
        #                  figure=fig)
        #nodes.mlab_source.dataset.point_data.scalars = color
        pts = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
        pts.add_attribute(color, 'colors')
        pts.data.point_data.set_active_scalars('colors')
        g = mlab.pipeline.glyph(pts)
        g.glyph.glyph.scale_factor = 0.05  # set scaling for all the points
        g.glyph.scale_mode = 'data_scaling_off'  # make all the points same size
        # draw origin
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        # draw axis
        axes = np.array([
            [2., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 2., 0.],
        ], dtype=np.float64)
        mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
        mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0,
                  figure=fig)
        return fig

    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI'))
    sensor = 'CAM_FRONT'
    objects = dataset.get_label_objects(sensor,data_idx)
    obj = objects[object_idx]
    obj.print_object()
    calib = dataset.get_calibration(data_idx)#utils.Calibration(calib_filename)
    box2d = obj.box2d
    xmin, ymin, xmax, ymax = box2d
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w, l = xmax - xmin, ymax - ymin
    # box3d
    x, y, z = obj.t
    # show 3d
    pc_velo = dataset.get_lidar(data_idx)[:, 0:3]
    pc_global = calib.project_lidar_to_global(pc_velo.T)
    pc_rect = calib.project_global_to_cam(pc_global,sensor).T
    pc_norm = pc_rect - obj.t
    keep = []
    for i in range(len(pc_norm)):
        if np.sum(pc_norm[i]**2) < 4:
            keep.append(i)
    pc_keep = pc_norm[keep,:]
    pc_keep[:,1] *= -1
    pc_keep = pc_keep[:,[0,2,1]]
    print("num_points:%d"%(pc_keep.shape[0]))
    fig = draw_3d_object(pc_keep)

    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[object_idx], getattr(calib,sensor))
    box3d_pts_3d -= obj.t
    box3d_pts_3d[:,1] *= -1
    box3d_pts_3d = box3d_pts_3d[:,[0,2,1]]
    draw_gt_boxes3d([box3d_pts_3d], fig=fig, draw_text=False)
    input()


def demo(data_idx=0,obj_idx=-1,show_lidar_with_box=True,show_image=True,
         show_bev=True,show_lidar_2d=True,show_lidar_in_box=True,show_project=True,show_frustum=True):
    sensor = 'CAM_FRONT'
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_gt_boxes3d
    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI'))

    # Load data from dataset
    objects = dataset.get_label_objects(sensor, data_idx)  # objects = [Object3d(line) for line in lines]
    for i,obj in enumerate(objects):
        print('obj %d'%(i))
        objects[i].print_object()

    calib = dataset.get_calibration(data_idx)  # utils.Calibration(calib_filename)
    box2d = objects[obj_idx].box2d
    xmin, ymin, xmax, ymax = box2d
    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    uvdepth = np.zeros((1, 3))
    uvdepth[0, 0:2] = box2d_center
    uvdepth[0, 2] = 20  # some random depth
    #box2d_center_rect = calib.project_image_to_rect(uvdepth)
    #frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
    #                                box2d_center_rect[0, 0])
    #print('frustum_angle:', frustum_angle)
    img = dataset.get_image(sensor, data_idx)  # (370, 1224, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    print(dataset.get_lidar(data_idx).shape)
    pc_velo = dataset.get_lidar(data_idx)[:, 0:3]  # (115384, 3)
    calib = dataset.get_calibration(data_idx)  # utils.Calibration(calib_filename)
    # 1.Draw lidar with boxes in LIDAR_TOP coord
    print(' -------- LiDAR points in LIDAR_TOP coordination --------')
    print('pc_velo.shape:',pc_velo.shape)
    print('pc_velo[:10,:]:',pc_velo[:10,:])
    ##view = np.eye(4)
    ##pc_velo[:, :3] = utils.view_points(pc_velo[:, :3].T, view, normalize=False).T
    ##pc_rect = calib.project_velo_to_rect(pc_velo)
    #fig = draw_lidar_simple(pc_velo)
    show_lidar_with_boxes(pc_velo, objects, calib, sensor, False, img_width, img_height)
    raw_input()

    if show_lidar_with_box:
        # 2.Draw frustum lidar with boxes in LIDAR_TOP coord
        print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
        #show_lidar_with_boxes(pc_velo, objects, calib)
        show_lidar_with_boxes(pc_velo.copy(), objects, calib, sensor, True, img_width, img_height)
        raw_input()

    if show_image:
        # 3.Draw 2d and 3d boxes on CAM_FRONT image
        print(' -------- 2D/3D bounding boxes in images --------')
        show_image_with_boxes(img, objects, calib, sensor)
        raw_input()

    if show_bev:
        print(' -------- render LiDAR points (and 3D boxes) in LIDAR_TOP coordinate --------')
        render_lidar_bev(pc_velo, objects, calib, sensor)
        raw_input()

    if show_lidar_2d:
        # Visualize LiDAR points on images
        print(' -------- LiDAR points projected to image plane --------')
        show_lidar_on_image(pc_velo, img.copy(), calib, sensor, img_width, img_height)#pc_velo:(n,3)
        raw_input()

    if show_lidar_in_box:
        # Show LiDAR points that are in the 3d box
        print(' -------- LiDAR points in a 3D bounding box --------')
        for obj_idx, obj in enumerate(objects):
            obj.print_object()
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[obj_idx], np.eye(4))
            box3d_pts_3d_global = calib.project_cam_to_global(box3d_pts_3d.T, sensor)  # (3,8)
            box3d_pts_3d_velo = calib.project_global_to_lidar(box3d_pts_3d_global)  # (3,8)
            box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo.T)
            print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))
            fig = mlab.figure(figure=None, bgcolor=(0,0,0),
                fgcolor=None, engine=None, size=(1000, 500))
            utils.draw_nusc_lidar(box3droi_pc_velo, fig=fig)
            draw_gt_boxes3d([box3d_pts_3d_velo.T], fig=fig)
            mlab.show(1)
            raw_input()


    if show_project:
        # UVDepth Image and its backprojection to point clouds
        print(' -------- LiDAR points in a frustum --------')

        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, sensor, 0, 0, img_width, img_height, True)
        imgfov_pts_2d = pts_2d[fov_inds,:]#(n, 3)
        imgfov_pc_global = calib.project_lidar_to_global(imgfov_pc_velo.T)
        imgfov_pc_cam = calib.project_global_to_cam(imgfov_pc_global, sensor)#(3,n)

        #cameraUVDepth = utils.view_points(imgfov_pc_cam[:3, :], getattr(calib,sensor), normalize=True)#(3,3067)
        #cameraUVDepth = cameraUVDepth#(3067, 3)
        #ipdb.set_trace()
        #cameraUVDepth = np.zeros_like(imgfov_pc_cam)
        #cameraUVDepth[:,0:2] = imgfov_pts_2d[:, 0:2]
        #cameraUVDepth[:,2] = imgfov_pc_cam[:,2]

        # miss intrisic
        # cameraUVDepth = imgfov_pc_cam
        # backprojected_pc_cam = cameraUVDepth

        #consider intrinsic
        print('imgfov_pc_cam.shape:',imgfov_pc_cam.shape)
        print('imgfov_pc_cam[:,0:5].T:\n',imgfov_pc_cam[:,0:5].T)
        cameraUVDepth = calib.project_cam_to_image(imgfov_pc_cam, sensor)#(3,n)
        cameraUVDepth[2,:] = imgfov_pc_cam[2,:]
        print('cameraUVDepth.shape:',cameraUVDepth.shape)
        print('cameraUVDepth[:,0:5].T:\n', cameraUVDepth[:, 0:5].T)
        backprojected_pc_cam = calib.project_image_to_cam(cameraUVDepth, sensor)#(3,n)
        print('backprojected_pc_cam.shape:', backprojected_pc_cam.shape)
        print('backprojected_pc_cam[:,0:5].T\n:', backprojected_pc_cam[:,0:5].T)
        print('error:')
        print(np.mean(backprojected_pc_cam - imgfov_pc_cam,axis=1))
        # Show that the points are exactly the same
        backprojected_pc_global = calib.project_cam_to_global(backprojected_pc_cam, sensor)#(3,n)
        backprojected_pc_velo = calib.project_global_to_lidar(backprojected_pc_global).T#(n,3)
        print('imgfov_pc_velo.shape:',imgfov_pc_velo.shape)
        print(imgfov_pc_velo[0:5,:])
        print('backprojected_pc_velo.shape:', backprojected_pc_velo.shape)
        print(backprojected_pc_velo[0:5,:])
        print('error:')
        print(np.mean(backprojected_pc_velo- imgfov_pc_velo,axis=0))
        fig = mlab.figure(figure=None, bgcolor=(0,0,0),
            fgcolor=None, engine=None, size=(1000, 500))
        utils.draw_nusc_lidar(backprojected_pc_velo, fig=fig)
        raw_input()

    if show_frustum:
        # Only display those points that fall into 2d box
        print(' -------- LiDAR points in a frustum from a 2D box --------')
        for obj_idx, obj in enumerate(objects):
            obj.print_object()
            xmin,ymin,xmax,ymax = \
                objects[obj_idx].xmin, obj.ymin, obj.xmax, obj.ymax
            boxfov_pc_velo = \
                get_lidar_in_image_fov(pc_velo, calib, sensor, xmin, ymin, xmax, ymax)
            print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

            fig = mlab.figure(figure=None, bgcolor=(0,0,0),
                fgcolor=None, engine=None, size=(1000, 500))
            utils.draw_nusc_lidar(boxfov_pc_velo, fig=fig)
            mlab.show(1)
            raw_input()

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])

def extract_frustum_data(idx_filename, split, sensor, output_filename, viz=False,
                         perturb_box2d=False, augmentX=1, type_whitelist=['Car'], with_image=False):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = []  # int number
    box2d_list = []  # [xmin,ymin,xmax,ymax]
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. Car
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis
    calib_list = [] # for fconvnet
    image_filename_list = [] # for fusion
    input_2d_list = []

    pos_cnt = 0.0
    all_cnt = 0.0
    time_get_fov = 0.0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)
        if with_image:
            image_filename = os.path.join(getattr(dataset,sensor+'_dir'),
                '%06d.png' % (data_idx))  # dataset.get_image(data_idx)#(370, 1224, 3),uint8
        objects = dataset.get_label_objects(sensor, data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_cam = np.zeros_like(pc_velo)
        pc_global = calib.project_lidar_to_global(pc_velo.T[0:3, :])
        pc_cam[:, 0:3] = calib.project_global_to_cam(pc_global, sensor).T
        pc_cam[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(sensor, data_idx)
        img_height, img_width, img_channel = img.shape
        time1 = time.perf_counter()
        _, pc_image_coord, img_fov_inds = \
            get_lidar_in_image_fov(pc_velo[:, 0:3],calib, sensor,
                                   0, 0, img_width, img_height, True)
        time_get_fov += (time.perf_counter() - time1)
        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist: continue

            # 2D BOX: Get pts rect backprojected
            box2d = objects[obj_idx].box2d
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                    #print(box2d)
                    #print(xmin, ymin, xmax, ymax)
                else:
                    xmin, ymin, xmax, ymax = box2d
                box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                               (pc_image_coord[:, 0] >= xmin) & \
                               (pc_image_coord[:, 1] < ymax) & \
                               (pc_image_coord[:, 1] >= ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_cam[box_fov_inds, :]  # (1607, 4)
                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_cam = calib.project_image_to_cam(uvdepth.T, sensor).T
                #box2d_center_rect = calib.project_image_to_rect(uvdepth.T).T
                frustum_angle = -1 * np.arctan2(box2d_center_cam[0, 2],
                                                box2d_center_cam[0, 0])
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, getattr(calib, sensor))  # (8, 2)(8, 3)
                _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)  # (375, 4)(1607,)
                label = np.zeros((pc_in_box_fov.shape[0]))  # (1607,)
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry  # 0.01
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])  # array([1.2 , 0.48, 1.89])

                # Reject too far away object or object without points
                if ymax - ymin < 25 or np.sum(label) == 0:
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)
                calib_list.append(getattr(calib,sensor))#for fconvnet, 3x3, not 3x4
                if with_image:
                    image_filename_list.append(image_filename)
                    input_2d_list.append(pc_image_coord[box_fov_inds,:])
                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / len(id_list)))
    print('Average time of get_lidar_in_image_fov: %.2fms'%(time_get_fov*1000/len(id_list)))

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(box3d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(calib_list, fp)
        if with_image:
            pickle.dump(image_filename_list, fp)
            pickle.dump(input_2d_list, fp)
    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            seg = label_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
                              fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:, 0], p1[:, 1], p1[:, 2], seg, mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
                              fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:, 2], -p1[:, 0], -p1[:, 1], seg, mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def print_npts_statistics(idx_filename,type_whitelist=['Car'],split='v1.0-mini'):
    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR,'dataset/nuScenes2KITTI'),split=split, sensor_list=['CAM_FRONT'])

    x = [0,50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000,10000]
    x = np.array(x).astype(np.int32)
    y = np.zeros(x.shape).astype(np.int32)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in tqdm(data_idx_list):
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        pc_velo = dataset.get_lidar(data_idx)
        pc_global = calib.project_lidar_to_global(pc_velo[:, 0:3].T)
        pc_rect = calib.project_global_to_cam(pc_global, sensor).T
        objects = dataset.get_label_objects(sensor, data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type not in type_whitelist:continue

            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[obj_idx], getattr(calib, sensor))
            pts_in_box3d, _ = extract_pc_in_box3d(pc_rect, box3d_pts_3d)
            npts = len(pts_in_box3d)
            for k in range(1, len(x)-1):
                if npts < x[k]:
                    y[k-1] += 1
                    break
    np.savetxt('kitti_npts_info', (x, y))

def print_npts_statistics_v2(idx_filename, type_whitelist=['Car'], split='train', sensor='CAM_FRONT'):
    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR, 'dataset/nuScenes2KITTI'), split=split,
                                    sensor_list=['CAM_FRONT'])
    N = 10000
    x = np.zeros(N)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in tqdm(data_idx_list):
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        pc_velo = dataset.get_lidar(data_idx)
        pc_global = calib.project_lidar_to_global(pc_velo[:, 0:3].T)
        pc_rect = calib.project_global_to_cam(pc_global, sensor).T
        objects = dataset.get_label_objects(sensor, data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type not in type_whitelist:continue

            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[obj_idx], getattr(calib, sensor))
            pts_in_box3d, _ = extract_pc_in_box3d(pc_rect, box3d_pts_3d)
            npts = len(pts_in_box3d)
            if npts >= 10000:continue
            x[npts] += 1
    tot = np.sum(x)
    prefix = np.zeros(N)
    prefix[0] = x[0]
    now = 0.1
    step = 0.1
    y = []
    for i in range(1,N):
        prefix[i] = prefix[i-1] + x[i]
        if prefix[i] > now*tot:
            y.append(i)
            now += step
    print(y)
    np.savetxt('kitti_npts_info_v2', (y), fmt='%d')
def get_box3d_dim_statistics(idx_filename):
    ''' Collect and dump 3D bounding box statistics '''
    pass

def print_box3d_statistics(idx_filename,type_whitelist=['Car','Pedestrian','Cyclist'],split='v1.0-mini', sensor='CAM_FRONT'):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR,'dataset/nuScenes2KITTI'),split, sensor_list=['CAM_FRONT'])

    dimension_list = []
    type_list = []
    ry_list = []
    mean_t_list = []
    mean_t_by_center_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in tqdm(data_idx_list):
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        pc_velo = dataset.get_lidar(data_idx)
        pc_global = calib.project_lidar_to_global(pc_velo[:, 0:3].T)
        pc_rect = calib.project_global_to_cam(pc_global, sensor).T
        objects = dataset.get_label_objects(sensor, data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type not in type_whitelist:continue
            dimension_list.append(np.array([obj.l,obj.w,obj.h]))
            type_list.append(obj.type)
            ry_list.append(obj.ry)

            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[obj_idx], getattr(calib, sensor))
            pts_in_box3d, _ = extract_pc_in_box3d(pc_rect, box3d_pts_3d)
            if len(pts_in_box3d) == 0: continue
            mean_t_list.append(pts_in_box3d.mean(0))
            pts_in_box3d -= obj.t
            mean_t_by_center_list.append(pts_in_box3d.mean(0))

    dimensions = np.array(dimension_list)
    mts = np.array(mean_t_list)
    rys = np.array(ry_list)
    mtbcs = np.array(mean_t_by_center_list)
    md = dimensions.mean(0)
    mmt = mts.mean(0)
    mry = rys.mean()
    mmtbcs = mtbcs.mean(0)

    print('mean points in 3d box: (%.1f,%.1f,%.1f)' % (mmt[0],mmt[1],mmt[2]))
    print('mean points related to box center: (%.1f,%.1f,%.1f)' % (mmtbcs[0], mmtbcs[1], mmtbcs[2]))
    print('mean size: (%.1f,%.1f,%.1f)' % (md[0],md[1],md[2]))
    print('mean ry: (%.2f)' % (mry))
    """
    v1.0-mini-caronly-CAM_FRONT
    mean points in 3d box: (2.2,0.8,27.8)
    mean points related to box center: (-0.1,-0.6,-1.0)
    mean size: (4.6,1.9,1.7)
    mean ry: (0.04)

    training
    
    v1.0-trainval
    mean points in 3d box: (-0.9,1.0,29.8)
    mean points related to box center: (0.0,-0.6,-1.2)
    mean size: (4.7,2.0,1.7)
    mean ry: (-0.12)
    """
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true',
                        help='Run demo.')
    parser.add_argument('--demo_object', action='store_true',
                        help='Run demo_object.')
    parser.add_argument('--show_stats', action='store_true',
                        help='Run print_box3d_statistics.')
    parser.add_argument('--npts_stats', action='store_true', help='npts_stats.')
    parser.add_argument('--data_idx', type=int, default=0,
                        help='data_idx for demo.')
    parser.add_argument('--obj_idx', type=int, default=-1,
                        help='obj_idx for demo.')
    parser.add_argument('--vis_label', action='store_true',
                        help='Run vis_label.')
    parser.add_argument('--vis_pred', action='store_true',
                        help='Run vis_pred.')
    parser.add_argument('--vis_pred_path', type=str, default='train/transfer_kitti2nuscenes_caronly_v1_fromgt/data',
                        help='vis predicted label path')
    parser.add_argument('--gen_mini', action='store_true',
                        help='Generate v1.0-mini split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_train', action='store_true',
                        help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--train_sets', type=str, default='train',
                        help='*.txt in nuscenes2kitti/image_sets')
    parser.add_argument('--val_sets', type=str, default='val',
                        help='*.txt in nuscenes2kitti/image_sets')
    parser.add_argument('--gen_val', action='store_true',
                        help='Generate val split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_split', type=str, default='training',
                        help='training,testing,v1.0-mini,v1.0-trainval')
    #parser.add_argument('--gen_trainval', action='store_true',
    #                    help='Generate trainval split frustum data with GT 2D boxes')
    #parser.add_argument('--gen_test', action='store_true',
    #                    help='Generate test split frustum data with GT 2D boxes')

    #parser.add_argument('--gen_val_rgb_detection', action='store_true',
    #                    help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true',
                        help='Only generate cars; otherwise cars, peds and cycs')
    parser.add_argument('--CAM_FRONT_only', action='store_true',
                        help='Only generate CAM_FRONT; otherwise six cameras')
    parser.add_argument('--with_image', action='store_true')
    args = parser.parse_args()



    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.CAM_FRONT_only:
        sensor_list = ['CAM_FRONT']
    else:
        sensor_list = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']

    if args.demo_object:
        demo_object(data_idx=args.data_idx, object_idx=args.obj_idx)

    if args.npts_stats:
        imagesets_file = os.path.join(BASE_DIR, 'image_sets/v1.0-mini.txt')
        print_npts_statistics_v2(imagesets_file, type_whitelist, 'v1.0-mini')
        imagesets_file = os.path.join(BASE_DIR, 'image_sets/v1.0-trainval.txt')
        print_npts_statistics_v2(imagesets_file, type_whitelist, 'v1.0-trainval')

    if args.show_stats:
        print_box3d_statistics('nuscenes2kitti/image_sets/v1.0-mini.txt',
                               type_whitelist=type_whitelist, split='v1.0-mini', sensor='CAM_FRONT')
        print_box3d_statistics('nuscenes2kitti/image_sets/v1.0-trainval.txt',
                               type_whitelist=type_whitelist, split='v1.0-trainval', sensor='CAM_FRONT')

    if args.demo:
        demo(args.data_idx,args.obj_idx,show_lidar_with_box=False,show_image=True,
         show_bev=True,show_lidar_2d=False,show_lidar_in_box=True,show_project=False,show_frustum=True)
        print('demo is done. Exit now.')
        exit()

    if args.vis_label:
        vis_label(split='training',sensor_list=sensor_list,type_whitelist=type_whitelist)
        print('vis_label is done. Exit now.')
        exit()

    if args.vis_pred:
        vis_pred(split='training',sensor_list=sensor_list,type_whitelist=type_whitelist,vis_pred_path=args.vis_pred_path)
        print('vis_pred is done. Exit now.')
        exit()

    if args.gen_mini:
        for sensor in sensor_list:
            sensor_prefix = sensor + '_'
            extract_frustum_data(\
                os.path.join(BASE_DIR, 'image_sets/v1.0-mini.txt'),
                'v1.0-mini',
                sensor,
                os.path.join(BASE_DIR, output_prefix + sensor_prefix + 'v1.0-mini.pickle'),
                viz=False, perturb_box2d=True, augmentX=5,
                type_whitelist=type_whitelist,
                with_image=args.with_image)
        print('gen_mini is done.')

    if args.gen_train:
        for sensor in sensor_list:
            sensor_prefix = sensor + '_'
            extract_frustum_data(\
                os.path.join(BASE_DIR, 'image_sets', args.train_sets+'.txt'),
                args.gen_split,
                sensor,
                os.path.join(BASE_DIR, output_prefix + sensor_prefix + args.train_sets+'.pickle'),
                viz=False, perturb_box2d=True, augmentX=5,
                type_whitelist=type_whitelist,
                with_image=args.with_image)
        print('gen_train is done.')

    if args.gen_val:
        for sensor in sensor_list:
            sensor_prefix = sensor + '_'
            extract_frustum_data(\
                os.path.join(BASE_DIR, 'image_sets', args.val_sets+'.txt'),
                args.gen_split,
                sensor,
                os.path.join(BASE_DIR, output_prefix + sensor_prefix + args.val_sets+'.pickle'),
                viz=False, perturb_box2d=False, augmentX=1,
                type_whitelist=type_whitelist,
                with_image=args.with_image)
        print('gen_val is done.')
