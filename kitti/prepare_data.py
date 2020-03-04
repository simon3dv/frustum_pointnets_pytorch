''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import kitti_util as utils
import _pickle as pickle
from kitti_object import *
import argparse
import ipdb
from tqdm import tqdm

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]] 
    box2d_corners[1,:] = [box2d[2],box2d[1]] 
    box2d_corners[2,:] = [box2d[2],box2d[3]] 
    box2d_corners[3,:] = [box2d[0],box2d[3]] 
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds

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

    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))
    objects = dataset.get_label_objects(data_idx)
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
    pc_rect = calib.project_velo_to_rect(pc_velo)
    pc_norm = pc_rect - obj.t
    keep = []
    for i in range(len(pc_norm)):
        if np.sum(pc_norm[i]**2) < 4:
            keep.append(i)
    pc_keep = pc_norm[keep,:]
    pc_keep[:,1] *= -1
    pc_keep = pc_keep[:,[0,2,1]]
    fig = draw_3d_object(pc_keep)

    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[object_idx], calib.P)
    box3d_pts_3d -= obj.t
    box3d_pts_3d[:,1] *= -1
    box3d_pts_3d = box3d_pts_3d[:,[0,2,1]]
    draw_gt_boxes3d([box3d_pts_3d], fig=fig, draw_text=False)
    input()

def demo(data_idx=11,object_idx=0,show_images=True,show_lidar=True,show_lidar_2d=True,show_lidar_box=True,show_project=True,show_lidar_frustum=True):
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)#objects = [Object3d(line) for line in lines]
    objects[object_idx].print_object()

    calib = dataset.get_calibration(data_idx)#utils.Calibration(calib_filename)
    box2d = objects[object_idx].box2d
    xmin, ymin, xmax, ymax = box2d
    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    uvdepth = np.zeros((1, 3))
    uvdepth[0, 0:2] = box2d_center
    uvdepth[0, 2] = 20  # some random depth
    box2d_center_rect = calib.project_image_to_rect(uvdepth)
    frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                    box2d_center_rect[0, 0])
    print('frustum_angle:',frustum_angle)
    '''
    Type, truncation, occlusion, alpha: Pedestrian, 0, 0, -0.200000
    2d bbox (x0,y0,x1,y1): 712.400000, 143.000000, 810.730000, 307.920000
    3d bbox h,w,l: 1.890000, 0.480000, 1.200000
    3d bbox location, ry: (1.840000, 1.470000, 8.410000), 0.010000
    '''
    img = dataset.get_image(data_idx)#(370, 1224, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    pc_velo = dataset.get_lidar(data_idx)[:,0:3]#(115384, 3)
    calib = dataset.get_calibration(data_idx)#utils.Calibration(calib_filename)

    ## Draw lidar in rect camera coord
    #print(' -------- LiDAR points in rect camera coordination --------')
    #pc_rect = calib.project_velo_to_rect(pc_velo)
    #fig = draw_lidar_simple(pc_rect)
    #raw_input()
    # Draw 2d and 3d boxes on image
    if show_images:
        print(' -------- 2D/3D bounding boxes in images --------')
        show_image_with_boxes(img, objects, calib)
        raw_input()

    if show_lidar:
        # Show all LiDAR points. Draw 3d box in LiDAR point cloud
        print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
        #show_lidar_with_boxes(pc_velo, objects, calib)
        #raw_input()
        show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
        raw_input()

    if show_lidar_2d:
        # Visualize LiDAR points on images
        print(' -------- LiDAR points projected to image plane --------')
        show_lidar_on_image(pc_velo, img, calib, img_width, img_height, showtime=True)
        raw_input()

    if show_lidar_box:
        # Show LiDAR points that are in the 3d box
        print(' -------- LiDAR points in a 3D bounding box --------')
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[object_idx], calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
        print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

        fig = mlab.figure(figure=None, bgcolor=(0,0,0),
            fgcolor=None, engine=None, size=(1000, 500))
        draw_lidar(box3droi_pc_velo, fig=fig)
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
        mlab.show(1)
        raw_input()

    if show_project:
        # UVDepth Image and its backprojection to point clouds
        print(' -------- LiDAR points in a frustum from a 2D box --------')
        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
            calib, 0, 0, img_width, img_height, True)
        imgfov_pts_2d = pts_2d[fov_inds,:]
        imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

        cameraUVDepth = np.zeros_like(imgfov_pc_rect)
        cameraUVDepth[:,0:2] = imgfov_pts_2d
        cameraUVDepth[:,2] = imgfov_pc_rect[:,2]

        # Show that the points are exactly the same
        backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
        print(imgfov_pc_velo[0:20])
        print(backprojected_pc_velo[0:20])

        fig = mlab.figure(figure=None, bgcolor=(0,0,0),
            fgcolor=None, engine=None, size=(1000, 500))
        draw_lidar(backprojected_pc_velo, fig=fig)
        raw_input()

    if show_lidar_frustum:
        # Only display those points that fall into 2d box
        print(' -------- LiDAR points in a frustum from a 2D box --------')
        xmin,ymin,xmax,ymax = \
            objects[object_idx].xmin, objects[object_idx].ymin, objects[object_idx].xmax, objects[object_idx].ymax
        boxfov_pc_velo = \
            get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
        print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

        fig = mlab.figure(figure=None, bgcolor=(0,0,0),
            fgcolor=None, engine=None, size=(1000, 500))
        draw_lidar(boxfov_pc_velo, fig=fig)
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
 
def extract_frustum_data(idx_filename, split, output_filename, viz=False,
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
    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    box3d_list = [] # (8,3) array in rect camera coord
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    label_list = [] # 1 for roi object, 0 for clutter
    type_list = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis
    calib_list = [] # calibration matrix 3x4 for fconvnet
    image_filename_list = [] # for fusion
    input_2d_list = []

    pos_cnt = 0
    all_cnt = 0
    for data_idx in tqdm(data_idx_list):
        #print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        if with_image:
            image_filename = os.path.join(dataset.image_dir, '%06d.png'%(data_idx))#dataset.get_image(data_idx)#(370, 1224, 3),uint8
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)
        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist :continue

            # 2D BOX: Get pts rect backprojected 
            box2d = objects[obj_idx].box2d
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin,ymin,xmax,ymax = random_shift_box2d(box2d)
                    #print(box2d)
                    #print(xmin,ymin,xmax,ymax)
                else:
                    xmin,ymin,xmax,ymax = box2d
                box_fov_inds = (pc_image_coord[:,0]<xmax) & \
                    (pc_image_coord[:,0]>=xmin) & \
                    (pc_image_coord[:,1]<ymax) & \
                    (pc_image_coord[:,1]>=ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds,:]#(1607, 4)
                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                uvdepth = np.zeros((1,3))
                uvdepth[0,0:2] = box2d_center
                uvdepth[0,2] = 20 # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
                    box2d_center_rect[0,0])
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) #(8, 2)(8, 3)
                _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)#(375, 4)(1607,)
                label = np.zeros((pc_in_box_fov.shape[0]))#(1607,)
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry#0.01
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])#array([1.2 , 0.48, 1.89])

                # Reject too far away object or object without points
                if ymax-ymin<25 or np.sum(label)==0:
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)
                calib_list.append(calib.P)
                if with_image:
                    image_filename_list.append(image_filename)
                    input_2d_list.append(pc_image_coord[box_fov_inds,:])
                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]
        
    print('Average pos ratio: %f' % (pos_cnt/float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt)/len(id_list)))
    
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(box3d_list,fp)
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
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def get_box3d_dim_statistics(idx_filename, type_whitelist=['Car','Pedestrian','Cyclist'],split='train'):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in tqdm(data_idx_list):
        #print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type not in type_whitelist:continue
            dimension_list.append(np.array([obj.l,obj.w,obj.h])) 
            type_list.append(obj.type) 
            ry_list.append(obj.ry)

    dimensions = np.array(dimension_list)
    if len(type_whitelist) == 1:
        dimensions_mean = dimensions.mean(0)
        print(dimensions_mean)
        type = type_whitelist[0]
        with open(os.path.join(BASE_DIR, split + '_' + type + '_' + 'box3d_mean_dimensions.pickle'),'wb') as fp:
            pickle.dump(dimensions_mean, fp)
    else:
        import pandas as pd
        df = pd.DataFrame(dimensions)
        grouped = df.groupby(type_list)
        for type in type_whitelist:
            dimensions = grouped.get_group(type)
            dimensions_mean = np.array(dimensions.mean(0))
            print(type,':',dimensions_mean)
            with open(os.path.join(BASE_DIR, split + '_' + type + '_' + 'box3d_mean_dimensions.pickle'), 'wb') as fp:
                pickle.dump(dimensions_mean, fp)

def print_box3d_statistics(idx_filename,type_whitelist=['Car','Pedestrian','Cyclist'],split='train'):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'))

    dimension_list = []
    type_list = []
    ry_list = []
    mean_t_list = []
    mean_t_by_center_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in tqdm(data_idx_list):
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = calib.project_velo_to_rect(pc_velo[:, 0:3])
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type not in type_whitelist:continue
            dimension_list.append(np.array([obj.l,obj.w,obj.h]))
            type_list.append(obj.type)
            ry_list.append(obj.ry)

            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[obj_idx], calib.P)
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
    train-carpedcyc
    mean points in 3d box: (-1.8,1.0,26.5)
    mean points related to box center: (0.0,-0.7,-0.8)
    mean size: (3.4,1.4,1.6)
    mean ry: (0.03)


    train-car
    mean points in 3d box: (-2.3,1.0,28.0)
    mean points related to box center: (0.0,-0.7,-1.0)
    mean size: (3.9,1.6,1.5)
    mean ry: (0.02)

    """

def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3,7)]))
    return id_list, type_list, box2d_list, prob_list

 
def extract_frustum_data_rgb_detection(det_filename, split, output_filename,
                                       viz=False,
                                       type_whitelist=['Car'],
                                       img_height_threshold=25,
                                       lidar_point_threshold=5):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)
        
    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    cache_id = -1
    cache = None
    
    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = [] # angle of 2d box center from pos x-axis
    calib_list = []

    for det_idx in tqdm(range(len(det_id_list))):
        data_idx = det_id_list[det_idx]
        #print('det idx: %d/%d, data idx: %d' % \
        #    (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
            pc_velo = dataset.get_lidar(data_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
            pc_rect[:,3] = pc_velo[:,3]
            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(\
                pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)
            cache = [calib,pc_rect,pc_image_coord,img_fov_inds]
            cache_id = data_idx
        else:
            calib,pc_rect,pc_image_coord,img_fov_inds = cache

        if det_type_list[det_idx] not in type_whitelist: continue

        # 2D BOX: Get pts rect backprojected 
        xmin,ymin,xmax,ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:,0]<xmax) & \
            (pc_image_coord[:,0]>=xmin) & \
            (pc_image_coord[:,1]<ymax) & \
            (pc_image_coord[:,1]>=ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds,:]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
        uvdepth = np.zeros((1,3))
        uvdepth[0,0:2] = box2d_center
        uvdepth[0,2] = 20 # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
            box2d_center_rect[0,0])
        
        # Pass objects that are too small
        if ymax-ymin<img_height_threshold or \
            len(pc_in_box_fov)<lidar_point_threshold:
            continue
       
        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)
        calib_list.append(calib.P)
    
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)
        pickle.dump(calib_list, fp)
    
    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], p1[:,1], mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def write_2d_rgb_detection(det_filename, split, result_dir):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format. 
        
    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        result_dir: string, folder path for results dumping
    Output:
        None (will write <xxx>.txt files to disk)

    Usage:
        write_2d_rgb_detection("val_det.txt", "training", "results")
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    # map from idx to list of strings, each string is a line without \n
    results = {} 
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close() 

def cluster():
    imagesets_file = os.path.join(BASE_DIR, 'image_sets/train.txt')
    all_type = ['Car','Pedestrian','Cyclist']
    get_box3d_dim_statistics(imagesets_file,all_type,split='train')
    '''
    train.txt
    Car : [3.89206519 1.61876715 1.52986348]
    Pedestrian : [0.81796103 0.62783416 1.76773901]
    Cyclist : [1.77081744 0.56961853 1.72332425]
    trainval.txt
    Car : [3.88395449 1.62858987 1.52608343]
    Pedestrian : [0.84228438 0.66018944 1.76070649]
    Cyclist : [1.7635464  0.5967732  1.73720344]
    val.txt
    Car : [3.87585958 1.63839347 1.52231074]
    Pedestrian : [0.86582895 0.69150877 1.75389912]
    Cyclist : [1.75756999 0.61909295 1.74861142]

    '''

if __name__=='__main__':
    #python kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--demo_object', action='store_true', help='Run demo_object.')
    parser.add_argument('--show_stats', action='store_true', help='show_stats.')
    parser.add_argument('--data_idx', type=int, default=0,
                        help='data_idx for demo.')
    parser.add_argument('--obj_idx', type=int, default=0,
                        help='obj_idx for demo.')
    parser.add_argument('--cluster', action='store_true', help='Run cluster.')
    parser.add_argument('--gen_train', action='store_true', help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection', action='store_true',
                        help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--gen_mini', action='store_true')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars; otherwise cars, peds and cycs')
    parser.add_argument('--with_image', action='store_true')
    args = parser.parse_args()

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.with_image:
        output_prefix += 'wimage_'


    if args.cluster:
        cluster()
        exit()
    if args.show_stats:
        imagesets_file = os.path.join(BASE_DIR, 'image_sets/train.txt')
        print_box3d_statistics(imagesets_file, type_whitelist, 'train')
    if args.demo_object:
        demo_object(data_idx=args.data_idx, object_idx=args.obj_idx)
    if args.demo:
        demo(data_idx=args.data_idx, object_idx=args.obj_idx, show_images=True, show_lidar=False,
             show_lidar_2d=False, show_lidar_box=True,
             show_project=False, show_lidar_frustum=True)  # draw 2d box and 3d box
        exit()

    if args.gen_mini:
        print('Start gen_train...')
        imagesets_file = os.path.join(BASE_DIR, 'image_sets/mini.txt')
        extract_frustum_data( \
            imagesets_file,
            'training',
            os.path.join(BASE_DIR, output_prefix + 'mini.pickle'),
            viz=False, perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist,
            with_image=args.with_image)
        get_box3d_dim_statistics(imagesets_file, type_whitelist, 'train')

    if args.gen_train:
        print('Start gen_train...')
        imagesets_file = os.path.join(BASE_DIR, 'image_sets/train.txt')
        extract_frustum_data(\
            imagesets_file,
            'training',
            os.path.join(BASE_DIR, output_prefix+'train.pickle'), 
            viz=False, perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist,
            with_image=args.with_image)
        get_box3d_dim_statistics(imagesets_file, type_whitelist,'train')

    if args.gen_val:
        print('Start gen_val...')
        imagesets_file = os.path.join(BASE_DIR, 'image_sets/val.txt')
        extract_frustum_data(\
            imagesets_file,
            'training',
            os.path.join(BASE_DIR, output_prefix+'val.pickle'),
            viz=False, perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist,
            with_image=args.with_image)
        get_box3d_dim_statistics(imagesets_file, type_whitelist,'val')

    if args.gen_val_rgb_detection:
        print('Start gen_val_rgb_detection...')
        extract_frustum_data_rgb_detection(\
            os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'val_rgb_detection.pickle'),
            viz=False,
            type_whitelist=type_whitelist)
