''' Helper class and functions for loading nuscenes2kitti objects

Authod: Siming Fan
Acknowledge: Charles R. Qi
Date: Jan 2020
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import nuscenes2kitti_util as utils
import ipdb
from pyquaternion import Quaternion
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


class nuscenes2kitti_object(object):
    '''Load and parse object data into a usable format.'''
    
    def __init__(self, root_dir, split='v1.0-mini',sensor_list=['CAM_FRONT']):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'v1.0-mini':
            self.num_samples = 404
        elif split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        elif split == 'v1.0-trainval':
            self.num_samples = 34149
        #elif split == 'v1.0-val':
        #    self.num_samples = 34149#17074
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.sensor_list = sensor_list
        self.CAM_FRONT_dir = os.path.join(self.split_dir, 'image_CAM_FRONT')
        self.CAM_FRONT_RIGHT_dir = os.path.join(self.split_dir, 'image_CAM_FRONT_RIGHT')
        self.CAM_BACK_RIGHT_dir = os.path.join(self.split_dir, 'image_CAM_BACK_RIGHT')
        self.CAM_BACK_dir = os.path.join(self.split_dir, 'image_CAM_BACK')
        self.CAM_BACK_LEFT_dir = os.path.join(self.split_dir, 'image_CAM_BACK_LEFT')
        self.CAM_FRONT_LEFT_dir = os.path.join(self.split_dir, 'image_CAM_FRONT_LEFT')

        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'LIDAR_TOP')

        self.label_CAM_FRONT_dir = os.path.join(self.split_dir, 'label_CAM_FRONT')
        self.label_CAM_FRONT_RIGHT_dir = os.path.join(self.split_dir, 'label_CAM_FRONT_RIGHT')
        self.label_CAM_BACK_RIGHT_dir = os.path.join(self.split_dir, 'label_CAM_BACK_RIGHT')
        self.label_CAM_BACK_dir = os.path.join(self.split_dir, 'label_CAM_BACK')
        self.label_CAM_BACK_LEFT_dir = os.path.join(self.split_dir, 'label_CAM_BACK_LEFT')
        self.label_CAM_FRONT_LEFT_dir = os.path.join(self.split_dir, 'label_CAM_FRONT_LEFT')

    def __len__(self):
        return self.num_samples

    def get_image(self, sensor, idx):
        assert(idx<self.num_samples)
        assert (sensor in self.sensor_list)
        img_filename = os.path.join(getattr(self,sensor+'_dir'), '%06d.jpg'%(idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx): 
        assert(idx<self.num_samples) 
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        assert(idx<self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.Calibration(calib_filename,sensor_list=self.sensor_list)

    def get_label_objects(self, sensor, idx):
        assert(idx<self.num_samples and self.split!='v1.0-test')
        assert (sensor in self.sensor_list)
        label_filename = os.path.join(getattr(self,'label_'+sensor+'_dir'), '%06d.txt'%(idx))
        return utils.read_label(label_filename)
        
    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass

def render_objects(img, objects, view=np.eye(4), colors = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),linewidth=2):
    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(img,
                     (int(prev[0]), int(prev[1])),
                     (int(corner[0]), int(corner[1])),
                     color, linewidth=linewidth)
            prev = corner

    for obj in objects:
        if obj.type=='DontCare':continue
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, view)

        # Draw the sides
        for i in range(4):
            cv2.line(img,
                     (int(corners_2d.T[i][0]), int(corners_2d.T[i][1])),
                     (int(corners_2d.T[i + 4][0]), int(corners_2d.T[i + 4][1])),
                     colors[c][::-1], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners_2d.T[:4], colors[c][::-1])
        draw_rect(corners_2d.T[4:], colors[c][::-1])

        corners_2d = box3d_pts_2d

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners_2d.T[0:2], axis=0)
        center_bottom = np.mean(corners_2d.T[[0, 1, 2, 3]], axis=0)
        # center_bottom_forward = np.mean(corners_2d.T[2:4], axis=0)
        # center_bottom = np.mean(corners_2d.T[[2, 3, 7, 6]], axis=0)
        cv2.line(img,
                 (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                 colors[0][::-1], linewidth)

def show_image_with_boxes(img, objects, calib, sensor, show3d=True,linewidth=2,colors = ((0, 0, 255), (255, 0, 0), (155, 155, 155))):
    ''' Show image with 2D bounding boxes '''
    view = getattr(calib, sensor)
    img1 = np.copy(img) # for 2d bbox
    img2 = np.copy(img) # for 3d bbox
    type2color = {'Pedestrian':0,
                  'Car':1,
                  'Cyclist':2}
    for obj in objects:
        if obj.type=='DontCare':continue
        if obj.type not in type2color.keys():continue
        c = type2color[obj.type]
        cv2.rectangle(img1, (int(obj.xmin),int(obj.ymin)),
            (int(obj.xmax),int(obj.ymax)), colors[c][::-1], 2)

        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, view)

        #img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)
        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(img2,
                         (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])),
                         color, linewidth)
                prev = corner

        corners_2d = box3d_pts_2d.T#(2,8)
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

    Image.fromarray(img1).show()
    if show3d:
        Image.fromarray(img2).show()

def project_velo_to_image(pc_velo, calib, sensor, return_time=False):
    ''' Input: nx3 points in velodyne coord.
        Output: nx3 points in image2 coord.
    '''
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    if return_time:
        import time
        start = time.time()
    pts_3d_velo = pc_velo.T
    pts_3d_ego = rotate(pts_3d_velo, getattr(calib,'lidar2ego_rotation'))
    pts_3d_ego = translate(pts_3d_ego, getattr(calib,'lidar2ego_translation'))

    # Second step: transform to the global frame.
    pts_3d_global=rotate(pts_3d_ego,getattr(calib,'ego2global_rotation'))
    pts_3d_global=translate(pts_3d_global,getattr(calib,'ego2global_translation'))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    pts_3d_ego_cam = translate(pts_3d_global, -getattr(calib,sensor+'_'+'ego2global_translation'))
    pts_3d_ego_cam = rotate(pts_3d_ego_cam, getattr(calib,sensor+'_'+'ego2global_rotation').T)

    # Fourth step: transform into the camera.
    pts_3d_cam = translate(pts_3d_ego_cam, -getattr(calib,sensor+'_'+'cam2ego_translation'))
    pts_3d_cam = rotate(pts_3d_cam, getattr(calib,sensor+'_'+'cam2ego_rotation').T)

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    depths = pts_3d_cam[2, :]
    pts_2d_cam = utils.view_points(pts_3d_cam[:3, :], getattr(calib,sensor), normalize=True)#(3,n)
    pts_2d_cam = pts_2d_cam.T
    pts_2d_cam[:,2] = depths.T
    if return_time:
        end = time.time()
        print('Time(project_velo_to_image):',end-start)
        return pts_2d_cam,end-start
    else:
        return pts_2d_cam

def get_lidar_in_image_fov(pc_velo, calib, sensor, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    '''    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        view, 0, 0, img_width, img_height, True)
    '''

    pts_global = calib.project_lidar_to_global(pc_velo.T)
    pts_3d_cam = calib.project_global_to_cam(pts_global, sensor)
    depths = pts_3d_cam[2, :]
    pts_2d = utils.view_points(pts_3d_cam[:3, :], getattr(calib,sensor), normalize=True)#(3,n)
    pts_2d = pts_2d.T
    pts_2d[:,2] = depths.T


    #pts_2d = project_velo_to_image(pc_velo.copy(), calib, sensor)

    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)#26414->7149
    # Use depth before projection
    fov_inds = fov_inds & (pts_2d[:,2]>clip_distance)#7149->3067
    imgfov_pc_velo = pc_velo[fov_inds,:]#(3067, 3),mean:array([-1.1616094e-02,  1.8041308e-01,  1.5962053e+01], dtype=float32)
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo

def render_lidar_bev(pc_velo, objects, calib, sensor, saveto=None):
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    print('pc_velo.shape:',pc_velo.shape)
    ax.scatter(pc_velo[:, 0], pc_velo[:, 1], c=pc_velo[:, 2], s=0.2)
    ax.set_xlim(-40,40)
    ax.set_ylim(-20,70)

    linewidth = 2
    colors=('b', 'r', 'k')
    for obj in objects:
        if obj.type=='DontCare':continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, np.eye(4))#(8,2),(8,3)
        box3d_pts_3d_global = calib.project_cam_to_global(box3d_pts_3d.T, sensor)  # (3,8)
        box3d_pts_3d_velo = calib.project_global_to_lidar(box3d_pts_3d_global)#(3,8)
        corners = box3d_pts_3d_velo#(3,8)
        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            ax.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        ax.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)
    if saveto:
        plt.savefig(saveto)
    else:
        plt.show()

def show_lidar_with_boxes(pc_velo, objects, calib, sensor,
                          img_fov=False, img_width=None, img_height=None): 
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple
    #from viz_util import draw_lidar
    from viz_util import draw_gt_boxes3d

    view = getattr(calib,sensor)
    print(('All point num: ', pc_velo.shape[0]))
    #fig = mlab.figure(figure=None, bgcolor=(0,0,0),
    #    fgcolor=None, engine=None, size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, sensor, 0, 0,
            img_width, img_height)
        print(('FOV point num: ', pc_velo.shape[0]))
    #draw_lidar(pc_velo, fig=fig)
    #fig = draw_nusc_lidar(pc_velo,pts_scale=0.1,pts_mode='sphere')
    fig = utils.draw_nusc_lidar(pc_velo)
    obj_mean = np.array([0.0,0.0,0.0])
    obj_count = 0
    for obj in objects:
        if obj.type=='DontCare':continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, np.eye(4))#(8,2),(8,3)
        box3d_pts_3d_global = calib.project_cam_to_global(box3d_pts_3d.T, sensor)  # (3,8)
        box3d_pts_3d_velo = calib.project_global_to_lidar(box3d_pts_3d_global).T#(8,3)
        # box3d_pts_3d_velo = box3d_pts_3d
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, np.eye(4))#(2,2),(2,3)
        ori3d_pts_3d_global = calib.project_cam_to_global(ori3d_pts_3d.T, sensor)#(3,2)
        ori3d_pts_3d_velo = calib.project_global_to_lidar(ori3d_pts_3d_global).T#(2,3)
        ori3d_pts_3d_velo = ori3d_pts_3d
        x1,y1,z1 = ori3d_pts_3d_velo[0,:]
        x2,y2,z2 = ori3d_pts_3d_velo[1,:]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)

        mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5),
            tube_radius=None, line_width=1, figure=fig)

        obj_mean += np.sum(box3d_pts_3d_velo,axis=0)
        obj_count += 1
    obj_mean = obj_mean / obj_count
    print("mean:",obj_mean)
    mlab.show(1)


def translate(points, x: np.ndarray) -> None:
    """
    Applies a translation to the point cloud.
    :param x: <np.float: 3, 1>. Translation in x, y, z.
    """
    for i in range(3):
        points[i, :] = points[i, :] + x[i]
    return points

def rotate(points, rot_matrix: np.ndarray) -> None:
    """
    Applies a rotation.
    :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
    """
    points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points

def show_lidar_on_image(pc_velo, img, calib, sensor, img_width, img_height):
    ''' Project LiDAR points to image '''
    view = getattr(calib,sensor)
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, sensor, 0, 0, img_width, img_height, True)

    imgfov_pts_2d = pts_2d[fov_inds,:]
    #imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255

    for i in range(imgfov_pts_2d.shape[0]):
        #depth = imgfov_pc_velo[i,2]
        depth = imgfov_pts_2d[i, 2]
        #color = cmap[int(640.0/depth),:]
        color = cmap[int(320.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i,0])),
            int(np.round(imgfov_pts_2d[i,1]))),
            2, color=tuple(color), thickness=-1)
    Image.fromarray(img).show() 
    return img

def dataset_viz():
    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))

    for data_idx in range(len(dataset)):
        # Load data from dataset
        objects = dataset.get_label_objects(data_idx)
        objects[0].print_object()
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_height, img_width, img_channel = img.shape
        print(('Image shape: ', img.shape))
        pc_velo = dataset.get_lidar(data_idx)[:,0:3]
        calib = dataset.get_calibration(data_idx)

        sensor = 'CAM_FRONT'
        # Draw 2d and 3d boxes on image
        show_image_with_boxes(img, objects, calib, sensor, False)
        raw_input()
        # Show all LiDAR points. Draw 3d box in LiDAR point cloud
        show_lidar_with_boxes(pc_velo, objects, calib, sensor, True, img_width, img_height)
        raw_input()

if __name__=='__main__':
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_gt_boxes3d
    dataset_viz()
