''' Prepare nuScenes data for 3D object detection.

Author: Siming Fan
Date: January 2020
'''
import os
import sys
import numpy as np
import cv2
from PIL import Image
from matplotlib.axes import Axes
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
#import kitti_util as utils
import _pickle as pickle
from nuscenes_object import *
import argparse
import ipdb
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
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

class Box2d:
    """ Simple data class representing a 2d box including, label, score. """

    def __init__(self,
                 corners: List[float],
                 label: int = np.nan,
                 score: float = np.nan,
                 name: str = None,
                 token: str = None,
                 visibility: str = "0",
                 filename: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(corners))
        assert len(corners) == 4

        self.corners = np.array(corners)
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.name = name
        self.token = token
        self.visibility = visibility
        self.filename = filename
    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """

        axis.plot([self.corners[0], self.corners[0]], [self.corners[1],self.corners[3]], color=colors[0], linewidth=linewidth)
        axis.plot([self.corners[2], self.corners[2]], [self.corners[1], self.corners[3]], color=colors[0],linewidth=linewidth)
        axis.plot([self.corners[0], self.corners[2]], [self.corners[1], self.corners[1]], color=colors[0],linewidth=linewidth)
        axis.plot([self.corners[0], self.corners[2]], [self.corners[3], self.corners[3]], color=colors[0],linewidth=linewidth)


def get_boxes2d(sample_data_token: str, image_annotations_token2ind = {}, image_annotations = []):
    nusc = NuScenes(version='v1.0-mini', dataroot='dataset/nuScenes/v1.0-mini', verbose=True)
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    curr_sample_record = nusc.get('sample', sd_record['sample_token'])
    #curr_sample_record['image_anns']

    boxes2d = []
    if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
        # If no previous annotations available, or if sample_data is keyframe just return the current ones.
        for i,x in enumerate(curr_sample_record['anns']):
            record = nusc.get('sample_annotation', x)
            instance_token = record['instance_token']
            record2d = image_annotations[image_annotations_token2ind[instance_token]]
            box2d = Box2d(record2d['bbox_corners'], name=record2d['category_name'], token=record2d['sample_annotation_token'],
                          visibility=record2d['visibility_token'],filename=record2d['filename'])
            boxes2d.append(box2d)

    else:
        prev_sample_record = nusc.get('sample', curr_sample_record['prev'])

        curr_ann_recs = [nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
        prev_ann_recs = [nusc.get('sample_annotation', token) for token in prev_sample_record['anns']]

        # Maps instance tokens to prev_ann records
        prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

        t0 = prev_sample_record['timestamp']
        t1 = curr_sample_record['timestamp']
        t = sd_record['timestamp']

        # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
        t = max(t0, min(t1, t))

        boxes = []
        for curr_ann_rec in curr_ann_recs:

            if curr_ann_rec['instance_token'] in prev_inst_map:
                # If the annotated instance existed in the previous frame, interpolate center & orientation.
                prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                # Interpolate center.
                center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                             curr_ann_rec['translation'])]

                # Interpolate orientation.
                rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                            q1=Quaternion(curr_ann_rec['rotation']),
                                            amount=(t - t0) / (t1 - t0))

                box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                          token=curr_ann_rec['token'])
            else:
                # If not, simply grab the current annotation.
                box = self.get_box(curr_ann_rec['token'])

            boxes.append(box)
    return boxes2d

def get_color(category_name: str) -> Tuple[int, int, int]:
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    if 'bicycle' in category_name or 'motorcycle' in category_name:
        return 255, 61, 99  # Red
    elif 'vehicle' in category_name or category_name in ['bus', 'car', 'construction_vehicle', 'trailer', 'truck']:
        return 255, 158, 0  # Orange
    elif 'pedestrian' in category_name:
        return 0, 0, 230  # Blue
    elif 'cone' in category_name or 'barrier' in category_name:
        return 0, 0, 0  # Black
    else:
        return 255, 0, 255  # Magenta

def render_image_annotations(sample_data_token: str, with_anns: bool = True, out_path = None,
                             box_vis_level: BoxVisibility = BoxVisibility.ANY,
                             axes_limit: float = 40,ax: Axes = None, image_annotations_token2ind = {},
                             image_annotations = [],visibilities = ['','1','2','3','4'],cam_type = 'CAM_FRONT'):
    _cam_type = '/' + cam_type + '/'
    nusc = NuScenes(version='v1.0-mini', dataroot='dataset/nuScenes/v1.0-mini', verbose=True)
    # Get sensor modality.
    sd_record = nusc.get('sample_data', sample_data_token)
    sensor_modality = sd_record['sensor_modality']#'camera'
    '''
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token,
                                                                   box_vis_level=box_vis_level)
    '''
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    data_path = nusc.get_sample_data_path(sample_data_token)#'dataset/nuScenes/v1.0-mini/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg'

    camera_intrinsic = np.array(cs_record['camera_intrinsic'])
    imsize = (sd_record['width'], sd_record['height'])

    '''
    boxes[0]:
        label: nan, score: nan, 
        xyz: [373.26, 1130.42, 0.80], 
        wlh: [0.62, 0.67, 1.64], 
        rot axis: [0.00, 0.00, -1.00], 
        ang(degrees): 21.09, ang(rad): 0.37, 
        vel: nan, nan, nan, 
        name: human.pedestrian.adult, 
        token: ef63a697930c4b20a6b9791f423351da
    '''
    boxes2d = get_boxes2d(sample_data_token=sample_data_token,
                          image_annotations_token2ind=image_annotations_token2ind,
                          image_annotations=image_annotations)
    '''
    box2d_list = []
    for box2d in boxes2d:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box2d_list.append(box2d)
    '''



    data = Image.open(data_path)

    # Init axes.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 16))

    # Show image.
    ax.imshow(data)
    # Show boxes.
    if with_anns:
        for box2d in boxes2d:
            print(box2d.corners, box2d.name,box2d.filename[8:22])
            print(box2d.token)
            c = np.array(get_color(box2d.name)) / 255.0
            #ipdb.set_trace()
            if box2d.visibility in visibilities and (_cam_type in box2d.filename):
                box2d.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

    # Limit visible range.
    #ax.set_xlim(0, data.size[0])
    #ax.set_ylim(data.size[1], 0)
    #ax.axis('off')
    ax.set_title(sd_record['channel'])
    #ax.set_aspect('equal')

    if out_path is not None:
        plt.savefig(out_path)

def demo():
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    #dataset = nuscenes_object(os.path.join(ROOT_DIR, 'dataset/nuScenes/v1.0-mini'))

    nusc = NuScenes(version='v1.0-mini', dataroot='dataset/nuScenes/v1.0-mini', verbose=True)
    my_scene = nusc.scene[0]
    # Load data from dataset
    #objects = dataset.get_label_objects(data_idx)  # objects = [Object3d(line) for line in lines]
    #objects[0].print_object()
    # 1. scene
    # 2. sample
    first_sample_token = my_scene['first_sample_token']
    # 3. sample_data
    my_sample = nusc.get('sample', first_sample_token)
    '''
     my_sample.keys()                                                                                       
    dict_keys(['token', 'timestamp', 'prev', 'next', 'scene_token', 'data', 'anns'])
    my_sample['data'].keys()                                                                               
    dict_keys(['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])

    '''

    # 3+(New). image_annotations
    image_annotations_token2ind = dict()
    image_annotations = nusc.__load_table__('image_annotations')
    for ind, member in enumerate(image_annotations):
        image_annotations_token2ind[member['instance_token']] = ind
    '''
    sensor = 'CAM_FRONT'
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
    render_image_annotations(cam_front_data['token'],image_annotations_token2ind=image_annotations_token2ind,
                             image_annotations = image_annotations,visibilities = ['','1','2','3','4'],cam_type=sensor)
    sensor = 'CAM_FRONT_LEFT'
    cam_front_left_data = nusc.get('sample_data', my_sample['data'][sensor])
    render_image_annotations(cam_front_left_data['token'],image_annotations_token2ind=image_annotations_token2ind,
                             image_annotations = image_annotations,visibilities = ['','1','2','3','4'],cam_type=sensor)
    nusc.render_sample_data(cam_front_data['token'])
    '''
    sensors = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']
    for sensor in sensors[:2]:
        sensor_data = nusc.get('sample_data', my_sample['data'][sensor])
        render_image_annotations(sensor_data['token'],image_annotations_token2ind=image_annotations_token2ind,
                                 image_annotations = image_annotations,visibilities = ['','1','2','3','4'],cam_type=sensor)
        nusc.render_sample_data(sensor_data['token'])

    # 4. sample_annotation
    #my_annotation_token = my_sample['anns'][0]
    #my_annotation_metadata = nusc.get('sample_annotation', my_annotation_token)
    #print("my_annotation_metadata",my_annotation_metadata)
    #nusc.render_annotation(my_annotation_token)

    # 5. instance
    # 6. category
    # 7. attribute
    # 8. visibility
    # 9. sensor
    # 10. calibrated_sensor
    # 11. ego_pose
    # 12. log
    # 13. map

    nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')
    nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5, underlay_map=True)
    nusc.render_sample_data(my_sample['data']['RADAR_FRONT_RIGHT'], nsweeps=5, underlay_map=True)
    plt.show()

    '''
    calib = dataset.get_calibration(data_idx)  # utils.Calibration(calib_filename)
    box2d = objects[0].box2d
    xmin, ymin, xmax, ymax = box2d
    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    uvdepth = np.zeros((1, 3))
    uvdepth[0, 0:2] = box2d_center
    uvdepth[0, 2] = 20  # some random depth
    box2d_center_rect = calib.project_image_to_rect(uvdepth)
    frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                    box2d_center_rect[0, 0])
    print('frustum_angle:', frustum_angle)
    '''
    '''
    Type, truncation, occlusion, alpha: Pedestrian, 0, 0, -0.200000
    2d bbox (x0,y0,x1,y1): 712.400000, 143.000000, 810.730000, 307.920000
    3d bbox h,w,l: 1.890000, 0.480000, 1.200000
    3d bbox location, ry: (1.840000, 1.470000, 8.410000), 0.010000
    '''
    '''
    img = dataset.get_image(data_idx)  # (370, 1224, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    pc_velo = dataset.get_lidar(data_idx)[:, 0:3]  # (115384, 3)
    calib = dataset.get_calibration(data_idx)  # utils.Calibration(calib_filename)
    '''
    ## Draw lidar in rect camera coord
    # print(' -------- LiDAR points in rect camera coordination --------')
    # pc_rect = calib.project_velo_to_rect(pc_velo)
    # fig = draw_lidar_simple(pc_rect)
    # raw_input()
    # Draw 2d and 3d boxes on image
    #print(' -------- 2D/3D bounding boxes in images --------')
    #show_image_with_boxes(img, objects, calib)
    #raw_input()

    # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    #print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    # show_lidar_with_boxes(pc_velo, objects, calib)
    # raw_input()
    #show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
    #raw_input()

    # Visualize LiDAR points on images
    #print(' -------- LiDAR points projected to image plane --------')
    #show_lidar_on_image(pc_velo, img, calib, img_width, img_height)
    #raw_input()

    # Show LiDAR points that are in the 3d box
    #print(' -------- LiDAR points in a 3D bounding box --------')
    #box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    #box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    #box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    #print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))
    '''
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show(1)
    raw_input()

    # UVDepth Image and its backprojection to point clouds
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
                                                              calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:, 0:2] = imgfov_pts_2d
    cameraUVDepth[:, 2] = imgfov_pc_rect[:, 2]

    # Show that the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print(imgfov_pc_velo[0:20])
    print(backprojected_pc_velo[0:20])

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into 2d box
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    xmin, ymin, xmax, ymax = \
        objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = \
        get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
    print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()
    '''

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx2 = cx + w * r * (np.random.random() * 2 - 1)
    cy2 = cy + h * r * (np.random.random() * 2 - 1)
    h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    return np.array([cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])


def extract_frustum_data(idx_filename, split, output_filename, viz=False,
                         perturb_box2d=False, augmentX=1, type_whitelist=['Car']):
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
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
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

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist: continue

            # 2D BOX: Get pts rect backprojected 
            box2d = objects[obj_idx].box2d
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                    print(box2d)
                    print(xmin, ymin, xmax, ymax)
                else:
                    xmin, ymin, xmax, ymax = box2d
                box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                               (pc_image_coord[:, 0] >= xmin) & \
                               (pc_image_coord[:, 1] < ymax) & \
                               (pc_image_coord[:, 1] >= ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds, :]  # (1607, 4)
                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                box2d_center_rect[0, 0])
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)  # (8, 2)(8, 3)
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

                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / len(id_list)))

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


def get_box3d_dim_statistics(idx_filename):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type == 'DontCare': continue
            dimension_list.append(np.array([obj.l, obj.w, obj.h]))
            type_list.append(obj.type)
            ry_list.append(obj.ry)

    with open('box3d_dimensions.pickle', 'wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
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
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    for det_idx in range(len(det_id_list)):
        data_idx = det_id_list[det_idx]
        print('det idx: %d/%d, data idx: %d' % \
              (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
            pc_velo = dataset.get_lidar(data_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
            pc_rect[:, 3] = pc_velo[:, 3]
            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov( \
                pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
            cache = [calib, pc_rect, pc_image_coord, img_fov_inds]
            cache_id = data_idx
        else:
            calib, pc_rect, pc_image_coord, img_fov_inds = cache

        if det_type_list[det_idx] not in type_whitelist: continue

        # 2D BOX: Get pts rect backprojected 
        xmin, ymin, xmax, ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                       (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & \
                       (pc_image_coord[:, 1] >= ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds, :]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                        box2d_center_rect[0, 0])

        # Pass objects that are too small
        if ymax - ymin < img_height_threshold or \
                len(pc_in_box_fov) < lidar_point_threshold:
            continue

        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)

    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
                              fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:, 0], p1[:, 1], p1[:, 2], p1[:, 1], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
                              fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:, 2], -p1[:, 0], -p1[:, 1], seg, mode='point',
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
        output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()


if __name__ == '__main__':
    # python kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--gen_train', action='store_true',
                        help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection', action='store_true',
                        help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars; otherwise cars, peds and cycs')
    args = parser.parse_args()

    if args.demo:
        demo()  # draw 2d box and 3d box
        exit()
        '''
        python kitti/prepare_data.py --demo
        Type, truncation, occlusion, alpha: Pedestrian, 0, 0, -0.200000
        2d bbox (x0,y0,x1,y1): 712.400000, 143.000000, 810.730000, 307.920000
        3d bbox h,w,l: 1.890000, 0.480000, 1.200000
        3d bbox location, ry: (1.840000, 1.470000, 8.410000), 0.010000
        ('Image shape: ', (370, 1224, 3))
         -------- 2D/3D bounding boxes in images --------
        ('pts_3d_extend shape: ', (8, 4))
        '''

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.gen_train:
        extract_frustum_data( \
            os.path.join(BASE_DIR, 'image_sets/train.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix + 'train.pickle'),
            viz=False, perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist)

    if args.gen_val:
        extract_frustum_data( \
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix + 'val.pickle'),
            viz=False, perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist)

    if args.gen_val_rgb_detection:
        extract_frustum_data_rgb_detection( \
            os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix + 'val_rgb_detection.pickle'),
            viz=False,
            type_whitelist=type_whitelist) 
