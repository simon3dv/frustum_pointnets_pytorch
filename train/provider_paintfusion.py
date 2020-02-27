''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import _pickle as pickle
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,'models'))
from box_util import box3d_iou
from model_util import g_type2class, g_class2type, g_type2onehotclass
from model_util import g_type_mean_size
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from configs.config import cfg
import ipdb
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc

def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle%(2*np.pi)
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = int(shifted_angle/angle_per_class)
    residual_angle = shifted_angle - \
        (class_id * angle_per_class + angle_per_class/2)
    return class_id, residual_angle

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi:
        angle = angle - 2*np.pi
    return angle
        
def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residual.
    todo (rqi): support multiple size clusters per type.
 
    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual

def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual


def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
    return pts_3d_hom

def project_rect_to_image(pts_3d_rect, P):
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = np.dot(pts_3d_rect, np.transpose(P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''
    def __init__(self, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False,
                 gen_ref=False, with_image=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
            gen_ref: bool, if True, generate ref data for fconvnet
        '''
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.gen_ref = gen_ref
        self.from_rgb_detection = from_rgb_detection
        self.with_image = with_image
        self._R_MEAN = 92.8403
        self._G_MEAN = 97.7996
        self._B_MEAN = 93.5843
        """
        self.norm = transforms.Compose([
            transforms.Normalize(mean=(_R_MEAN, _G_MEAN, _B_MEAN), std=(1, 1, 1)),
            #transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
            ]
        )
        """
        self.resize = transforms.Resize(size=(cfg.DATA.W_CROP,cfg.DATA.H_CROP))#,interpolation=Image.NEAREST)
        if from_rgb_detection:
            with open(overwritten_data_path,'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp) 
                self.prob_list = pickle.load(fp)
                self.calib_list = pickle.load(fp)
                if with_image:
                    self.image_filename_list = pickle.load(fp)
                    self.input_2d_list = pickle.load(fp)
        else:
            with open(overwritten_data_path,'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.box3d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.label_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.heading_list = pickle.load(fp)
                self.size_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
                self.calib_list = pickle.load(fp)
                if with_image:
                    self.image_filename_list = pickle.load(fp)
                    self.input_2d_list = pickle.load(fp)

        if gen_ref:
            s1, s2, s3, s4 = cfg.DATA.STRIDE#(0.25, 0.5, 1.0, 2.0)
            self.z1 = np.arange(0, 70, s1) + s1 / 2.#[0.125,0.375,...,69.875]
            self.z2 = np.arange(0, 70, s2) + s2 / 2.#[0.25,0.75,...,69.75]
            self.z3 = np.arange(0, 70, s3) + s3 / 2.#[0.5,1.5,...,69.5]
            self.z4 = np.arange(0, 70, s4) + s4 / 2.#[1,3,...,69]


    def norm(self,x):
        x = x.astype(np.float32)
        _MEAN = [self._R_MEAN,self._G_MEAN,self._B_MEAN]
        x -= _MEAN
        x /= 255
        return x

    def __len__(self):
            return len(self.input_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)
        #np.pi/2.0 + self.frustum_angle_list [index]float,[-pi/2,pi/2]

        box = self.box2d_list[index]

        # Compute one hot vector
        if self.one_hot:#True
            cls_type = self.type_list[index]
            assert(cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:#True
            point_set = self.get_center_view_point_set(index)#(n,4) #pts after Frustum rotation
        else:
            point_set = self.input_list[index]
        pts_2d = self.input_2d_list[index]

        # Get image
        if self.with_image:
            # With whole Image(whole size, not region)
            import time
            #tic = time.perf_counter()
            image_filename = self.image_filename_list[index]
            #image = cv2.imread(image_filename)#(370, 1224, 3),uint8 or (375, 1242, 3)
            image = Image.open(image_filename)
            image = np.array(image)
            image = self.norm(image)
            image = np.array(image)
            n_point = point_set.shape[0]
            point_rgb = np.zeros((n_point,3))
            for n in range(n_point):
                x = int(pts_2d[n,0])
                y = int(pts_2d[n,1])
                if x<0: x = 0
                if y<0: y = 0
                if x >= image.shape[1]: x = image.shape[1] - 1
                if y >= image.shape[0]: y = image.shape[0] - 1
                point_rgb[n,:] = image[y,x,:]
            point_rgb = np.array(point_rgb)
            #if cfg.DATA.BLACK_TEST:
            #    image_crop_resized = np.zeros(image_crop_resized.shape)

            #if cfg.DATA.WHITE_TEST:
            #    image_crop_resized = np.full(image_crop_resized.shape,255)

            """
            query_v1 = np.full(pts_2d.shape[0],-1)
            image_crop_indices_resized = image_crop_indices_resized.reshape(-1)#(45000=H*W)
            for i in range(image_crop_indices_resized.shape[0]):
                if image_crop_indices_resized[i] != -1:
                    query_v1[image_crop_indices_resized[i]] = i
            """

            #print("%.3fs"%(time.perf_counter()-tic))
        # Use extra feature as channel
        if not cfg.DATA.USE_REFLECTION_AS_CHANNEL and not cfg.DATA.USE_RGB_AS_CHANNEL:
            point_set = point_set[:,:3]
        elif cfg.DATA.USE_REFLECTION_AS_CHANNEL and not cfg.DATA.USE_RGB_AS_CHANNEL:
            point_set = point_set[:, :4]
        elif not cfg.DATA.USE_REFLECTION_AS_CHANNEL and cfg.DATA.USE_RGB_AS_CHANNEL:
            point_set = np.concatenate([point_set[:, :3],point_rgb],1)
        point_set = np.concatenate([point_set[:, :3], point_rgb], 1)


        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        if self.gen_ref:#fconvnet
            P = self.calib_list[index]  # 3x4(kitti) or 3x3(nuscenes)
            ref1, ref2, ref3, ref4 = self.generate_ref(box, P)
            if self.rotate_to_center:
                ref1 = self.get_center_view(ref1, index)#(280, 3)
                ref2 = self.get_center_view(ref2, index)#(140, 3)
                ref3 = self.get_center_view(ref3, index)#(70, 3)
                ref4 = self.get_center_view(ref4, index)#(35, 3)

        if self.from_rgb_detection:
            data_inputs = {
                'point_cloud': torch.FloatTensor(point_set).transpose(1, 0),
                'rot_angle': torch.FloatTensor([rot_angle]),
                'rgb_prob': torch.FloatTensor([self.prob_list[index]]),
            }

            if not self.rotate_to_center:
                data_inputs.update({'rot_angle': torch.zeros(1)})

            if self.one_hot:
                data_inputs.update({'one_hot':torch.FloatTensor(one_hot_vec)})

            if self.gen_ref:
                data_inputs.update({'center_ref1': torch.FloatTensor(ref1).transpose(1, 0)})
                data_inputs.update({'center_ref2': torch.FloatTensor(ref2).transpose(1, 0)})
                data_inputs.update({'center_ref3': torch.FloatTensor(ref3).transpose(1, 0)})
                data_inputs.update({'center_ref4': torch.FloatTensor(ref4).transpose(1, 0)})

            if self.with_image:
                data_inputs.update({'P': torch.FloatTensor(P)})

            return data_inputs
        # ------------------------------ LABELS ----------------------------
        if not self.gen_ref:#not fconvnet
            seg = self.label_list[index]
            seg = seg[choice]#(1024,),array([0., 1., 0., ..., 1., 1., 1.])

        # Get center point of 3D box
        if self.rotate_to_center:#True
            box3d_center = self.get_center_view_box3d_center(index) #array([ 0.07968819,  0.39      , 46.06915834])
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:#True
            heading_angle = self.heading_list[index] - rot_angle #-1.6480684951683866 #alpha
        else:
            heading_angle = self.heading_list[index]# rotation_y

        angle_class, angle_residual = angle2class(heading_angle,
            NUM_HEADING_BIN)

        # Size
        size_class, size_residual = size2class(self.size_list[index],
            self.type_list[index]) #5, array([0.25717603, 0.00293633, 0.12301873])
        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random()>0.5: # 50% chance flipping
                point_set[:,0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle

                if self.gen_ref:
                    ref1[:, 0] *= -1
                    ref2[:, 0] *= -1
                    ref3[:, 0] *= -1
                    ref4[:, 0] *= -1

        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
            shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
            point_set[:,2] += shift
            box3d_center[2] += shift

        box3d_size = self.size_list[index]

        data_inputs = {
            'point_cloud': torch.FloatTensor(point_set).transpose(1, 0),
            'rot_angle': torch.FloatTensor([rot_angle]),
            'box3d_center': torch.FloatTensor(box3d_center),
            #'size_class': torch.LongTensor([size_class]),
            #'size_residual': torch.FloatTensor([size_residual]),
            #'angle_class': torch.LongTensor([angle_class]),
            #'angle_residual': torch.FloatTensor([angle_residual]),
        }

        if self.one_hot:
            data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})

        if self.gen_ref:#F-ConvNet
            labels = self.generate_ref_labels(box3d_center, box3d_size, heading_angle, ref2, P)
            data_inputs.update({'ref_label': torch.LongTensor(labels)})
            data_inputs.update({'center_ref1': torch.FloatTensor(ref1).transpose(1, 0)})
            data_inputs.update({'center_ref2': torch.FloatTensor(ref2).transpose(1, 0)})
            data_inputs.update({'center_ref3': torch.FloatTensor(ref3).transpose(1, 0)})
            data_inputs.update({'center_ref4': torch.FloatTensor(ref4).transpose(1, 0)})
            data_inputs.update({'size_class': torch.LongTensor([size_class])})
            data_inputs.update({'box3d_size': torch.FloatTensor(box3d_size)})
            data_inputs.update({'box3d_heading': torch.FloatTensor([heading_angle])})
        else:#F-Pointnets
            data_inputs.update({'seg': seg})
            data_inputs.update({'size_class':torch.LongTensor([size_class])})
            data_inputs.update({'size_residual':torch.FloatTensor([size_residual])})
            data_inputs.update({'angle_class':torch.LongTensor([angle_class])})
            data_inputs.update({'angle_residual':torch.FloatTensor([angle_residual])})

        if self.with_image:
            data_inputs.update({'P': torch.FloatTensor(P)})

        return data_inputs

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0,:] + \
            self.box3d_list[index][6,:])/2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0,:] + \
            self.box3d_list[index][6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), \
            self.get_center_view_rot_angle(index)).squeeze()
        
    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
            self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
            self.get_center_view_rot_angle(index))

    def generate_ref(self, box, P):
        cx, cy = (box[0] + box[2]) / 2., (box[1] + box[3]) / 2.,#678.73,206.76

        xyz1 = np.zeros((len(self.z1), 3))
        xyz1[:, 0] = cx
        xyz1[:, 1] = cy
        xyz1[:, 2] = self.z1
        xyz1_rect = project_image_to_rect(xyz1, P)

        xyz2 = np.zeros((len(self.z2), 3))
        xyz2[:, 0] = cx
        xyz2[:, 1] = cy
        xyz2[:, 2] = self.z2
        xyz2_rect = project_image_to_rect(xyz2, P)

        xyz3 = np.zeros((len(self.z3), 3))
        xyz3[:, 0] = cx
        xyz3[:, 1] = cy
        xyz3[:, 2] = self.z3
        xyz3_rect = project_image_to_rect(xyz3, P)

        xyz4 = np.zeros((len(self.z4), 3))
        xyz4[:, 0] = cx
        xyz4[:, 1] = cy
        xyz4[:, 2] = self.z4
        xyz4_rect = project_image_to_rect(xyz4, P)

        return xyz1_rect, xyz2_rect, xyz3_rect, xyz4_rect

    def generate_ref_labels(self, center, dimension, angle, ref_xyz, P):
        box_corner1 = get_3d_box(dimension * 0.5, angle, center)
        box_corner2 = get_3d_box(dimension, angle, center)

        labels = np.zeros(len(ref_xyz))#(140,)
        _, inside1 = extract_pc_in_box3d(ref_xyz, box_corner1)#(140,)
        _, inside2 = extract_pc_in_box3d(ref_xyz, box_corner2)#(140,)

        labels[inside2] = -1
        labels[inside1] = 1
        # dis = np.sqrt(((ref_xyz - center)**2).sum(1))
        # print(dis.min())
        if inside1.sum() == 0:
            dis = np.sqrt(((ref_xyz - center) ** 2).sum(1))
            argmin = np.argmin(dis)
            labels[argmin] = 1

        return labels

    def get_center_view(self, point_set, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(point_set)
        return rotate_pc_along_y(point_set,
                                 self.get_center_view_rot_angle(index))
# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def compute_box3d_iou(center_pred,
                      heading_logits, heading_residual,
                      size_logits, size_residual,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residual: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residual: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1) # B
    heading_residual = np.array([heading_residual[i,heading_class[i]] \
        for i in range(batch_size)]) # B,
    size_class = np.argmax(size_logits, 1) # B
    size_residual = np.vstack([size_residual[i,size_class[i],:] \
        for i in range(batch_size)])

    iou2d_list = [] 
    iou3d_list = [] 
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
            heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i],
            heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label,
            heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label) 
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
        np.array(iou3d_list, dtype=np.float32)


def from_prediction_to_label_format(center, angle_class, angle_res,\
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l,w,h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx,ty,tz = rotate_pc_along_y(np.expand_dims(center,0),-rot_angle).squeeze()
    ty += h/2.0
    return h,w,l,tx,ty,tz,ry

def project_image_to_rect(uv_depth, P):
    '''
    :param box: nx3 first two channels are uv in image coord, 3rd channel
                is depth in rect camera coord
    :param P: 3x3 or 3x4
    :return: nx3 points in rect camera coord
    '''
    c_u = P[0,2]
    c_v = P[1,2]
    f_u = P[0, 0]
    f_v = P[1, 1]
    if P.shape[1] == 4:
        b_x = P[0, 3] / (-f_u)  # relative
        b_y = P[1, 3] / (-f_v)
    else:
        b_x = 0
        b_y = 0
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
    pts_3d_rect = np.zeros((n, 3), dtype=uv_depth.dtype)
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


if __name__=='__main__':
    gen_ref = True
    show = False
    with_image = True
    import mayavi.mlab as mlab 
    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_lidar, draw_gt_boxes3d
    median_list = []
    dataset = FrustumDataset(1024, split='val',
        rotate_to_center=True, random_flip=True, random_shift=True,
        overwritten_data_path='kitti/frustum_caronly_wimage_val.pickle',
        gen_ref = gen_ref, with_image=with_image)
    for i in range(len(dataset)):
        print(i)
        data_dicts = dataset[i]
        print(data_dicts.keys())
        for key, value in data_dicts.items():
            print('%s:%s,%s'%(key,str(value.shape),str(value.dtype)))
        print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        point_cloud = data_dicts['point_cloud']
        print('max:',point_cloud.max(1)[0])
        print('mean:',point_cloud.mean(1))
        print('min:',point_cloud.min(1)[0])
        if show:
            box3d_from_label = get_3d_box(
                class2size(data_dicts['size_class'].item(),data_dicts['size_residual'].squeeze().numpy()),
                class2angle(data_dicts['angle_class'].item(), data_dicts['angle_residual'].squeeze().numpy(),12),
                data_dicts['box3d_center'].numpy())
            ps = point_cloud.T
            seg = data_dicts['seg']
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(1000, 500))
            mlab.points3d(ps[:,0], ps[:,1], ps[:,2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
            mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2, figure=fig)
            draw_gt_boxes3d([box3d_from_label], fig, color=(1,0,0))
            mlab.orientation_axes()
        input()
    '''
    0
    dict_keys(['point_cloud', 'rot_angle', 'box3d_center', 'size_class', 'size_residual', 'angle_class', 'angle_residual', 'label', 'center_ref1', 'center_ref2', 'center_ref3', 'center_ref4'])
    point_cloud:torch.Size([4, 1024]),torch.float32
    rot_angle:torch.Size([1]),torch.float32
    box3d_center:torch.Size([3]),torch.float32
    size_class:torch.Size([1]),torch.int64
    size_residual:torch.Size([1, 3]),torch.float32
    angle_class:torch.Size([1]),torch.int64
    angle_residual:torch.Size([1]),torch.float32
    label:torch.Size([140]),torch.int64
    center_ref1:torch.Size([3, 280]),torch.float32
    center_ref2:torch.Size([3, 140]),torch.float32
    center_ref3:torch.Size([3, 70]),torch.float32
    center_ref4:torch.Size([3, 35]),torch.float32
    ('Frustum angle: ', -1.4783037599141617)
    max: tensor([ 1.0734,  2.8949, 77.1032,  0.9900])
    mean: tensor([3.6379e-02, 1.7382e+00, 4.1251e+01, 1.2603e-01])
    min: tensor([-1.6102,  0.9591, 33.8156,  0.0000])
    '''
