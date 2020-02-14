""" Helper methods for loading and parsing nuscenes2kitti data.

Authod: Siming Fan
Acknowledge: Charles R. Qi
Date: Jan 2020
"""
from __future__ import print_function

import os

import cv2
import numpy as np
import ipdb
import mayavi.mlab as mlab
class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
            (self.t[0],self.t[1],self.t[2],self.ry))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.


            P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
            P2:
            P2: 7.070493000000e+02 0               6.040814000000e+02 4.575831000000e+01
            0            7.070493000000e+02 1.805066000000e+02 -3.454157000000e-01
            0            0                  1                 4.981016000000e-03
            cam_intrinsic(CAM_FRONT):
            CAM_FRONT:
            1266.417203046554 0.0 816.2670197447984
            0.0 1266.417203046554 491.50706579294757
            0.0 0.0 1.0

    '''
    def __init__(self, calib_filepath, from_video=False, sensor_list = ['CAM_FRONT']):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from global coord to image2 coord
        self.sensor_list = sensor_list
        if 'CAM_FRONT' in self.sensor_list:
            self.CAM_FRONT = np.reshape(calibs['CAM_FRONT'], [3, 3])
        if 'CAM_BACK' in self.sensor_list:
            self.CAM_BACK = np.reshape(calibs['CAM_BACK'], [3, 3])
        if 'CAM_FRONT_LEFT' in self.sensor_list:
            self.CAM_FRONT_LEFT = np.reshape(calibs['CAM_FRONT_LEFT'], [3, 3])
        if 'CAM_BACK_LEFT' in self.sensor_list:
            self.CAM_BACK_LEFT = np.reshape(calibs['CAM_BACK_LEFT'], [3, 3])
        if 'CAM_FRONT_RIGHT' in self.sensor_list:
            self.CAM_FRONT_RIGHT = np.reshape(calibs['CAM_FRONT_RIGHT'], [3, 3])
        if 'CAM_BACK_RIGHT' in self.sensor_list:
            self.CAM_BACK_RIGHT = np.reshape(calibs['CAM_BACK_RIGHT'], [3, 3])
        self.lidar2ego_translation = np.reshape(calibs['lidar2ego_translation'], [3, 1])
        self.lidar2ego_rotation = np.reshape(calibs['lidar2ego_rotation'], [3, 3])
        self.ego2global_translation = np.reshape(calibs['ego2global_translation'], [3, 1])
        self.ego2global_rotation = np.reshape(calibs['ego2global_rotation'], [3, 3])
        for sensor in self.sensor_list:
            for m in [ 'cam2ego_translation','ego2global_translation']:
                attrt = sensor + '_'+ m
                exec('self.'+attrt+' = np.reshape(calibs["'+attrt+'"],[3,1])')
            for m in ['cam2ego_rotation','ego2global_rotation']:
                attrt = sensor + '_'+ m
                exec('self.'+attrt+' = np.reshape(calibs["'+attrt+'"],[3,3])')
        #self.CAM_FRONT = calibs['CAM_FRONT']
        #self.CAM_FRONT = np.reshape(self.CAM_FRONT, [3, 3])
        # Rigid transform from Velodyne coord to reference camera coord
        # self.V2C = calibs['Tr_velo_to_cam']
        # self.V2C = np.reshape(self.V2C, [3,4])
        # self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        # self.R0 = calibs['R0_rect']
        # self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        #self.c_u = self.CAM_FRONT[0,2]
        #self.c_v = self.CAM_FRONT[1,2]
        #self.f_u = self.CAM_FRONT[0,0]
        #self.f_v = self.CAM_FRONT[1,1]
        # self.b_x = self.P[0,3]/(-self.f_u) # relative
        # self.b_y = self.P[1,3]/(-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data
    
    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3,4))
        Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
        Tr_velo_to_cam[:,3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
 
    # =========================== 
    # ------- 3d to 3d ---------- 
    # ===========================
    # input:3xn
    # output:3xn
    # tips: not nx3!
    '''
    def translate(self, points, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            points[i, :] = points[i, :] + x[i]
        return points
    '''
    def translate(self, points, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        pts = points.copy()
        for i in range(3):
            pts[i, :] = pts[i, :] + x[i]
        return pts

    def rotate(self, points, rot_matrix):
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        return np.dot(rot_matrix, points[:, :])

    # ====lidar - ego(lidar) - global - ego_cam - cam====
    def project_lidar_to_ego(self, pts_3d_velo):
        pts_3d_ego = self.rotate(pts_3d_velo, getattr(self, 'lidar2ego_rotation'))
        pts_3d_ego = self.translate(pts_3d_ego, getattr(self, 'lidar2ego_translation'))
        return pts_3d_ego

    def project_ego_to_lidar(self, pts_3d_ego):
        pts_3d_velo = self.translate(pts_3d_ego, -getattr(self, 'lidar2ego_translation'))
        pts_3d_velo = self.rotate(pts_3d_velo, getattr(self, 'lidar2ego_rotation').T)
        return pts_3d_velo

    def project_ego_to_global(self, pts_3d_ego):
        pts_3d_global = self.rotate(pts_3d_ego, getattr(self, 'ego2global_rotation'))
        pts_3d_global = self.translate(pts_3d_global, getattr(self, 'ego2global_translation'))
        return pts_3d_global

    def project_global_to_ego(self, pts_3d_global):
        pts_3d_ego = self.translate(pts_3d_global, -getattr(self, 'ego2global_translation'))
        pts_3d_ego = self.rotate(pts_3d_ego, getattr(self, 'ego2global_rotation').T)
        return pts_3d_ego

    def project_cam_to_ego(self, pts_3d_cam, sensor):
        pts_3d_ego_cam = self.rotate(pts_3d_cam, getattr(self, sensor + '_' + 'cam2ego_rotation'))
        pts_3d_ego_cam = self.translate(pts_3d_ego_cam, getattr(self,sensor+'_'+'cam2ego_translation'))
        return pts_3d_ego_cam

    def project_ego_to_cam(self, pts_3d_ego_cam, sensor):
        pts_3d_cam = self.translate(pts_3d_ego_cam, -getattr(self,sensor+'_'+'cam2ego_translation'))
        pts_3d_cam = self.rotate(pts_3d_cam, getattr(self, sensor + '_' + 'cam2ego_rotation').T)
        return pts_3d_cam

    def project_ego_to_global_cam(self, pts_3d_ego_cam, sensor):
        pts_3d_global_cam = self.rotate(pts_3d_ego_cam, getattr(self, sensor + '_' + 'ego2global_rotation'))
        pts_3d_global_cam = self.translate(pts_3d_global_cam, getattr(self,sensor+'_'+'ego2global_translation'))
        return pts_3d_global_cam

    def project_global_to_ego_cam(self, pts_3d_global_cam, sensor):
        pts_3d_ego_cam = self.translate(pts_3d_global_cam, -getattr(self,sensor+'_'+'ego2global_translation'))
        pts_3d_ego_cam = self.rotate(pts_3d_ego_cam, getattr(self, sensor + '_' + 'ego2global_rotation').T)
        return pts_3d_ego_cam

    # ====lidar - global - cam====
    def project_global_to_lidar(self, pts_3d_global):
        pts_3d_ego = self.project_global_to_ego(pts_3d_global)
        pts_3d_velo = self.project_ego_to_lidar(pts_3d_ego)
        return pts_3d_velo

    def project_lidar_to_global(self, pts_3d_velo):
        pts_3d_ego = self.project_lidar_to_ego(pts_3d_velo)
        pts_3d_global = self.project_ego_to_global(pts_3d_ego)
        return pts_3d_global

    def project_cam_to_global(self, pts_3d_cam, sensor):
        pts_3d_ego_cam = self.project_cam_to_ego(pts_3d_cam, sensor)
        pts_3d_global_cam = self.project_ego_to_global_cam(pts_3d_ego_cam, sensor)
        return pts_3d_global_cam

    def project_global_to_cam(self, pts_3d_global_cam, sensor):
        pts_3d_ego_cam = self.project_global_to_ego_cam(pts_3d_global_cam, sensor)
        pts_3d_cam = self.project_ego_to_cam(pts_3d_ego_cam, sensor)
        return pts_3d_cam


    #=========intrinsic=========#
    def project_image_to_cam(self, uv_depth, sensor):
        ''' Input: 3xn first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: 3xn points in (rect) camera coord.
        '''
        # Camera intrinsics and extrinsics
        c_u = getattr(self,sensor)[0,2]
        c_v = getattr(self,sensor)[1,2]
        f_u = getattr(self,sensor)[0,0]
        f_v = getattr(self,sensor)[1,1]
        n = uv_depth.shape[1]
        x = ((uv_depth[0,:]-c_u)*uv_depth[2,:])/f_u
        y = ((uv_depth[1,:]-c_v)*uv_depth[2,:])/f_v
        pts_3d_cam = np.zeros((3,n))
        pts_3d_cam[0,:] = x
        pts_3d_cam[1,:] = y
        pts_3d_cam[2,:] = uv_depth[2,:]
        return pts_3d_cam

    def project_cam_to_image(self, pts_3d_cam, sensor):
        pts_2d = view_points(pts_3d_cam[:3, :], getattr(self,sensor), normalize=True)#(3,n)
        return pts_2d

    """
    def project_global_to_velo(self, pts_3d_global):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        ''' 
        #pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        #return self.project_ref_to_velo(pts_3d_ref)
        pts_3d_velo = pts_3d_global[:,[0,2,1]]
        pts_3d_velo[:,2] *= -1
        return pts_3d_velo
    """

def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def load_image(img_filename):
    return cv2.imread(img_filename)

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n,1))))
    print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def compute_box_3d(obj,view):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''

    # compute rotational matrix around yaw axis
    R = roty(obj.ry)    

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    # x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    # y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    # z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.t[0];
    corners_3d[1,:] = corners_3d[1,:] + obj.t[1];
    corners_3d[2,:] = corners_3d[2,:] + obj.t[2];
    #print 'cornsers_3d: ', corners_3d


    # only draw 3d bounding box for objs in front of the camera
    '''
    if np.any(corners_3d[2,:]<0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)
    '''
    # project the 3d bounding box into the image plane
    # corners_2d = project_to_image(np.transpose(corners_3d), P);
    #sensor = 'CAM_FRONT'
    #view = getattr(calib,sensor)# 3x3
    corners_2d = view_points(corners_3d, view, normalize=True)[:2, :].T#2x8, mean=590.067...
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, view):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''
    
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)
   
    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l],[0,0],[0,0]])
    
    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0,:] = orientation_3d[0,:] + obj.t[0]
    orientation_3d[1,:] = orientation_3d[1,:] + obj.t[1]
    orientation_3d[2,:] = orientation_3d[2,:] + obj.t[2]
    
    # vector behind image plane?
    if np.any(orientation_3d[2,:]<0.1):
      orientation_2d = None
      return orientation_2d, np.transpose(orientation_3d)
    
    # project orientation into the image plane
    # orientation_2d = project_to_image(np.transpose(orientation_3d), P);
    orientation_2d = view_points(orientation_3d, view, normalize=True)[:2, :]
    return orientation_2d, np.transpose(orientation_3d)

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    return image

def draw_nusc_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:, 2]
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
                  scale_factor=pts_scale, figure=fig)

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

    # draw fov (todo: update to real sensor spec.)
    fov = np.array([  # 45 degree
        [20., 20., 0., 0.],
        [-20., 20., 0., 0.],
    ], dtype=np.float64)

    mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)
    mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)

    # draw square region
    TOP_Y_MIN = 0#-20
    TOP_Y_MAX = 40#20
    TOP_X_MIN = -20#0
    TOP_X_MAX = 20#40
    TOP_Z_MIN = -2.0
    TOP_Z_MAX = 0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)

    # mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig