# Author: https://github.com/zcc31415926/NuScenes2KITTI
# convertion from NuScenes dataset to KITTI format
# inspired by https://github.com/poodarchu/nuscenes_to_kitti_format
# converting only camera captions (JPG files)
# converting all samples in every sequence data

# regardless of attributes indexed 2(if blocked) in KITTI label
# however, object minimum visibility level is adjustable

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
import numpy as np
import cv2
import os
import shutil
import ipdb
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
import argparse
from pyquaternion import Quaternion
category_reflection = \
{
    'human.pedestrian.adult': 'Pedestrian',
    'human.pedestrian.child': 'Pedestrian',
    'human.pedestrian.wheelchair': 'DontCare',
    'human.pedestrian.stroller': 'DontCare',
    'human.pedestrian.personal_mobility': 'DontCare',
    'human.pedestrian.police_officer': 'Pedestrian',
    'human.pedestrian.construction_worker': 'Pedestrian',
    'animal': 'DontCare',
    'vehicle.car': 'Car',
    'vehicle.motorcycle': 'Cyclist',
    'vehicle.bicycle': 'Cyclist',
    'vehicle.bus.bendy': 'Tram',
    'vehicle.bus.rigid': 'Tram',
    'vehicle.truck': 'Truck',
    'vehicle.construction': 'DontCare',
    'vehicle.emergency.ambulance': 'DontCare',
    'vehicle.emergency.police': 'DontCare',
    'vehicle.trailer': 'Tram',
    'movable_object.barrier': 'DontCare',
    'movable_object.trafficcone': 'DontCare',
    'movable_object.pushable_pullable': 'DontCare',
    'movable_object.debris': 'DontCare',
    'static_object.bicycle_rack': 'DontCare',
}

def write_array_to_file(output_f,name,array):
    line = "{}: ".format(name)
    output_f.write(line)
    for i in range(array.shape[0]):
        line = ""
        for j in range(array.shape[1]):
            line += str(array[i, j])
            line += ' '
        if i == array.shape[0] - 1:
            line += '\n'
        output_f.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version',type=str,default='v1.0-mini',
                        help='v1.0-mini/v1.0-trainval/v1.0-test')
    parser.add_argument('--start_index', type=int, default='0')
    parser.add_argument('--number',type=int,default='-1',
                        help='number of frames to generate(-1 means all)')
    parser.add_argument('--CAM_FRONT_only', action='store_true',
                        help='Only generate CAM_FRONT; otherwise six cameras')
    args = parser.parse_args()

    split = args.version
    start_index = 0
    data_root = 'dataset/nuScenes/'+split+'/'
    sets_root = 'dataset/nuScenes2KITTI/image_sets/'
    out_root = 'dataset/nuScenes2KITTI/'
    img_output_root = out_root + split + '/'
    label_output_root = out_root + split + '/'
    velodyne_output_root = out_root + split + '/LIDAR_TOP/'
    calib_output_root = out_root + split + '/calib/'
    min_visibility_level = '2'
    truncation_level = 'ANY'
    '''
        """ Enumerates the various level of box visibility in an image """
        ALL = 0  # Requires all corners are inside the image.
        ANY = 1  # Requires at least one corner visible in the image.
        NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.
    '''
    delete_dontcare_objects = True

    nusc = NuScenes(version=split, dataroot=data_root, verbose=True)
    NUMBER = args.number
    if args.number == -1:
        NUMBER = len(nusc.sample)
    print('Number:',NUMBER)
    end_index = start_index + NUMBER
    if args.CAM_FRONT_only:
        sensor_list = ['CAM_FRONT']
    else:
        sensor_list = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']
    #sensor_list = ['CAM_FRONT']
    frame_counter = start_index

    os.makedirs(sets_root,exist_ok=True)
    if os.path.isdir(img_output_root) == True:
        print('previous image output found. deleting...')
        shutil.rmtree(img_output_root)
    os.makedirs(img_output_root)
    if os.path.isdir(velodyne_output_root) == True:
        print('previous velodyne output found. deleting...')
        shutil.rmtree(velodyne_output_root)
    os.makedirs(velodyne_output_root)
    if os.path.isdir(calib_output_root) == True:
        print('previous calib output found. deleting...')
        shutil.rmtree(calib_output_root)
    os.makedirs(calib_output_root)

    for present_sensor in sensor_list:
        sensor_img_output_dir = os.path.join(img_output_root,'image_'+present_sensor)
        if os.path.isdir(sensor_img_output_dir) == True:
            print('previous image output found. deleting...')
            shutil.rmtree(sensor_img_output_dir)
        os.makedirs(sensor_img_output_dir)
        sensor_label_output_dir = os.path.join(label_output_root,'label_'+present_sensor)
        if os.path.isdir(sensor_label_output_dir) == True:
            print('previous image output found. deleting...')
            shutil.rmtree(sensor_label_output_dir)
        os.makedirs(sensor_label_output_dir)

    print('Running...(saving to {})'.format(os.path.dirname(img_output_root)))
    seqname_list = []

    for present_sample in tqdm(nusc.sample):
        calib = {}
        # converting image data from 6 cameras (in the sensor list)
        for present_sensor in sensor_list:
            img_token = present_sample['data'][present_sensor]

            sd_rec = nusc.get('sample_data', img_token)
            cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
            cam2ego_translation = cs_record['translation']
            cam2ego_rotation = cs_record[
                'rotation']
            ego2global_translation = pose_record['translation']
            ego2global_rotation = pose_record[
                'rotation']

            c2e_t_mat = np.array(cam2ego_translation).reshape(3, 1)
            e2g_t_mat = np.array(ego2global_translation).reshape(3, 1)
            c2e_r_mat = Quaternion(cam2ego_rotation).rotation_matrix  # (3, 3)
            e2g_r_mat = Quaternion(ego2global_rotation).rotation_matrix  # (3, 3)
            calib[present_sensor+'_'+'cam2ego_translation'] = c2e_t_mat
            calib[present_sensor+'_'+'cam2ego_rotation'] = c2e_r_mat
            calib[present_sensor+'_'+'ego2global_translation'] = e2g_t_mat
            calib[present_sensor+'_'+'ego2global_rotation'] = e2g_r_mat

            sensor_img_output_dir = os.path.join(img_output_root, 'image_' + present_sensor)
            sensor_label_output_dir = os.path.join(label_output_root, 'label_' + present_sensor)
            # each sensor_data corresponds to one specific image in the dataset
            sensor_data = nusc.get('sample_data', img_token)
            data_path, box_list, cam_intrinsic = nusc.get_sample_data(img_token, getattr(BoxVisibility,truncation_level))
            """
            get_sample_data:        
                Returns the data path as well as all annotations related to that sample_data.
                Note that the boxes are transformed into the current sensor's coordinate frame.
            """
            calib[present_sensor] = cam_intrinsic
            img_file = data_root + sensor_data['filename']
            seqname = str(frame_counter).zfill(6)
            output_label_file = sensor_label_output_dir  +'/' + seqname + '.txt'
            with open(output_label_file, 'a') as output_f:
                for box in box_list:
                    # obtaining visibility level of each 3D box
                    present_visibility_token = nusc.get('sample_annotation', box.token)['visibility_token']
                    if present_visibility_token > min_visibility_level:
                        if not (category_reflection[box.name] == 'DontCare' and delete_dontcare_objects):
                            w, l, h = box.wlh
                            x, y, z = box.center
                            yaw, pitch, roll = box.orientation.yaw_pitch_roll; yaw = -yaw
                            alpha = yaw - np.arctan2(x, z)
                            box_name = category_reflection[box.name]
                            # projecting 3D points to image plane
                            points_2d = view_points(box.corners(), cam_intrinsic, normalize=True)
                            left_2d = int(min(points_2d[0]))
                            top_2d = int(min(points_2d[1]))
                            right_2d = int(max(points_2d[0]))
                            bottom_2d = int(max(points_2d[1]))

                            # save labels
                            line = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                                box_name, 0, -1, alpha, left_2d, top_2d, right_2d, bottom_2d, h, w, l, x, y+h/2, z, yaw)
                            output_f.write(line)
            # save images
            #if not os.path.getsize(output_label_file):
            #    ipdb.set_trace()
            #    del_cmd = 'rm ' + output_label_file
            #    os.system(del_cmd)
            #else:
            #    cmd = 'cp ' + img_file + ' ' + sensor_img_output_dir + '/' + seqname + '.jpg'
            #    print('copying', sensor_data['filename'], 'to', seqname + '.jpg')
            #    os.system(cmd)
            #    frame_counter += 1
            cmd = 'cp ' + img_file + ' ' + sensor_img_output_dir + '/' + seqname + '.jpg'
            # print('copying', sensor_data['filename'], 'to', seqname + '.jpg')
            os.system(cmd)

            # save lidar
            #nusc.get_sample_data(present_sample['data']['LIDAR_TOP'])
            """ read from api """
            sd_record = nusc.get('sample_data', present_sample['data']['LIDAR_TOP'])

        # Get boxes in lidar frame.
        lidar_token = present_sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', present_sample['data']["LIDAR_TOP"])
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, cam_intrinsic = nusc.get_sample_data(lidar_token)

        lidar2ego_translation = cs_record['translation']#[0.943713, 0.0, 1.84023]
        lidar2ego_rotation = cs_record['rotation']#[0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]
        ego2global_translation = pose_record['translation']#[411.3039349319818, 1180.8903791765097, 0.0]
        ego2global_rotation = pose_record['rotation']#[0.5720320396729045, -0.0016977771610471074, 0.011798001930183783, -0.8201446642457809]

        l2e_t_mat = np.array(lidar2ego_translation).reshape(3,1)
        e2g_t_mat = np.array(ego2global_translation).reshape(3,1)
        l2e_r_mat = Quaternion(lidar2ego_rotation).rotation_matrix#(3, 3)
        e2g_r_mat = Quaternion(ego2global_rotation).rotation_matrix#(3, 3)
        calib['lidar2ego_translation'] = l2e_t_mat
        calib['lidar2ego_rotation'] = l2e_r_mat
        calib['ego2global_translation'] = e2g_t_mat
        calib['ego2global_rotation'] = e2g_r_mat

        # Get aggregated point cloud in lidar frame.
        sample_rec = nusc.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']
        ref_chan = 'LIDAR_TOP'
        #pc, times = LidarPointCloud.from_file_multisweep(nusc,sample_rec,chan,ref_chan,nsweeps=10)
        pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=10)
        #import matplotlib.pyplot as plt
        #fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        #view = np.eye(4)
        #LidarPointCloud.from_file(lidar_path).render_height(axes[0], view=view)
        #plt.show()

        #print(pc.points.shape)#(4,n)
        pc.points[:3, :] = view_points(pc.points[:3, :], np.eye(4), normalize=False)
        pc.points = pc.points.T
        #print(pc.points.shape)#(n,4)
        lidar_path = os.path.join(velodyne_output_root, seqname+".bin")
        pc.points.astype('float32').tofile(open(lidar_path, "wb"))

        info = {
            "lidar_path": lidar_path,
            "token": present_sample["token"],
            # "timestamp": times,
        }
            # = os.path.join(data_root, "sample_10sweeps/LIDAR_TOP",
            #                      present_sample['data']['LIDAR_TOP'] + ".bin")
            #pc.points.astype('float32').tofile(open(lidar_path, "wb"))

        # save calib
        output_calib_file = calib_output_root + seqname + '.txt'
        with open(output_calib_file, 'a') as output_f:
            for sensor in sensor_list:
                line = "{}: ".format(sensor)
                output_f.write(line)
                for i in range(calib[sensor].shape[0]):
                    line = ""
                    for j in range(calib[sensor].shape[1]):
                        line += str(calib[sensor][i,j])
                        line += ' '
                    if i==calib[sensor].shape[0]-1:
                        line+='\n'
                    output_f.write(line)
            write_array_to_file(output_f, 'lidar2ego_translation', calib['lidar2ego_translation'])
            write_array_to_file(output_f, 'lidar2ego_rotation', calib['lidar2ego_rotation'])
            write_array_to_file(output_f, 'ego2global_translation', calib['ego2global_translation'])
            write_array_to_file(output_f, 'ego2global_rotation', calib['ego2global_rotation'])

            for sensor in sensor_list:
                write_array_to_file(output_f, sensor+'_'+'cam2ego_rotation', calib[sensor+'_'+'cam2ego_rotation'])
                write_array_to_file(output_f, sensor+'_'+'cam2ego_translation', calib[sensor+'_'+'cam2ego_translation'])
                write_array_to_file(output_f, sensor+'_'+'ego2global_rotation', calib[sensor+'_'+'ego2global_rotation'])
                write_array_to_file(output_f, sensor+'_'+'ego2global_translation', calib[sensor+'_'+'ego2global_translation'])

        frame_counter += 1
        seqname_list.append(seqname)
        if frame_counter == end_index:
            break

    with open(sets_root + split + '.txt',"w") as f:
        for seqname in seqname_list:
            f.write(seqname+'\n')
    if args.version == 'v1.0-trainval':
        f_train = open(sets_root + 'train.txt','w')
        f_val = open(sets_root + 'val.txt','w')
        for i,seqname in enumerate(seqname_list):
            if i < NUMBER/2:
                f_train.write(seqname+'\n')
            else:
                f_val.write(seqname+'\n')
        f_train.close()
        f_val.close()

    print('Done!(saving to {})'.format(os.path.dirname(img_output_root)))
