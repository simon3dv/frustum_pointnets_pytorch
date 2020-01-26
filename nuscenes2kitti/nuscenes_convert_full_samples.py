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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version',type=str,default='v1.0-mini')
    parser.add_argument('--start_index', type=int, default='0')
    parser.add_argument('--numbers',type=int,default='-1')
    args = parser.parse_args()

    split = args.version
    start_index = 0
    end_index = start_index + args.numbers
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
    print('Numbers:',args.numbers)
    for present_sample in tqdm(nusc.sample):
        calib = {}
        # converting image data from 6 cameras (in the sensor list)
        for present_sensor in sensor_list:

            sensor_img_output_dir = os.path.join(img_output_root, 'image_' + present_sensor)
            sensor_label_output_dir = os.path.join(label_output_root, 'label_' + present_sensor)
            # each sensor_data corresponds to one specific image in the dataset
            sensor_data = nusc.get('sample_data', present_sample['data'][present_sensor])
            data_path, box_list, cam_intrinsic = nusc.get_sample_data(present_sample['data'][present_sensor], getattr(BoxVisibility,truncation_level))
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
        lidar_path, boxes, cam_intrinsic = nusc.get_sample_data(
            present_sample['data']['LIDAR_TOP'])
        calib['LIDAR_TOP'] = cam_intrinsic

        # Get aggregated point cloud in lidar frame.
        sample_rec = nusc.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']
        ref_chan = 'LIDAR_TOP'
        pc, times = LidarPointCloud.from_file_multisweep(nusc,sample_rec,chan,ref_chan,nsweeps=10)

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
        frame_counter += 1
        seqname_list.append(seqname)
        if frame_counter == end_index:
            break

    with open(sets_root + split + '.txt',"w") as f:
        for seqname in seqname_list:
            f.write(seqname+'\n')
    if args.version == 'v1.0-trainval':
        f_train = open('train.txt','w')
        f_val = open('val.txt','w')
        for i,seqname in enumerate(seqname_list):
            if i < args.numbers/2:
                f_train.write(seqname+'\n')
            else:
                f_val.write(seqname+'\n')
        f_train.close()
        f_val.close()

    print('Done!(saving to {})'.format(os.path.dirname(img_output_root)))
