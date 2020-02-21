import os
import sys
import argparse
import importlib
import numpy as np
import torch
import _pickle as pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import provider
from torch.utils.data import DataLoader
from tqdm import tqdm
import ipdb
from model_util import FrustumPointNetLoss
import time
import torch.nn.functional as F
from configs.config import cfg
from configs.config import merge_cfg_from_file
from configs.config import merge_cfg_from_list
from configs.config import assert_and_infer_cfg
from utils import import_from_file
from ops.pybind11.rbbox_iou import cube_nms_np
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='cfgs/fpointnet/fpointnet_v1_kitti.yaml', help='Config file for training (and optionally testing)')
parser.add_argument('opts',help='See configs/config.py for all options',default=None,nargs=argparse.REMAINDER)
parser.add_argument('--debug', default=False, action='store_true',help='debug mode')
args = parser.parse_args()
if args.cfg is not None:
    merge_cfg_from_file(args.cfg)

if args.opts is not None:
    merge_cfg_from_list(args.opts)

assert_and_infer_cfg()

if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

# Set configurations
CONFIG_FILE = args.cfg
RESUME = cfg.RESUME
OUTPUT_DIR = cfg.OUTPUT_DIR
USE_TFBOARD = cfg.USE_TFBOARD
NUM_WORKERS = cfg.NUM_WORKERS
FROM_RGB_DET = cfg.FROM_RGB_DET
SAVE_SUB_DIR = cfg.SAVE_SUB_DIR
SAVE_DIR = os.path.join(OUTPUT_DIR, SAVE_SUB_DIR)
## TRAIN
TRAIN_FILE = cfg.TRAIN.FILE
BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
START_EPOCH = cfg.TRAIN.START_EPOCH
MAX_EPOCH = cfg.TRAIN.MAX_EPOCH
OPTIMIZER = cfg.TRAIN.OPTIMIZER
BASE_LR = cfg.TRAIN.BASE_LR
MIN_LR = cfg.TRAIN.MIN_LR
GAMMA = cfg.TRAIN.GAMMA
LR_STEPS = cfg.TRAIN.LR_STEPS
MOMENTUM = cfg.TRAIN.MOMENTUM
WEIGHT_DECAY = cfg.TRAIN.WEIGHT_DECAY
NUM_POINT = cfg.TRAIN.NUM_POINT
TRAIN_SETS = cfg.TRAIN.TRAIN_SETS
## TEST
TEST_WEIGHTS = cfg.TEST.WEIGHTS
TEST_FILE = cfg.TEST.FILE
TEST_BATCH_SIZE = cfg.TEST.BATCH_SIZE
TEST_NUM_POINT = cfg.TEST.NUM_POINT
TEST_SETS = cfg.TEST.TEST_SETS
## MODEL
MODEL_FILE = cfg.MODEL.FILE
NUM_CLASSES = cfg.MODEL.NUM_CLASSES
## DATA
DATA_FILE = cfg.DATA.FILE
DATASET = cfg.DATA.DATASET
DATAROOT = cfg.DATA.DATA_ROOT
OBJTYPE = cfg.DATA.OBJTYPE
SENSOR = cfg.DATA.SENSOR
ROTATE_TO_CENTER = cfg.DATA.ROTATE_TO_CENTER
NUM_CHANNEL = cfg.DATA.NUM_CHANNEL
NUM_SAMPLES = cfg.DATA.NUM_SAMPLES
strtime = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
if 'nuscenes' in DATASET:
    NAME = DATASET + '_' + OBJTYPE + '_' + SENSOR + '_' + strtime
else:
    NAME = DATASET + '_' + OBJTYPE + '_' + strtime

MODEL = import_from_file(MODEL_FILE)  # import network module
LOG_DIR = OUTPUT_DIR + '/' + NAME
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'test.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(args) + '\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

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
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc = pc.copy()
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc

def from_prediction_to_label_format(center, angle, size, rot_angle, ref_center=None):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = size
    ry = angle + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()

    if ref_center is not None:
        tx = tx + ref_center[0]
        ty = ty + ref_center[1]
        tz = tz + ref_center[2]

    ty += h / 2.0
    return h, w, l, tx, ty, tz, ry

def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

def write_detection_results(output_dir, det_results):

    results = {}  # map from idx to list of strings, each string is a line (without \n)
    for idx in det_results:
        for class_type in det_results[idx]:
            dets = det_results[idx][class_type]
            for i in range(len(dets)):
                output_str = class_type + " -1 -1 -10 "
                box2d = dets[i][:4]
                output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
                tx, ty, tz, h, w, l, ry = dets[i][4:-1]
                score = dets[i][-1]
                output_str += "%f %f %f %f %f %f %f %f" % (h, w, l, tx, ty, tz, ry, score)
                if idx not in results:
                    results[idx] = []
                results[idx].append(output_str)

    result_dir = os.path.join(output_dir, 'data')
    os.system('rm -rf %s' % (result_dir))
    os.mkdir(result_dir)

    for idx in results:
        pred_filename = os.path.join(result_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()

    # Make sure for each frame (no matter if we have measurement for that frame),
    # there is a TXT file
    idx_path = 'kitti/image_sets/%s.txt' % cfg.TEST.TEST_SETS

    to_fill_filename_list = [line.rstrip() + '.txt' for line in open(idx_path)]
    fill_files(result_dir, to_fill_filename_list)


def write_detection_results_nms(output_dir, det_results, threshold=cfg.TEST.THRESH):

    nms_results = {}
    for idx in det_results:
        for class_type in det_results[idx]:
            dets = np.array(det_results[idx][class_type], dtype=np.float32)
            # scores = dets[:, -1]
            # keep = (scores > 0.001).nonzero()[0]
            # print(len(scores), len(keep))
            # dets = dets[keep]
            if len(dets) > 1:
                dets_for_nms = dets[:, 4:][:, [0, 1, 2, 5, 4, 3, 6, 7]]
                keep = cube_nms_np(dets_for_nms, threshold)
                # print(len(dets_for_nms), len(keep))
                dets_keep = dets[keep]
            else:
                dets_keep = dets
            if idx not in nms_results:
                nms_results[idx] = {}
            # if class_type not in nms_results[idx]:
            #     nms_results[idx][class_type] = []
            nms_results[idx][class_type] = dets_keep

    write_detection_results(output_dir, nms_results)

def test_one_epoch(model, loader):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''

    ps_list = []
    seg_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []

    test_idxs = np.arange(0, len(TEST_DATASET))
    batch_size = BATCH_SIZE
    num_batches = len(TEST_DATASET)//batch_size

    test_n_samples = 0
    pos_cnt = 0.0
    pos_pred_cnt = 0.0
    all_cnt = 0.0
    eval_time = 0.0
    max_info = np.zeros(3)
    min_info = np.zeros(3)
    mean_info = np.zeros(3)

    for i, data_dicts in tqdm(enumerate(loader), \
                              total=len(loader), smoothing=0.9):
        # for debug
        if FLAGS.debug == True:
            if i == 1:
                break

        # 1. Load data
        data_dicts_var = {key: value.squeeze().cuda() for key, value in data_dicts.items()}
        ipdb.set_trace()
        test_n_samples += data_dicts['point_cloud'].shape[0]


        # 2. Eval one batch
        model = model.eval()
        eval_t1 = time.perf_counter()
        outputs = model(data_dicts_var)
        eval_t2 = time.perf_counter()
        eval_time += (eval_t2 - eval_t1)
        # 3. Compute Loss(deprecated)

        # 4. compute seg acc, IoU and acc(IoU) (deprecated)

        # 5. write all Results
        if 'frustum_poinrnets' in cfg.MODEL:
            print('Deprecated')
            exit(0)
        elif 'frustum_convnet' in cfg.MODEL:
            cls_probs, center_preds, heading_preds, size_preds = outputs
            num_pred = cls_probs.shape[1]

            cls_probs = cls_probs.data.cpu().numpy()
            center_preds = center_preds.data.cpu().numpy()
            heading_preds = heading_preds.data.cpu().numpy()
            size_preds = size_preds.data.cpu().numpy()


        #ipdb.set_trace()
        # batch_scores = heading_prob
        for j in range(batch_output.shape[0]):
            ps_list.append(batch_data[j, ...])
            seg_list.append(batch_label[j, ...])
            segp_list.append(batch_output[j, ...])
            center_list.append(batch_center_pred[j, :])
            heading_cls_list.append(batch_hclass_pred[j])
            heading_res_list.append(batch_hres_pred[j])
            size_cls_list.append(batch_sclass_pred[j])
            size_res_list.append(batch_sres_pred[j, :])
            rot_angle_list.append(batch_rot_angle[j])
            score_list.append(batch_scores[j])
            pos_cnt += np.sum(batch_label[j,:].cpu().detach().numpy())
            pos_pred_cnt += np.sum(batch_output[j, :])
            pts_np = batch_data[j,:3,:].cpu().detach().numpy()#(3,1024)
            max_xyz = np.max(pts_np,axis=1)
            max_info= np.maximum(max_info,max_xyz)
            min_xyz = np.min(pts_np,axis=1)
            min_info= np.minimum(min_info,min_xyz)
            mean_info += np.sum(pts_np,axis=1)
    '''
    return np.argmax(logits, 2), centers, heading_cls, heading_res, \
        size_cls, size_res, scores
        
	batch_output, batch_center_pred, \
        batch_hclass_pred, batch_hres_pred, \
        batch_sclass_pred, batch_sres_pred, batch_scores = \
            inference(sess, ops, batch_data,
                batch_one_hot_vec, batch_size=batch_size)
    '''
    if FLAGS.dump_result:
        print('dumping...')
        with open(output_filename, 'wp') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(seg_list, fp)
            pickle.dump(segp_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)

    # Write detection results for KITTI evaluation
    print('Number of point clouds: %d' % (len(ps_list)))
    write_detection_results(result_dir, TEST_DATASET.id_list,
        TEST_DATASET.type_list, TEST_DATASET.box2d_list,
        center_list, heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list)
    # Make sure for each frame (no matter if we have measurment for that frame),
    # there is a TXT file
    output_dir = os.path.join(result_dir, 'data')
    if FLAGS.idx_path is not None:
        to_fill_filename_list = [line.rstrip()+'.txt' \
            for line in open(FLAGS.idx_path)]
        fill_files(output_dir, to_fill_filename_list)

    all_cnt = FLAGS.num_point * len(ps_list)
    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average pos prediction ratio: %f' % (pos_pred_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / len(ps_list)))
    mean_info = mean_info / len(ps_list) / FLAGS.num_point
    print('Mean points: x%f y%f z%f' % (mean_info[0],mean_info[1],mean_info[2]))
    print('Max points: x%f y%f z%f' % (max_info[0],max_info[1],max_info[2]))
    print('Min points: x%f y%f z%f' % (min_info[0],min_info[1],min_info[2]))
    '''
    2020.2.9
    
    nuscenes->nuscenes:
     python train/test.py --model_path log/2020-01-29-21\:40\:20_1-_caronly_nuscenes2kitti_CAM_FRONT_/2020-01-29-21\:40\:20_1-_caronly_nuscenes2kitti_CAM_FRONT_-acc0.242665-epoch114.pth --data_path nuscenes2kitti/frustum_caronly_CAM_FRONT_val.pickle --idx_path nuscenes2kitti/image_sets/val.txt --dataset nuscenes2kitti --output train/n2n/test_result
    
    Number of point clouds: 6408
    Average pos ratio: 0.384588
    Average pos prediction ratio: 0.377492
    Average npoints: 1024.000000
    Mean points: x-0.064339 y0.845485 z35.268281
    Max points: x69.445435 y9.607244 z104.238876
    Min points: x-67.831200 y-14.161563 z0.000000
    test from 2d gt: Done
    test loss: 0.951463
    test segmentation accuracy: 0.841936
    test box IoU(ground/3D): 0.562068/0.482008
    test box estimation accuracy (IoU=0.7): 0.235487
    
    train/kitti_eval/evaluate_object_3d_offline dataset/nuScenes2KITTI/training/label_CAM_FRONT/ train/n2n/test_result/
    
    car_detection AP: 90.909096 90.909096 90.909096
    car_detection_ground AP: 21.611404 20.712748 20.712748
    car_detection_3d AP: 7.032246 6.656452 6.656452

    ---
    kitti->kitti:
    Number of point clouds: 12538
    Average pos ratio: 0.436596
    Average pos prediction ratio: 0.475597
    Average npoints: 1024.000000
    Mean points: x0.025302 y0.986846 z24.966636
    Max points: x16.992741 y9.347202 z79.747406
    Min points: x-17.201658 y-3.882158 z0.000000
    test from 2d gt: Done
    test loss: 0.104107
    test segmentation accuracy: 0.901423
    test box IoU(ground/3D): 0.795341/0.743109
    test box estimation accuracy (IoU=0.7): 0.761206
    
    car_detection AP: 100.000000 100.000000 100.000000
    car_detection_ground AP: 73.766235 72.695000 68.604942
    car_detection_3d AP: 56.872833 56.373001 53.764130

    ---
    kitti->nuscenes:
    Number of point clouds: 6408
    Average pos ratio: 0.384301
    Average pos prediction ratio: 0.000000
    Average npoints: 1024.000000
    Mean points: x-0.064567 y0.845504 z35.269298
    Max points: x69.445435 y8.194828 z104.238876
    Min points: x-75.259071 y-14.069223 z0.000000
    test from 2d gt: Done
    test loss: 25.181181
    test segmentation accuracy: 0.615699
    test box IoU(ground/3D): 0.023954/0.019012
    test box estimation accuracy (IoU=0.7): 0.000000

    train/kitti_eval/evaluate_object_3d_offline dataset/nuScenes2KITTI/training/label_CAM_FRONT/ train/k2n/test_result/
    car_detection AP: 90.909096 90.909096 90.909096
    car_detection_ground AP: 0.000000 0.000000 0.000000
    car_detection_3d AP: 0.000000 0.000000 0.000000


    nuscenes->kitti:
    Number of point clouds: 12538
    Average pos ratio: 0.436536
    Average pos prediction ratio: 0.613729
    Average npoints: 1024.000000
    Mean points: x0.026297 y0.987077 z24.957791
    Max points: x17.198647 y9.372097 z79.747406
    Min points: x-19.410744 y-3.865781 z0.000000
    test from 2d gt: Done
    test loss: 0.661662
    test segmentation accuracy: 0.676350
    test box IoU(ground/3D): 0.511144/0.446354
    test box estimation accuracy (IoU=0.7): 0.174509

    
    car_detection AP: 100.000000 100.000000 100.000000
    car_detection_ground AP: 11.732146 17.377474 15.602872
    car_detection_3d AP: 4.295296 4.503248 4.734365


    '''
    if FLAGS.return_all_loss:
        return test_total_loss / test_n_samples, \
               test_iou2d / test_n_samples, \
               test_iou3d / test_n_samples, \
               test_acc / test_n_samples, \
               test_iou3d_acc / test_n_samples,\
               test_mask_loss / test_n_samples, \
               test_center_loss / test_n_samples, \
               test_heading_class_loss / test_n_samples, \
               test_size_class_loss / test_n_samples, \
               test_heading_residuals_normalized_loss / test_n_samples, \
               test_size_residuals_normalized_loss / test_n_samples, \
               test_stage1_center_loss / test_n_samples, \
               test_corners_loss / test_n_samples
    else:
        return test_total_loss/test_n_samples,  \
               test_iou2d/test_n_samples, \
               test_iou3d/test_n_samples, \
               test_acc/test_n_samples, \
               test_iou3d_acc/test_n_samples

def evaluate_py_wrapper(output_dir, async_eval=False):
    # official evaluation
    gt_dir = 'dataset/KITTI/object/training/label_2/'
    command_line = './train/kitti_eval/evaluate_object_3d_offline %s %s' % (gt_dir, output_dir)
    command_line += ' 2>&1 | tee -a  %s/log_test.txt' % (os.path.join(output_dir))
    print(command_line)
    if async_eval:
        subprocess.Popen(command_line, shell=True)
    else:
        if os.system(command_line) != 0:
            assert False

def test(model, test_dataset, test_loader, output_filename, result_dir=None):

    load_batch_size = test_loader.batch_size
    num_batches = len(test_loader)

    model.eval()

    test_time_total = 0.0

    det_results = {}

    for i, data_dicts in enumerate(test_loader):

        point_clouds = data_dicts['point_cloud']
        rot_angles = data_dicts['rot_angle']
        # optional
        ref_centers = data_dicts.get('ref_center')
        rgb_probs = data_dicts.get('rgb_prob')

        # from ground truth box detection
        if rgb_probs is None:
            rgb_probs = torch.ones_like(rot_angles)

        # not belong to refinement stage
        if ref_centers is None:
            ref_centers = torch.zeros((point_clouds.shape[0], 3))

        batch_size = point_clouds.shape[0]
        rot_angles = rot_angles.view(-1)
        rgb_probs = rgb_probs.view(-1)

        if 'box3d_center' in data_dicts:
            data_dicts.pop('box3d_center')

        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}

        torch.cuda.synchronize()
        tic = time.time()
        with torch.no_grad():
            outputs = model(data_dicts_var)

        cls_probs, center_preds, heading_preds, size_preds = outputs

        torch.cuda.synchronize()
        batch_time = time.time() - tic
        test_time_total += batch_time

        num_pred = cls_probs.shape[1]
        log_string('%d/%d %.3f' % (i, num_batches, batch_time))

        cls_probs = cls_probs.data.cpu().numpy()
        center_preds = center_preds.data.cpu().numpy()
        heading_preds = heading_preds.data.cpu().numpy()
        size_preds = size_preds.data.cpu().numpy()

        rgb_probs = rgb_probs.numpy()
        rot_angles = rot_angles.numpy()
        ref_centers = ref_centers.numpy()

        for b in range(batch_size):

            if cfg.TEST.METHOD == 'nms':
                fg_idx = (cls_probs[b, :, 0] < cls_probs[b, :, 1]).nonzero()[0]
                if fg_idx.size == 0:
                    fg_idx = np.argmax(cls_probs[b, :, 1])
                    fg_idx = np.array([fg_idx])
            else:
                fg_idx = np.argmax(cls_probs[b, :, 1])
                fg_idx = np.array([fg_idx])

            num_pred = len(fg_idx)

            single_centers = center_preds[b, fg_idx]
            single_headings = heading_preds[b, fg_idx]
            single_sizes = size_preds[b, fg_idx]
            single_scores = cls_probs[b, fg_idx, 1] + rgb_probs[b]

            data_idx = test_dataset.id_list[load_batch_size * i + b]
            class_type = test_dataset.type_list[load_batch_size * i + b]
            box2d = test_dataset.box2d_list[load_batch_size * i + b]
            rot_angle = rot_angles[b]
            ref_center = ref_centers[b]

            if data_idx not in det_results:
                det_results[data_idx] = {}

            if class_type not in det_results[data_idx]:
                det_results[data_idx][class_type] = []

            for n in range(num_pred):
                x1, y1, x2, y2 = box2d
                score = single_scores[n]
                h, w, l, tx, ty, tz, ry = from_prediction_to_label_format(
                    single_centers[n], single_headings[n], single_sizes[n], rot_angle, ref_center)
                output = [x1, y1, x2, y2, tx, ty, tz, h, w, l, ry, score]
                det_results[data_idx][class_type].append(output)

    num_images = len(det_results)

    log_string('Average time:')
    log_string(('avg_per_batch:%0.3f' % (test_time_total / num_batches)))
    log_string(('avg_per_object:%0.3f' % (test_time_total / num_batches / load_batch_size)))
    log_string(('avg_per_image:%.3f' % (test_time_total / num_images)))

    # Write detection results for KITTI evaluation

    if cfg.TEST.METHOD == 'nms':
        write_detection_results_nms(result_dir, det_results)
    else:
        write_detection_results(result_dir, det_results)

    output_dir = os.path.join(result_dir, 'data')

    if 'test' not in cfg.TEST.TEST_SETS:
        evaluate_py_wrapper(result_dir)
        # evaluate_cuda_wrapper(output_dir, cfg.TEST.DATASET)
    else:
        logger.info('results file save in  {}'.format(result_dir))
        os.system('cd %s && zip -q -r ../results.zip *' % (result_dir))

if __name__=='__main__':

    # Load Frustum Datasets.
    if 'frustum_pointnet' in MODEL_FILE:
        gen_ref = False
    elif 'frustum_convnet' in MODEL_FILE:
        gen_ref = True
    else:
        print("Wrong model parameter.")
        exit(0)

    provider = import_from_file(DATA_FILE)

    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split=TEST_SETS,
                                           rotate_to_center=True, one_hot=True,
                                           overwritten_data_path=TEST_FILE,
                                           gen_ref=gen_ref)

    test_dataloader = DataLoader(TEST_DATASET, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
    # Model
    if 'frustum_pointnets_v1' in MODEL_FILE:
        from frustum_pointnets_v1 import FrustumPointNetv1

        model = FrustumPointNetv1(n_classes=NUM_CLASSES, n_channel=NUM_CHANNEL).cuda()
    elif 'frustum_convnet_v1' in MODEL_FILE:
        from frustum_convnet_v1 import FrustumConvNetv1

        model = FrustumConvNetv1(n_classes=NUM_CLASSES, n_channel=NUM_CHANNEL).cuda()

    # Pre-trained Model
    if not os.path.isfile(TEST_WEIGHTS):
        log_string("no checkpoint found at '{}'".format(TEST_WEIGHTS))
        exit(0)

    pth = torch.load(TEST_WEIGHTS)
    model.load_state_dict(pth['model_state_dict'])
    log_string("loaded checkpoint '{}'".format(TEST_WEIGHTS))

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    save_file_name = os.path.join(SAVE_DIR, 'detection.pkl')
    result_folder = os.path.join(SAVE_DIR, 'result')

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    test(model, TEST_DATASET, test_dataloader, save_file_name, result_folder)


