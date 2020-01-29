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
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
#ex. log/20200121-decay_rate=0.7-decay_step=20_caronly/20200121-decay_rate=0.7-decay_step=20_caronly-acc0.777317-epoch130.pth
parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
parser.add_argument('--idx_path', default='idx_path kitti/image_sets/val.txt', help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
parser.add_argument('--return_all_loss', default=False, action='store_true',help='only return total loss default')
parser.add_argument('--objtype', type=str, default='caronly', help='caronly or carpedcyc')
parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='only consider CAM_FRONT')
parser.add_argument('--dataset', type=str, default='kitti', help='kitti or nuscenes or nuscenes2kitti')
parser.add_argument('--split', type=str, default='val', help='v1.0-mini or val')
parser.add_argument('--debug', default=False, action='store_true',help='debug mode')
FLAGS = parser.parse_args()

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)
NUM_CLASSES = 2
NUM_CHANNEL = 4
if FLAGS.objtype == 'carpedcyc':
    n_classes = 3
elif FLAGS.objtype == 'caronly':
    n_classes = 1

# Loss
Loss = FrustumPointNetLoss(return_all = FLAGS.return_all_loss)

# Load Frustum Datasets.
if FLAGS.dataset == 'kitti':
    if FLAGS.data_path == None:
        overwritten_data_path = 'kitti/frustum_' + FLAGS.objtype + '_' + FLAGS.split + '.pickle'
    else:
        overwritten_data_path = FLAGS.data_path
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
                                           rotate_to_center=True, one_hot=True,
                                           overwritten_data_path=overwritten_data_path)
elif FLAGS.dataset == 'nuscenes2kitti':
    SENSOR = FLAGS.sensor
    overwritten_data_path_prefix = 'nuscenes2kitti/frustum_' + FLAGS.objtype + '_' + SENSOR + '_'
    if FLAGS.data_path == None:
        overwritten_data_path = overwritten_data_path_prefix + FLAGS.split + '.pickle'
    else:
        overwritten_data_path = FLAGS.data_path
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
                                           rotate_to_center=True, one_hot=True,
                                           overwritten_data_path=overwritten_data_path)
else:
    print('Unknown dataset: %s' % (FLAGS.dataset))
    exit(-1)

test_dataloader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, \
                             num_workers=8, pin_memory=True)
# Model
if FLAGS.model == 'frustum_pointnets_v1':
    from frustum_pointnets_v1 import FrustumPointNetv1
    FrustumPointNet = FrustumPointNetv1(n_classes=n_classes).cuda()

# Pre-trained Model
pth = torch.load(FLAGS.model_path)
FrustumPointNet.load_state_dict(pth['model_state_dict'])

# output file dir and name
output_filename = FLAGS.output + '.pickle'
result_dir = FLAGS.output

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list, \
                            heading_cls_list, heading_res_list, \
                            size_cls_list, size_res_list, \
                            rot_angle_list, score_list):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i], rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close() 

def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

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
    test_total_loss = 0.0
    test_iou2d = 0.0
    test_iou3d = 0.0
    test_acc = 0.0
    test_iou3d_acc = 0.0
    eval_time = 0.0
    if FLAGS.return_all_loss:
        test_mask_loss = 0.0
        test_center_loss = 0.0
        test_heading_class_loss = 0.0
        test_size_class_loss = 0.0
        test_heading_residuals_normalized_loss = 0.0
        test_size_residuals_normalized_loss = 0.0
        test_stage1_center_loss = 0.0
        test_corners_loss = 0.0

    for i, data in tqdm(enumerate(loader), \
                        total=len(loader), smoothing=0.9):
        # for debug
        if FLAGS.debug == True:
            if i == 1:
                break
        test_n_samples += data[0].shape[0]
        '''
        batch_data:[32, 2048, 4], pts in frustum
        batch_label:[32, 2048], pts ins seg label in frustum
        batch_center:[32, 3],
        batch_hclass:[32],
        batch_hres:[32],
        batch_sclass:[32],
        batch_sres:[32,3],
        batch_rot_angle:[32],
        batch_one_hot_vec:[32,3],
        '''
        # 1. Load data
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = data

        batch_data = batch_data.transpose(2, 1).float().cuda()
        batch_label = batch_label.float().cuda()
        batch_center = batch_center.float().cuda()
        batch_hclass = batch_hclass.float().cuda()
        batch_hres = batch_hres.float().cuda()
        batch_sclass = batch_sclass.float().cuda()
        batch_sres = batch_sres.float().cuda()
        # batch_rot_angle = batch_rot_angle.float().cuda()
        batch_one_hot_vec = batch_one_hot_vec.float().cuda()

        # 2. Eval one batch
        model = model.eval()
        eval_t1 = time.perf_counter()
        logits, mask, stage1_center, center_boxnet, \
        heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals, center = \
            model(batch_data, batch_one_hot_vec)
        #logits:[32, 1024, 2] , mask:[32, 1024]
        eval_t2 = time.perf_counter()
        eval_time += (eval_t2 - eval_t1)

        # 3. Compute Loss
        if FLAGS.return_all_loss:
            total_loss, mask_loss, center_loss, heading_class_loss, \
                size_class_loss, heading_residuals_normalized_loss, \
                size_residuals_normalized_loss, stage1_center_loss, \
                corners_loss = \
                Loss(logits, batch_label, \
                     center, batch_center, stage1_center, \
                     heading_scores, heading_residuals_normalized, \
                     heading_residuals, \
                     batch_hclass, batch_hres, \
                     size_scores, size_residuals_normalized, \
                     size_residuals, \
                     batch_sclass, batch_sres)
        else:
            total_loss = \
                Loss(logits, batch_label, \
                     center, batch_center, stage1_center, \
                     heading_scores, heading_residuals_normalized, \
                     heading_residuals, \
                     batch_hclass, batch_hres, \
                     size_scores, size_residuals_normalized, \
                     size_residuals, \
                     batch_sclass, batch_sres)

        test_total_loss += total_loss.item()
        if FLAGS.return_all_loss:
            test_mask_loss += mask_loss.item()
            test_center_loss += center_loss.item()
            test_heading_class_loss += heading_class_loss.item()
            test_size_class_loss += size_class_loss.item()
            test_heading_residuals_normalized_loss += heading_residuals_normalized_loss.item()
            test_size_residuals_normalized_loss += size_residuals_normalized_loss.item()
            test_stage1_center_loss += stage1_center_loss.item()
            test_corners_loss += corners_loss.item()

        # 4. compute seg acc, IoU and acc(IoU)
        correct = torch.argmax(logits, 2).eq(batch_label.detach().long()).cpu().numpy()
        accuracy = np.sum(correct) / float(NUM_POINT)
        test_acc += accuracy

        logits = logits.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        center_boxnet = center_boxnet.cpu().detach().numpy()
        #stage1_center = stage1_center.cpu().detach().numpy()#
        center = center.cpu().detach().numpy()
        heading_scores = heading_scores.cpu().detach().numpy()
        #heading_residuals_normalized = heading_residuals_normalized.cpu().detach().numpy()
        heading_residuals = heading_residuals.cpu().detach().numpy()
        size_scores = size_scores.cpu().detach().numpy()
        size_residuals = size_residuals.cpu().detach().numpy()
        #size_residuals_normalized = size_residuals_normalized.cpu().detach().numpy()#
        batch_center = batch_center.cpu().detach().numpy()
        batch_hclass = batch_hclass.cpu().detach().numpy()
        batch_hres = batch_hres.cpu().detach().numpy()
        batch_sclass = batch_sclass.cpu().detach().numpy()
        batch_sres = batch_sres.cpu().detach().numpy()

        iou2ds, iou3ds = provider.compute_box3d_iou(
            center,
            heading_scores,
            heading_residuals,
            size_scores,
            size_residuals,
            batch_center,
            batch_hclass,
            batch_hres,
            batch_sclass,
            batch_sres)
        test_iou2d += np.sum(iou2ds)
        test_iou3d += np.sum(iou3ds)
        test_iou3d_acc += np.sum(iou3ds >= 0.7)

        # 5. Compute and write all Results
        batch_output = mask#torch.Size([32, 1024])
        batch_center_pred = center_boxnet#torch.Size([32, 3])
        batch_hclass_pred = np.argmax(heading_scores, 1)  # (32,)
        batch_hres_pred = np.array([heading_residuals[j, batch_hclass_pred[j]] \
                                    for j in range(batch_data.shape[0])]) # (32,)
        # batch_size_cls,batch_size_res
        batch_sclass_pred = np.argmax(size_scores, 1)  # (32,)
        batch_sres_pred = np.vstack([size_residuals[j, batch_sclass_pred[j], :] \
                                     for j in range(batch_data.shape[0])]) # (32,3)

        # batch_scores
        batch_seg_prob = softmax(logits)[:, :, 1]  # (32, 1024, 2) ->(32, 1024)
        batch_seg_mask = np.argmax(logits, 2)  # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1)  # B,
        heading_prob = np.max(softmax(heading_scores), 1)  # B
        size_prob = np.max(softmax(size_scores), 1)  # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)

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


if __name__=='__main__':
    '''
    example:
    python train/test.py 
    --model_path log/20190113_decay_rate0.7/20190113_decay_rate0.7-acc0000-epoch156.pth
    --output train/detection_results_v1 
    --data_path kitti/frustum_carpedcyc_val.pickle 
    --idx_path kitti/image_sets/val.txt 
    train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_results_v1
    '''

    if FLAGS.from_rgb_detection:
        test_from_rgb_detection(FLAGS.output+'.pickle', FLAGS.output)
    else:
        # test one epoch from 2d gt
        if FLAGS.return_all_loss:
            test_total_loss, test_iou2d, test_iou3d, test_acc, test_iou3d_acc, \
            test_mask_loss, \
            test_center_loss, \
            test_heading_class_loss, \
            test_size_class_loss, \
            test_heading_residuals_normalized_loss, \
            test_size_residuals_normalized_loss, \
            test_stage1_center_loss, \
            test_corners_loss \
                = \
                test_one_epoch(FrustumPointNet, test_dataloader)
        else:
            test_total_loss, test_iou2d, test_iou3d, test_acc, test_iou3d_acc, \
                = \
                test_one_epoch(FrustumPointNet, test_dataloader)
    epoch = 0
    blue = lambda x: '\033[94m' + x + '\033[0m'
    print('[%d] %s loss: %.6f' % \
          (epoch + 1, blue('test'), test_total_loss))
    print('%s segmentation accuracy: %.6f' % (blue('test'), test_acc))
    print('%s box IoU(ground/3D): %.6f/%.6f' % (blue('test'), test_iou2d, test_iou3d))
    print('%s box estimation accuracy (IoU=0.7): %.6f' % (blue('test'), test_iou3d_acc))


    '''
    log:
    CUDA_VISIBLE_DEVICES=0 python train/test.py 
    --model_path log/20200121-decay_rate=0.7-decay_step=20_caronly/20200121-decay_rate=0.7-decay_step=20_caronly-acc0.777317-epoch130.pth 
    --data_path kitti/frustum_caronly_val.pickle 
    --idx_path kitti/image_sets/val.txt 
    --output train/kitti_caronly_v1

    train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/kitti_caronly_v1

    '''