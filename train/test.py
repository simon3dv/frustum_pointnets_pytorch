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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
parser.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
FLAGS = parser.parse_args()

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)
NUM_CLASSES = 2
NUM_CHANNEL = 4

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
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
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

def test(output_filename, result_dir=None):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''

    # Load Frustum Datasets.
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
                                           rotate_to_center=True, overwritten_data_path=FLAGS.data_path,
                                           from_rgb_detection=FLAGS.from_rgb_detection, one_hot=True)
    loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, \
                    num_workers=8, pin_memory=True)
    if FLAGS.model == 'frustum_pointnets_v1':
        from frustum_pointnets_v1 import FrustumPointNetv1
        FrustumPointNet = FrustumPointNetv1().cuda()
    pth = torch.load(FLAGS.model_path)
    FrustumPointNet.load_state_dict(pth['model_state_dict'])

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

    correct_cnt = 0
    n_samples = 0



    for i, data in tqdm(enumerate(loader), \
                        total=len(loader), smoothing=0.9):
        n_samples += data[0].shape[0]

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
        batch_one_hot_vec = batch_one_hot_vec.float().cuda()

        FrustumPointNet = FrustumPointNet.eval()

        batch_logits, batch_mask, batch_stage1_center, batch_center_boxnet, \
        batch_heading_scores, batch_heading_residuals_normalized, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals_normalized, batch_size_residuals, batch_center = \
            FrustumPointNet(batch_data, batch_one_hot_vec)

        batch_label = batch_label.detach().cpu().numpy()
        batch_logits = batch_logits.detach().cpu().numpy()
        batch_mask = batch_mask.detach().cpu().numpy()
        batch_stage1_center = batch_stage1_center.detach().cpu().numpy()
        batch_center_boxnet = batch_center_boxnet.detach().cpu().numpy()
        batch_heading_scores = batch_heading_scores.detach().cpu().numpy()
        batch_heading_residuals_normalized = batch_heading_residuals_normalized.detach().cpu().numpy()
        batch_heading_residuals = batch_heading_residuals.detach().cpu().numpy()
        batch_size_scores = batch_size_scores.detach().cpu().numpy()
        batch_size_residuals_normalized = batch_size_residuals_normalized.detach().cpu().numpy()
        batch_size_residuals = batch_size_residuals.detach().cpu().numpy()
        batch_center = batch_center.detach().cpu().numpy()




        #batch_output:(32, 1024)
        batch_output = batch_mask
        #batch_center_pred:(32, 3)
        batch_center_pred = batch_center_boxnet
        #heading_cls,heading_res
        batch_hclass_pred = np.argmax(batch_heading_scores, 1)# bs
        batch_hres_pred = np.array([batch_heading_residuals[j,batch_hclass_pred[j]] \
            for j in range(batch_data.shape[0])])
        #batch_size_cls,batch_size_res
        batch_sclass_pred = np.argmax(batch_size_scores, 1)# bs
        batch_sres_pred = np.vstack([batch_size_residuals[j,batch_sclass_pred[j],:] \
            for j in range(batch_data.shape[0])])#(32,3)

        #batch_scores
        batch_seg_prob = softmax(batch_logits)[:,:,1] # BxN
        batch_seg_mask = np.argmax(batch_logits, 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
        heading_prob = np.max(softmax(batch_heading_scores),1) # B
        size_prob = np.max(softmax(batch_size_scores),1) # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)

        correct_cnt += np.sum(batch_output == batch_label)

        for j in range(batch_output.shape[0]):
            ps_list.append(batch_data[j,...])
            seg_list.append(batch_label[j,...])
            segp_list.append(batch_output[j,...])
            center_list.append(batch_center_pred[j,:])
            heading_cls_list.append(batch_hclass_pred[j])
            heading_res_list.append(batch_hres_pred[j])
            size_cls_list.append(batch_sclass_pred[j])
            size_res_list.append(batch_sres_pred[j,:])
            rot_angle_list.append(batch_rot_angle[j])
            score_list.append(batch_scores[j])

    print("Segmentation accuracy: %f" % \
        (correct_cnt / float(batch_size*num_batches*NUM_POINT)))

    if FLAGS.dump_result:
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

    write_detection_results(result_dir, TEST_DATASET.id_list,
        TEST_DATASET.type_list, TEST_DATASET.box2d_list, center_list,
        heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list)

if __name__=='__main__':
    '''
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
        test(FLAGS.output+'.pickle', FLAGS.output)
