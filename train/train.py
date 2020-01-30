import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model_util import FrustumPointNetLoss
import argparse
import importlib
import time
import ipdb
import numpy as np
import random
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import provider

parser = argparse.ArgumentParser()
###parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=150, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=20, help='Decay step for lr decay [default: 60]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--ckpt',type=str,default=None,help='Pre-trained model file')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight Decay of Adam [default: 1e-4]')
parser.add_argument('--name', type=str, default='1-', help='tensorboard writer name')
parser.add_argument('--return_all_loss', default=False, action='store_true',help='only return total loss default')
parser.add_argument('--debug', default=False, action='store_true',help='debug mode')
parser.add_argument('--objtype', type=str, default='caronly', help='caronly or carpedcyc')
parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='only consider CAM_FRONT')
parser.add_argument('--dataset', type=str, default='kitti', help='kitti or nuscenes or nuscenes2kitti')
parser.add_argument('--train_sets', type=str, default='train')
parser.add_argument('--val_sets', type=str, default='val')
FLAGS = parser.parse_args()

# Set training configurations

strtime = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
if 'nuscenes' in FLAGS.dataset:
    NAME = strtime + '_' + FLAGS.name  + '_' + FLAGS.objtype + '_' + FLAGS.dataset + '_' + FLAGS.sensor + '_'
else:
    NAME = strtime + '_' + FLAGS.name  + '_' + FLAGS.objtype + '_' + FLAGS.dataset + '_' + strtime
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
# GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4 # point feature channel
NUM_CLASSES = 2 # segmentation has two classes
if FLAGS.objtype == 'carpedcyc':
    n_classes = 3
elif FLAGS.objtype == 'caronly':
    n_classes = 1
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(LOG_DIR + '/' + NAME): os.mkdir(LOG_DIR + '/' + NAME)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
# BN_INIT_DECAY = 0.5
# BN_DECAY_DECAY_RATE = 0.5
# BN_DECAY_DECAY_STEP = float(DECAY_STEP)
# BN_DECAY_CLIP = 0.99

# Load Frustum Datasets. Use default data paths.
if FLAGS.dataset == 'kitti':
    TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split=FLAGS.train_sets,
        rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True,
        overwritten_data_path='kitti/frustum_'+FLAGS.objtype+'_'+FLAGS.train_sets+'.pickle')
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split=FLAGS.val_sets,
        rotate_to_center=True, one_hot=True,
        overwritten_data_path='kitti/frustum_'+FLAGS.objtype+'_'+FLAGS.val_sets+'.pickle')
elif FLAGS.dataset == 'nuscenes2kitti':
    SENSOR = FLAGS.sensor
    overwritten_data_path_prefix = 'nuscenes2kitti/frustum_' +FLAGS.objtype + '_' + SENSOR + '_'
    TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split=FLAGS.train_sets,
        rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True,
        overwritten_data_path=overwritten_data_path_prefix + '_'+FLAGS.train_sets+'.pickle')
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split=FLAGS.val_sets,
        rotate_to_center=True, one_hot=True,
        overwritten_data_path=overwritten_data_path_prefix + '_'+FLAGS.val_sets+'.pickle')
else:
    print('Unknown dataset: %s' % (FLAGS.dataset))
    exit(-1)
train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,\
                                num_workers=8,pin_memory=True)
test_dataloader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,\
                                num_workers=8,pin_memory=True)
Loss = FrustumPointNetLoss(return_all = FLAGS.return_all_loss)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
def test_one_epoch(model, loader):
    test_n_samples = 0
    test_total_loss = 0.0
    test_iou2d = 0.0
    test_iou3d = 0.0
    test_acc = 0.0
    test_iou3d_acc = 0.0

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
        batch_rot_angle = batch_rot_angle.float().cuda()
        batch_one_hot_vec = batch_one_hot_vec.float().cuda()

        model = model.eval()

        logits, mask, stage1_center, center_boxnet, \
        heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals, center = \
            model(batch_data, batch_one_hot_vec)

        logits = logits.detach()
        stage1_center = stage1_center.detach()
        center_boxnet = center_boxnet.detach()
        heading_scores = heading_scores.detach()
        heading_residuals_normalized = heading_residuals_normalized.detach()
        heading_residuals = heading_residuals.detach()
        size_scores = size_scores.detach()
        size_residuals_normalized = size_residuals_normalized.detach()
        size_residuals = size_residuals.detach()
        center = center.detach()

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

        iou2ds, iou3ds = provider.compute_box3d_iou( \
            center.cpu().detach().numpy(), \
            heading_scores.cpu().detach().numpy(), \
            heading_residuals.cpu().detach().numpy(), \
            size_scores.cpu().detach().numpy(), \
            size_residuals.cpu().detach().numpy(), \
            batch_center.cpu().detach().numpy(), \
            batch_hclass.cpu().detach().numpy(), \
            batch_hres.cpu().detach().numpy(), \
            batch_sclass.cpu().detach().numpy(), \
            batch_sres.cpu().detach().numpy())
        test_iou2d += np.sum(iou2ds)
        test_iou3d += np.sum(iou3ds)

        correct = torch.argmax(logits, 2).eq(batch_label.detach().long()).cpu().numpy()
        accuracy = np.sum(correct) / float(NUM_POINT)
        test_acc += accuracy

        test_iou3d_acc += np.sum(iou3ds >= 0.7)

        if FLAGS.return_all_loss:
            test_mask_loss += mask_loss.item()
            test_center_loss += center_loss.item()
            test_heading_class_loss += heading_class_loss.item()
            test_size_class_loss += size_class_loss.item()
            test_heading_residuals_normalized_loss += heading_residuals_normalized_loss.item()
            test_size_residuals_normalized_loss += size_residuals_normalized_loss.item()
            test_stage1_center_loss += stage1_center_loss.item()
            test_corners_loss += corners_loss.item()

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

def train():

    ''' Main function for training and simple evaluation. '''
    start= time.perf_counter()
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    blue = lambda x: '\033[94m' + x + '\033[0m'

    # set model
    if FLAGS.model == 'frustum_pointnets_v1':
        from frustum_pointnets_v1 import FrustumPointNetv1
        FrustumPointNet = FrustumPointNetv1(n_classes=n_classes).cuda()

    # load pre-trained model
    if FLAGS.ckpt:
        ckpt = torch.load(FLAGS.ckpt)
        FrustumPointNet.load_state_dict(ckpt['model_state_dict'])

    # set optimizer and scheduler
    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            FrustumPointNet.parameters(), lr=BASE_LEARNING_RATE,
            betas=(0.9, 0.999),eps=1e-08,
            weight_decay=FLAGS.weight_decay)
    def lr_func(epoch, init=BASE_LEARNING_RATE, step_size=DECAY_STEP, gamma=DECAY_RATE, eta_min=0.00001):
        f = gamma**(epoch//DECAY_STEP)
        if init*f>eta_min:
            return f
        else:
            return 0.01#0.001*0.01 = eta_min
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    # train
    if os.path.exists('runs/' + NAME):
        print('name has been existed')
        shutil.rmtree('runs/' + NAME)

    writer = SummaryWriter('runs/' + NAME)
    num_batch = len(TRAIN_DATASET) / BATCH_SIZE
    best_iou3d_acc = 0.0
    best_epoch = 1
    best_file = ''

    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch + 1))
        sys.stdout.flush()
        print('Epoch %d/%s:' % (epoch + 1, MAX_EPOCH))

        # record for one epoch
        train_total_loss = 0.0
        train_iou2d = 0.0
        train_iou3d = 0.0
        train_acc = 0.0
        train_iou3d_acc = 0.0

        if FLAGS.return_all_loss:
            train_mask_loss = 0.0
            train_center_loss = 0.0
            train_heading_class_loss = 0.0
            train_size_class_loss = 0.0
            train_heading_residuals_normalized_loss = 0.0
            train_size_residuals_normalized_loss = 0.0
            train_stage1_center_loss = 0.0
            train_corners_loss = 0.0

        n_samples = 0
        for i, data in tqdm(enumerate(train_dataloader),\
                total=len(train_dataloader), smoothing=0.9):
            n_samples += data[0].shape[0]
            #for debug
            if FLAGS.debug==True:
                if i==1 :
                    break

            '''
            data after frustum rotation
            1. For Seg
            batch_data:[32, 2048, 4], pts in frustum, 
            batch_label:[32, 2048], pts ins seg label in frustum,
            2. For T-Net
            batch_center:[32, 3],
            3. For Box Est.
            batch_hclass:[32],
            batch_hres:[32],
            batch_sclass:[32],
            batch_sres:[32,3],
            4. Others
            batch_rot_angle:[32],alpha, not rotation_y,
            batch_one_hot_vec:[32,3],
            '''
            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec = data

            batch_data = batch_data.transpose(2,1).float().cuda()
            batch_label = batch_label.float().cuda()
            batch_center = batch_center.float().cuda()
            batch_hclass = batch_hclass.long().cuda()
            batch_hres = batch_hres.float().cuda()
            batch_sclass = batch_sclass.long().cuda()
            batch_sres = batch_sres.float().cuda()
            batch_rot_angle = batch_rot_angle.float().cuda()###Not Use?
            batch_one_hot_vec = batch_one_hot_vec.float().cuda()

            optimizer.zero_grad()
            FrustumPointNet = FrustumPointNet.train()

            '''
            #bn_decay(defaut 0.1)
            bn_momentum = BN_INIT_DECAY * BN_DECAY_DECAY_RATE**(epoch//BN_DECAY_DECAY_STEP)
            if bn_momentum < 1 - BN_DECAY_CLIP:
                bn_momentum = 1 - BN_DECAY_CLIP
            '''
            logits, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residuals_normalized, heading_residuals, \
            size_scores, size_residuals_normalized, size_residuals, center = \
                FrustumPointNet(batch_data, batch_one_hot_vec)


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
                        size_scores,size_residuals_normalized,\
                        size_residuals,\
                        batch_sclass,batch_sres)
            else:
                total_loss = \
                    Loss(logits, batch_label, \
                        center, batch_center, stage1_center, \
                        heading_scores, heading_residuals_normalized, \
                        heading_residuals, \
                        batch_hclass, batch_hres, \
                        size_scores,size_residuals_normalized,\
                        size_residuals,\
                        batch_sclass,batch_sres)

            total_loss.backward()
            optimizer.step()
            train_total_loss += total_loss.item()

            iou2ds, iou3ds = provider.compute_box3d_iou(\
                center.cpu().detach().numpy(),\
                heading_scores.cpu().detach().numpy(),\
                heading_residuals.cpu().detach().numpy(), \
                size_scores.cpu().detach().numpy(), \
                size_residuals.cpu().detach().numpy(), \
                batch_center.cpu().detach().numpy(), \
                batch_hclass.cpu().detach().numpy(), \
                batch_hres.cpu().detach().numpy(), \
                batch_sclass.cpu().detach().numpy(), \
                batch_sres.cpu().detach().numpy())
            train_iou2d += np.sum(iou2ds)
            train_iou3d += np.sum(iou3ds)
            train_iou3d_acc += np.sum(iou3ds>=0.7)

            correct = torch.argmax(logits, 2).eq(batch_label.long()).detach().cpu().numpy()
            accuracy = np.sum(correct)
            train_acc += accuracy
            if FLAGS.return_all_loss:
                train_mask_loss += mask_loss.item()
                train_center_loss += center_loss.item()
                train_heading_class_loss += heading_class_loss.item()
                train_size_class_loss += size_class_loss.item()
                train_heading_residuals_normalized_loss += heading_residuals_normalized_loss.item()
                train_size_residuals_normalized_loss += size_residuals_normalized_loss.item()
                train_stage1_center_loss += stage1_center_loss.item()
                train_corners_loss += corners_loss.item()

            '''
            print('[%d: %d/%d] train loss: %.6f' % \
                  (epoch + 1, i, len(train_dataloader),(train_total_loss/n_samples)))
            print('box IoU(ground/3D): %.6f/%.6f' % (train_iou2d/n_samples, train_iou3d/n_samples))
            print('box estimation accuracy (IoU=0.7): %.6f' % (train_iou3d_acc/n_samples))
            if FLAGS.return_all_loss:
                print('train_mask_loss:%.6f'%(train_mask_loss/n_samples))
                print('train_stage1_center_loss:%.6f' % (train_stage1_center_loss/n_samples))
                print('train_heading_class_loss:%.6f' % (train_heading_class_loss/n_samples))
                print('train_size_class_loss:%.6f' % (train_size_class_loss/n_samples))
                print('train_heading_residuals_normalized_loss:%.6f' % (train_heading_residuals_normalized_loss/n_samples))
                print('train_size_residuals_normalized_loss:%.6f' % (train_size_residuals_normalized_loss/n_samples))
                print('train_stage1_center_loss:%.6f' % (train_stage1_center_loss/n_samples))
                print('train_corners_loss:%.6f'%(train_corners_loss/n_samples))
            '''
        train_total_loss /= n_samples
        train_acc /= n_samples*float(NUM_POINT)
        train_iou2d /= n_samples
        train_iou3d /= n_samples
        train_iou3d_acc /= n_samples

        if FLAGS.return_all_loss:
            train_mask_loss /= n_samples
            train_center_loss /= n_samples
            train_heading_class_loss /= n_samples
            train_size_class_loss /= n_samples
            train_heading_residuals_normalized_loss /= n_samples
            train_size_residuals_normalized_loss /= n_samples
            train_stage1_center_loss /= n_samples
            train_corners_loss /= n_samples

        print('[%d: %d/%d] train loss: %.6f' % \
              (epoch + 1, i, len(train_dataloader),train_total_loss))
        print('segmentation accuracy: %.6f'% train_acc )
        print('box IoU(ground/3D): %.6f/%.6f'% (train_iou2d, train_iou3d))
        print('box estimation accuracy (IoU=0.7): %.6f'%(train_iou3d_acc))

        # test one epoch
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
                    test_one_epoch(FrustumPointNet,test_dataloader)
        else:
            test_total_loss, test_iou2d, test_iou3d, test_acc, test_iou3d_acc,\
                = \
                test_one_epoch(FrustumPointNet,test_dataloader)

        print('[%d] %s loss: %.6f' % \
              (epoch + 1, blue('test'), test_total_loss))
        print('%s segmentation accuracy: %.6f'% (blue('test'),test_acc))
        print('%s box IoU(ground/3D): %.6f/%.6f'% (blue('test'),test_iou2d, test_iou3d))
        print('%s box estimation accuracy (IoU=0.7): %.6f'%(blue('test'), test_iou3d_acc))
        print("learning rate: {:.6f}".format(optimizer.param_groups[0]['lr']))
        scheduler.step()

        if not FLAGS.debug:
            writer.add_scalar('train_total_loss',train_total_loss, epoch)
            writer.add_scalar('train_iou2d',train_iou2d, epoch)
            writer.add_scalar('train_iou3d',train_iou3d, epoch)
            writer.add_scalar('train_acc',train_acc, epoch)
            writer.add_scalar('train_iou3d_acc',train_iou3d_acc, epoch)

        if FLAGS.return_all_loss and not FLAGS.debug:
            writer.add_scalar('train_mask_loss',train_mask_loss)
            writer.add_scalar('train_center_loss',train_center_loss, epoch)
            writer.add_scalar('train_heading_class_loss',train_heading_class_loss, epoch)
            writer.add_scalar('train_size_class_loss',train_size_class_loss, epoch)
            writer.add_scalar('train_heading_residuals_normalized_loss',train_heading_residuals_normalized_loss, epoch)
            writer.add_scalar('train_size_residuals_normalized_loss',train_size_residuals_normalized_loss, epoch)
            writer.add_scalar('train_stage1_center_loss',train_stage1_center_loss, epoch)
            writer.add_scalar('train_corners_loss',train_corners_loss, epoch)

        if not FLAGS.debug:
            writer.add_scalar('test_total_loss',test_total_loss, epoch)
            writer.add_scalar('test_iou2d_loss',test_iou2d, epoch)
            writer.add_scalar('test_iou3d_loss',test_iou3d, epoch)
            writer.add_scalar('test_acc',test_acc, epoch)
            writer.add_scalar('test_iou3d_acc',test_iou3d_acc, epoch)

        if FLAGS.return_all_loss:
            writer.add_scalar('test_mask_loss',test_mask_loss, epoch)
            writer.add_scalar('test_center_loss',test_center_loss, epoch)
            writer.add_scalar('test_heading_class_loss',test_heading_class_loss, epoch)
            writer.add_scalar('test_size_class_loss',test_size_class_loss, epoch)
            writer.add_scalar('test_heading_residuals_normalized_loss',test_heading_residuals_normalized_loss, epoch)
            writer.add_scalar('test_size_residuals_normalized_loss',test_size_residuals_normalized_loss, epoch)
            writer.add_scalar('test_stage1_center_loss',test_stage1_center_loss, epoch)
            writer.add_scalar('test_corners_loss',test_corners_loss, epoch)

        if test_iou3d_acc >= best_iou3d_acc:
            best_iou3d_acc = test_iou3d_acc
            best_epoch = epoch + 1
            if epoch > MAX_EPOCH / 5:
                savepath = LOG_DIR + '/' + NAME + '/%s-acc%04f-epoch%03d.pth' % \
                           (NAME, test_iou3d_acc, epoch)
                print('save to:',savepath)
                if os.path.exists(best_file):
                    os.remove(best_file)# update to newest best epoch
                best_file = savepath
                state = {
                    'epoch': epoch + 1,
                    'train_iou3d_acc': train_iou3d_acc,
                    'test_iou3d_acc': test_iou3d_acc,
                    'model_state_dict': FrustumPointNet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state,savepath)
                print('Saving model to %s'%savepath)
        print('Best Test acc: %f(Epoch %d)' % (best_iou3d_acc, best_epoch))

        # Save the variables to disk.
        #if epoch % 10 == 0:
        #    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
        #    log_string("Model saved in file: %s" % save_path)
    print("Time {} hours".format(
        float(time.perf_counter()-start)/3600))
    writer.close()

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    print('Your FLAGS:')
    print(FLAGS)
    train()
    LOG_FOUT.close()

    '''
    Official Result:
     -- 010 / 2294 --
    mean loss: 74.339147
    segmentation accuracy: 0.566290
    box IoU (ground/3D): 0.040029 / 0.021777
    box estimation accuracy (IoU=0.7): 0.000000
     -- 020 / 2294 --
    mean loss: 47.905965
    segmentation accuracy: 0.567178
    box IoU (ground/3D): 0.124319 / 0.087477
    box estimation accuracy (IoU=0.7): 0.000000
    
    After One Epoch:
     -- 2290 / 2294 --
    mean loss: 10.270118
    segmentation accuracy: 0.843689
    box IoU (ground/3D): 0.546752 / 0.480549
    box estimation accuracy (IoU=0.7): 0.175000
    2019-12-06 22:03:58.427488
    ---- EPOCH 000 EVALUATION ----
    eval mean loss: 9.794078
    eval segmentation accuracy: 0.865757
    eval segmentation avg class acc: 0.867919
    eval box IoU (ground/3D): 0.573301 / 0.524547
    eval box estimation accuracy (IoU=0.7): 0.270213
    Model saved in file: train/log_v1/model.ckpt
    
    ---- EPOCH 200 EVALUATION ----
    eval mean loss: 4.023258
    eval segmentation accuracy: 0.905076
    eval segmentation avg class acc: 0.907372
    eval box IoU (ground/3D): 0.746727 / 0.698527
    eval box estimation accuracy (IoU=0.7): 0.631738
    Model saved in file: train/log_v1/model.ckpt

    
    My Result:
    **** EPOCH 001 ****
    Epoch 1/201:
    100%|██████████████████████████████████████████████| 2296/2296 [06:58<00:00,  5.49it/s]
    [1: 2295/2296] train loss: 49.650354
    segmentation accuracy: 0.819838
    box IoU(ground/3D): 0.365691/0.315628
    box estimation accuracy (IoU=0.7)
    100%|████████████████████████████████████████████████| 488/488 [00:32<00:00, 14.99it/s]
    [1: 2295/2296] test loss: 49.650354
    test segmentation accuracy: 0.819838
    test box IoU(ground/3D): 0.365691/0.315628
    test box estimation accuracy (IoU=0.7): 0.026070
    Best Test acc: 0.852231(Epoch 1)
    **** EPOCH 002 ****
    Epoch 2/201:
    100%|██████████████████████████████████████████████| 2296/2296 [07:00<00:00,  5.46it/s]
    [2: 2295/2296] train loss: 6.207534
    segmentation accuracy: 0.860321
    box IoU(ground/3D): 0.424854/0.372507
    box estimation accuracy (IoU=0.7)
    100%|████████████████████████████████████████████████| 488/488 [00:35<00:00, 13.78it/s]
    [2: 2295/2296] test loss: 6.207534
    test segmentation accuracy: 0.860321
    test box IoU(ground/3D): 0.424854/0.372507
    test box estimation accuracy (IoU=0.7): 0.049431
    Best Test acc: 0.873361(Epoch 2)
    

    **** EPOCH 001 ****
    Epoch 1/201:
    100%|██████████████████████████████████████████████████████| 2296/2296 [05:22<00:00,  7.13it/s]
    [1: 2295/2296] train loss: 0.414606
    segmentation accuracy: 0.815620
    box IoU(ground/3D): 0.471504/0.422675
    box estimation accuracy (IoU=0.7): 0.114817
    100%|████████████████████████████████████████████████████████| 488/488 [00:28<00:00, 17.11it/s]
    [1: 2295/488] test loss: 0.419938
    test segmentation accuracy: 0.816805
    test box IoU(ground/3D): 0.527124/0.469022
    test box estimation accuracy (IoU=0.7): 0.130477
    Best Test acc: 0.130477(Epoch 1)
    **** EPOCH 002 ****
    Epoch 2/201:
    100%|██████████████████████████████████████████████████████| 2296/2296 [05:20<00:00,  7.17it/s]
    [2: 2295/2296] train loss: 0.252226
    segmentation accuracy: 0.859254
    box IoU(ground/3D): 0.563677/0.513837
    box estimation accuracy (IoU=0.7): 0.208206
    100%|████████████████████████████████████████████████████████| 488/488 [00:30<00:00, 16.13it/s]
    [2: 2295/488] test loss: 0.181613
    test segmentation accuracy: 0.860691
    test box IoU(ground/3D): 0.609513/0.558039
    test box estimation accuracy (IoU=0.7): 0.254474
    Best Test acc: 0.254474(Epoch 2)
    ...
    [12: 2295/2296] train loss: 0.151711
    box IoU(ground/3D): 0.661419/0.610747
    box estimation accuracy (IoU=0.7): 0.405086
    train_mask_loss:0.007788
    train_stage1_center_loss:0.002503
    train_heading_class_loss:0.019569
    train_size_class_loss:0.000064
    train_heading_residuals_normalized_loss:0.001008
    train_size_residuals_normalized_loss:0.009315
    train_stage1_center_loss:0.002503
    train_corners_loss:0.096189
    100%|██████████████████████████████████████████████████████| 2296/2296 [07:22<00:00,  5.19it/s]
    [12: 2295/2296] train loss: 0.151711
    segmentation accuracy: 0.895629
    box IoU(ground/3D): 0.661419/0.610747
    box estimation accuracy (IoU=0.7): 0.405086
    100%|████████████████████████████████████████████████████████| 488/488 [00:32<00:00, 15.12it/s]
    [12: 2295/488] test loss: 0.181593
    test segmentation accuracy: 0.892657
    test box IoU(ground/3D): 0.668825/0.623052
    test box estimation accuracy (IoU=0.7): 0.436333
    Best Test acc: 0.436333(Epoch 12)
    **** EPOCH 013 ****


    19.12.15
    [201: 2295/2296] train loss: 0.121046
    box IoU(ground/3D): 0.701009/0.650920
    box estimation accuracy (IoU=0.7): 0.506494
    train_mask_loss:0.006702
    train_stage1_center_loss:0.002294
    train_heading_class_loss:0.013876
    train_size_class_loss:0.000045
    train_heading_residuals_normalized_loss:0.001029
    train_size_residuals_normalized_loss:0.007258
    train_stage1_center_loss:0.002294
    train_corners_loss:0.077865
    100%|██████████████████████████████████████████████████████| 2296/2296 [07:23<00:00,  5.18it/s]
    [201: 2295/2296] train loss: 0.121046
    segmentation accuracy: 0.911900
    box IoU(ground/3D): 0.701009/0.650920
    box estimation accuracy (IoU=0.7): 0.506494
    100%|████████████████████████████████████████████████████████| 488/488 [00:31<00:00, 15.31it/s]
    [201: 2295/488] test loss: 0.142039
    test segmentation accuracy: 0.894511
    test box IoU(ground/3D): 0.678571/0.633051
    test box estimation accuracy (IoU=0.7): 0.462313
    Best Test acc: 0.502021(Epoch 80)
    Time 26.7969213378053 hours
    
    
    CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python train/train.py --return_all_loss --name logitsnotdetach
    
    our FLAGS:
    Namespace(batch_size=32, ckpt=None, decay_rate=0.7, decay_step=20, learning_rate=0.001, log_dir='log', max_epoch=201, model='frustum_pointnets_v1', momentum=0.9, name='logitswodetach_lr1e-3_ds20_dr0.7', no_intensity=False, num_point=1024, optimizer='adam', restore_model_path=None, return_all_loss=True, weight_decay=0.0001)
    **** EPOCH 001 ****
    Epoch 1/201:
    100%|██████████████████████████████████████████████████████| 2296/2296 [07:11<00:00,  5.32it/s]
    [1: 2295/2296] train loss: 0.410411
    segmentation accuracy: 0.816149
    box IoU(ground/3D): 0.473171/0.424222
    box estimation accuracy (IoU=0.7): 0.117553
    100%|████████████████████████████████████████████████████████| 488/488 [00:30<00:00, 15.99it/s]
    [1: 2295/488] test loss: 0.265497
    test segmentation accuracy: 0.852275
    test box IoU(ground/3D): 0.554998/0.503402
    test box estimation accuracy (IoU=0.7): 0.144782
    Best Test acc: 0.144782(Epoch 1)

    CUDA_VISIBLE_DEVICES=0 python train/train.py --name 20200112_lambdal
r_step11_clamp_debug --debug

    '''