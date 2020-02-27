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
from configs.config import cfg
from configs.config import merge_cfg_from_file
from configs.config import merge_cfg_from_list
from configs.config import assert_and_infer_cfg
from utils import import_from_file

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='cfgs/fpointnet/fpointnet_v1_kitti.yaml', help='Config file for training (and optionally testing)')
parser.add_argument('opts',help='See configs/config.py for all options',default=None,nargs=argparse.REMAINDER)
parser.add_argument('--debug', default=False, action='store_true',help='debug mode')
'''
###parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=150, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--min_lr', type=float, default=1e-5, help='min learning rate [default: 1e-5]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=20, help='Decay step for lr decay [default: 60]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--ckpt',type=str,default=None,help='Pre-trained model file')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight Decay of Adam [default: 1e-4]')
parser.add_argument('--name', type=str, default='default', help='tensorboard writer name')
parser.add_argument('--return_all_loss', default=False, action='store_true',help='only return total loss default')
parser.add_argument('--debug', default=False, action='store_true',help='debug mode')
parser.add_argument('--objtype', type=str, default='caronly', help='caronly or carpedcyc')
parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='only consider CAM_FRONT')
parser.add_argument('--dataset', type=str, default='kitti', help='kitti or nuscenes or nuscenes2kitti')
parser.add_argument('--train_sets', type=str, default='train')
parser.add_argument('--val_sets', type=str, default='val')
'''
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

strtime = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))
strtime = strtime[4:8]
NAME = '_'.join(OUTPUT_DIR.split('/')) + '_' + strtime
print(NAME)
MODEL = import_from_file(MODEL_FILE) # import network module
LOG_DIR = OUTPUT_DIR + '/' + NAME
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (CONFIG_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')
# BN_INIT_DECAY = 0.5
# BN_DECAY_DECAY_RATE = 0.5
# BN_DECAY_DECAY_STEP = float(DECAY_STEP)
# BN_DECAY_CLIP = 0.99


# Load Frustum Datasets.
if 'frustum_pointnet' in MODEL_FILE:
    gen_ref = False
elif 'frustum_convnet' in MODEL_FILE:
    gen_ref = True
else:
    print("Wrong model parameter.")
    exit(0)

if 'fusion' in MODEL_FILE:
    with_image = True
else:
    with_image = False

provider = import_from_file(DATA_FILE)

TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split=TRAIN_SETS,
        rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True,
        overwritten_data_path=TRAIN_FILE,
        gen_ref = gen_ref, with_image = with_image)
TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split=TEST_SETS,
        rotate_to_center=True, one_hot=True,
        overwritten_data_path=TEST_FILE,
        gen_ref = gen_ref, with_image = with_image)
train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS,pin_memory=True)
test_dataloader = DataLoader(TEST_DATASET, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS,pin_memory=True)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def test_one_epoch(model, loader):
    time1 = time.perf_counter()

    test_losses = {
        'total_loss': 0.0,
        'cls_loss': 0.0,  # fconvnet
        'mask_loss': 0.0,  # fpointnet
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }
    test_metrics = {
        'seg_acc': 0.0,  # fpointnet
        'cls_acc': 0.0,  # fconvnet
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.7': 0.0,
    }

    n_batches = 0
    for i, data_dicts in tqdm(enumerate(loader), \
                              total=len(loader), smoothing=0.9):
        n_batches += 1
        # for debug
        if args.debug == True:
            if i == 1:
                break

        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}

        model = model.eval()

        with torch.no_grad():
            losses, metrics = model(data_dicts_var)

        for key in test_losses.keys():
            if key in losses.keys():
                test_losses[key] += losses[key].detach().item()
        for key in test_metrics.keys():
            if key in metrics.keys():
                test_metrics[key] += metrics[key]

    for key in test_losses.keys():
        test_losses[key] /= n_batches
    for key in test_metrics.keys():
        test_metrics[key] /= n_batches

    time2 = time.perf_counter()
    print('test time:%.2f s/batch'%((time2-time1)/n_batches))
    return test_losses, test_metrics

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
    if 'frustum_pointnets_v1' in MODEL_FILE:
        from frustum_pointnets_v1 import FrustumPointNetv1
        model = FrustumPointNetv1(n_classes=NUM_CLASSES,n_channel=NUM_CHANNEL).cuda()
    elif 'frustum_convnet_v1' in MODEL_FILE:
        from frustum_convnet_v1 import FrustumConvNetv1
        model = FrustumConvNetv1(n_classes=NUM_CLASSES,n_channel=NUM_CHANNEL).cuda()
    elif 'frustum_convnet_densefusion_v1' in MODEL_FILE:
        from frustum_convnet_densefusion_v1 import FrustumConvNetv1
        model = FrustumConvNetv1(n_classes=NUM_CLASSES,n_channel=NUM_CHANNEL).cuda()
    elif 'frustum_convnet_globalfusion_v1' in MODEL_FILE:
        from frustum_convnet_globalfusion_v1 import FrustumConvNetv1
        model = FrustumConvNetv1(n_classes=NUM_CLASSES,n_channel=NUM_CHANNEL).cuda()
    elif 'frustum_convnet_paintfusion_v1' in MODEL_FILE:
        from frustum_convnet_paintfusion_v1 import FrustumConvNetv1
        model = FrustumConvNetv1(n_classes=NUM_CLASSES,n_channel=NUM_CHANNEL).cuda()

    # set optimizer and scheduler
    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=BASE_LR,
            betas=(0.9, 0.999),eps=1e-08,
            weight_decay=WEIGHT_DECAY)
    '''
    def lr_func(epoch, init=BASE_LR, step_size=LR_STEPS, gamma=GAMMA, eta_min=MIN_LR):
        f = gamma**(epoch//LR_STEPS)
        if init*f>eta_min:
            return f
        else:
            return 0.01#0.001*0.01 = eta_min
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)
    '''
    if len(LR_STEPS) > 1:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEPS, gamma=GAMMA)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEPS[0], gamma=GAMMA)
    # train
    if USE_TFBOARD:
        if os.path.exists('runs/' + NAME):
            print('name has been existed')
            shutil.rmtree('runs/' + NAME)
        writer = SummaryWriter('runs/' + NAME)

    num_batch = len(TRAIN_DATASET) / BATCH_SIZE
    best_iou3d_70 = 0.0
    best_epoch = 1
    best_file = ''

    for epoch in range(MAX_EPOCH):
        log_string('**** cfg:%s ****' % (args.cfg))
        log_string('**** output_dir:%s ****' % (OUTPUT_DIR))
        log_string('**** EPOCH %03d ****' % (epoch + 1))
        sys.stdout.flush()
        log_string('Epoch %d/%s:' % (epoch + 1, MAX_EPOCH))
        # record for one epoch
        train_total_loss = 0.0
        train_iou2d = 0.0
        train_iou3d = 0.0
        train_acc = 0.0
        train_iou3d_70 = 0.0
        '''deprecated
        if FLAGS.return_all_loss:
            train_mask_loss = 0.0
            train_center_loss = 0.0
            train_heading_class_loss = 0.0
            train_size_class_loss = 0.0
            train_heading_residual_normalized_loss = 0.0
            train_size_residual_normalized_loss = 0.0
            train_stage1_center_loss = 0.0
            train_corners_loss = 0.0
        '''
        train_losses = {
            'total_loss': 0.0,
            'cls_loss': 0.0, #fconvnet
            'mask_loss': 0.0,#fpointnet
            'heading_class_loss': 0.0,
            'size_class_loss': 0.0,
            'heading_residual_normalized_loss': 0.0,
            'size_residual_normalized_loss': 0.0,
            'stage1_center_loss': 0.0,
            'corners_loss': 0.0
        }
        train_metrics = {
            'seg_acc': 0.0,#fpointnet
            'cls_acc': 0.0,#fconvnet
            'iou2d': 0.0,
            'iou3d': 0.0,
            'iou3d_0.7': 0.0,
        }
        n_batches = 0
        for i, data_dicts in tqdm(enumerate(train_dataloader),\
                total=len(train_dataloader), smoothing=0.9):
            n_batches += 1
            #for debug
            if args.debug==True:
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
            data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}
            #ipdb.set_trace()
            '''deprecated
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
            '''

            optimizer.zero_grad()
            model = model.train()

            '''
            #bn_decay(defaut 0.1)
            bn_momentum = BN_INIT_DECAY * BN_DECAY_DECAY_RATE**(epoch//BN_DECAY_DECAY_STEP)
            if bn_momentum < 1 - BN_DECAY_CLIP:
                bn_momentum = 1 - BN_DECAY_CLIP
            '''
            '''deprecated
            logits, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residual_normalized, heading_residual, \
            size_scores, size_residual_normalized, size_residual, center = \
                model(data_dicts_var)
            '''

            losses, metrics = model(data_dicts_var)
            '''deprecated
            if FLAGS.return_all_loss:
                total_loss, mask_loss, center_loss, heading_class_loss, \
                    size_class_loss, heading_residual_normalized_loss, \
                    size_residual_normalized_loss, stage1_center_loss, \
                    corners_loss = \
                    Loss(logits, batch_label, \
                        center, batch_center, stage1_center, \
                        heading_scores, heading_residual_normalized, \
                        heading_residual, \
                        batch_hclass, batch_hres, \
                        size_scores,size_residual_normalized,\
                        size_residual,\
                        batch_sclass,batch_sres)
            else:
                total_loss = \
                    Loss(logits, batch_label, \
                        center, batch_center, stage1_center, \
                        heading_scores, heading_residual_normalized, \
                        heading_residual, \
                        batch_hclass, batch_hres, \
                        size_scores,size_residual_normalized,\
                        size_residual,\
                        batch_sclass,batch_sres)
            '''

            total_loss = losses['total_loss']
            #total_loss = total_loss.mean()
            total_loss.backward()

            optimizer.step()

            '''deprecated
            train_total_loss += total_loss.item()

            iou2ds, iou3ds = provider.compute_box3d_iou(\
                center.cpu().detach().numpy(),\
                heading_scores.cpu().detach().numpy(),\
                heading_residual.cpu().detach().numpy(), \
                size_scores.cpu().detach().numpy(), \
                size_residual.cpu().detach().numpy(), \
                batch_center.cpu().detach().numpy(), \
                batch_hclass.cpu().detach().numpy(), \
                batch_hres.cpu().detach().numpy(), \
                batch_sclass.cpu().detach().numpy(), \
                batch_sres.cpu().detach().numpy())
            train_iou2d += np.sum(iou2ds)
            train_iou3d += np.sum(iou3ds)
            train_iou3d_70 += np.sum(iou3ds>=0.7)
            
            correct = torch.argmax(logits, 2).eq(batch_label.long()).detach().cpu().numpy()
            accuracy = np.sum(correct)
            train_acc += accuracy
            if FLAGS.return_all_loss:
                train_mask_loss += mask_loss.item()
                train_center_loss += center_loss.item()
                train_heading_class_loss += heading_class_loss.item()
                train_size_class_loss += size_class_loss.item()
                train_heading_residual_normalized_loss += heading_residual_normalized_loss.item()
                train_size_residual_normalized_loss += size_residual_normalized_loss.item()
                train_stage1_center_loss += stage1_center_loss.item()
                train_corners_loss += corners_loss.item()

            '''

            for key in train_losses.keys():
                if key in losses.keys():
                    train_losses[key] += losses[key].detach().item()
            for key in train_metrics.keys():
                if key in metrics.keys():
                    train_metrics[key] += metrics[key]

            '''deprecated
            print('[%d: %d/%d] train loss: %.6f' % \
                  (epoch + 1, i, len(train_dataloader),(train_total_loss/n_samples)))
            print('box IoU(ground/3D): %.6f/%.6f' % (train_iou2d/n_samples, train_iou3d/n_samples))
            print('box estimation accuracy (IoU=0.7): %.6f' % (train_iou3d_70/n_samples))
            if FLAGS.return_all_loss:
                print('train_mask_loss:%.6f'%(train_mask_loss/n_samples))
                print('train_stage1_center_loss:%.6f' % (train_stage1_center_loss/n_samples))
                print('train_heading_class_loss:%.6f' % (train_heading_class_loss/n_samples))
                print('train_size_class_loss:%.6f' % (train_size_class_loss/n_samples))
                print('train_heading_residual_normalized_loss:%.6f' % (train_heading_residual_normalized_loss/n_samples))
                print('train_size_residual_normalized_loss:%.6f' % (train_size_residual_normalized_loss/n_samples))
                print('train_stage1_center_loss:%.6f' % (train_stage1_center_loss/n_samples))
                print('train_corners_loss:%.6f'%(train_corners_loss/n_samples))
            '''
        for key in train_losses.keys():
            train_losses[key] /= n_batches
        for key in train_metrics.keys():
            train_metrics[key] /= n_batches

        log_string('[%d: %d/%d] train' % (epoch + 1, i, len(train_dataloader)))
        for key, value in train_losses.items():
            if value < 1e-6: continue
            log_string(str(key)+':'+"%.6f"%(value))
        for key, value in train_metrics.items():
            if value < 1e-6: continue
            log_string(str(key)+':'+"%.6f"%(value))
        '''
        train_total_loss /= n_samples
        train_acc /= n_samples*float(NUM_POINT)
        train_iou2d /= n_samples
        train_iou3d /= n_samples
        train_iou3d_70 /= n_samples

        if FLAGS.return_all_loss:
            train_mask_loss /= n_samples
            train_center_loss /= n_samples
            train_heading_class_loss /= n_samples
            train_size_class_loss /= n_samples
            train_heading_residual_normalized_loss /= n_samples
            train_size_residual_normalized_loss /= n_samples
            train_stage1_center_loss /= n_samples
            train_corners_loss /= n_samples
        
        print('[%d: %d/%d] train loss: %.6f' % \
              (epoch + 1, i, len(train_dataloader),train_total_loss))
        print('segmentation accuracy: %.6f'% train_acc )
        print('box IoU(ground/3D): %.6f/%.6f'% (train_iou2d, train_iou3d))
        print('box estimation accuracy (IoU=0.7): %.6f'%(train_iou3d_70))
        '''
        # test one epoch
        test_losses, test_metrics = test_one_epoch(model,test_dataloader)
        log_string('[%d: %d/%d] %s' % (epoch + 1, i, len(train_dataloader),blue('test')))
        for key, value in test_losses.items():
            if value < 1e-6: continue
            log_string(str(key)+':'+"%.6f"%(value))
        for key, value in test_metrics.items():
            if value < 1e-6: continue
            log_string(str(key)+':'+"%.6f"%(value))

        '''deprecated
        print('[%d] %s loss: %.6f' % \
              (epoch + 1, blue('test'), test_total_loss))
        print('%s segmentation accuracy: %.6f'% (blue('test'),test_acc))
        print('%s box IoU(ground/3D): %.6f/%.6f'% (blue('test'),test_iou2d, test_iou3d))
        print('%s box estimation accuracy (IoU=0.7): %.6f'%(blue('test'), test_iou3d_70))
        '''
        scheduler.step()
        if MIN_LR > 0:
            if scheduler.get_lr()[0] < MIN_LR:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = MIN_LR
        log_string("learning rate: {:.6f}".format(optimizer.param_groups[0]['lr']))

        if USE_TFBOARD:
            writer.add_scalar('train_total_loss',train_losses['total_loss'],epoch)
            writer.add_scalar('train_iou3d_0.7',train_metrics['iou3d_0.7'],epoch)
            writer.add_scalar('test_total_loss',test_losses['total_loss'],epoch)
            writer.add_scalar('test_iou3d_0.7',test_metrics['iou3d_0.7'],epoch)
        '''
        if not FLAGS.debug:
            writer.add_scalar('train_total_loss',train_total_loss, epoch)
            writer.add_scalar('train_iou2d',train_iou2d, epoch)
            writer.add_scalar('train_iou3d',train_iou3d, epoch)
            writer.add_scalar('train_acc',train_acc, epoch)
            writer.add_scalar('train_iou3d_0.7',train_iou3d_70, epoch)

        if FLAGS.return_all_loss and not FLAGS.debug:
            writer.add_scalar('train_mask_loss',train_mask_loss)
            writer.add_scalar('train_center_loss',train_center_loss, epoch)
            writer.add_scalar('train_heading_class_loss',train_heading_class_loss, epoch)
            writer.add_scalar('train_size_class_loss',train_size_class_loss, epoch)
            writer.add_scalar('train_heading_residual_normalized_loss',train_heading_residual_normalized_loss, epoch)
            writer.add_scalar('train_size_residual_normalized_loss',train_size_residual_normalized_loss, epoch)
            writer.add_scalar('train_stage1_center_loss',train_stage1_center_loss, epoch)
            writer.add_scalar('train_corners_loss',train_corners_loss, epoch)

        if not FLAGS.debug:
            writer.add_scalar('test_total_loss',test_total_loss, epoch)
            writer.add_scalar('test_iou2d_loss',test_iou2d, epoch)
            writer.add_scalar('test_iou3d_loss',test_iou3d, epoch)
            writer.add_scalar('test_acc',test_acc, epoch)
            writer.add_scalar('test_iou3d_0.7',test_iou3d_70, epoch)

        if FLAGS.return_all_loss:
            writer.add_scalar('test_mask_loss',test_mask_loss, epoch)
            writer.add_scalar('test_center_loss',test_center_loss, epoch)
            writer.add_scalar('test_heading_class_loss',test_heading_class_loss, epoch)
            writer.add_scalar('test_size_class_loss',test_size_class_loss, epoch)
            writer.add_scalar('test_heading_residual_normalized_loss',test_heading_residual_normalized_loss, epoch)
            writer.add_scalar('test_size_residual_normalized_loss',test_size_residual_normalized_loss, epoch)
            writer.add_scalar('test_stage1_center_loss',test_stage1_center_loss, epoch)
            writer.add_scalar('test_corners_loss',test_corners_loss, epoch)
        '''
        if test_metrics['iou3d_0.7'] >= best_iou3d_70:
            best_iou3d_70 = test_metrics['iou3d_0.7']
            best_epoch = epoch + 1
            if epoch > MAX_EPOCH / 5:
                savepath = LOG_DIR + '/acc%.3f-epoch%03d.pth' % \
                           (test_metrics['iou3d_0.7'], epoch)
                log_string('save to:'+str(savepath))
                if os.path.exists(best_file):
                    os.remove(best_file)# update to newest best epoch
                best_file = savepath
                state = {
                    'epoch': epoch + 1,
                    'train_iou3d_0.7': train_metrics['iou3d_0.7'],
                    'test_iou3d_0.7': test_metrics['iou3d_0.7'],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state,savepath)
                log_string('Saving model to %s'%savepath)
        log_string('Best Test acc: %f(Epoch %d)' % (best_iou3d_70, best_epoch))
    log_string("Time {} hours".format(float(time.perf_counter()-start)/3600))
    if USE_TFBOARD:
        writer.close()

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    print('Your args:')
    print(args)
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
    train_heading_residual_normalized_loss:0.001008
    train_size_residual_normalized_loss:0.009315
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
    train_heading_residual_normalized_loss:0.001029
    train_size_residual_normalized_loss:0.007258
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
