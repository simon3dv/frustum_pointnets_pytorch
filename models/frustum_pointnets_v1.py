import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'train'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from torch.nn import init
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, parse_output_to_tensors
from model_util import FrustumPointNetLoss
#from model_util import g_type2class, g_class2type, g_type2onehotclass
#from model_util import g_type_mean_size
#from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from provider import compute_box3d_iou


NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512

g_type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3,
              'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
g_class2type = {g_type2class[t]:t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                    'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Misc': np.array([3.64300781,1.54298177,1.92320313])}


g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]


class PointNetInstanceSeg(nn.Module):
    def __init__(self,n_classes=3,n_channel=3):
        '''v1 3D Instance Segmentation PointNet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetInstanceSeg, self).__init__()
        self.conv1 = nn.Conv1d(n_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.n_classes = n_classes
        self.dconv1 = nn.Conv1d(1088+n_classes, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)

    def forward(self, pts, one_hot_vec): # bs,4,n
        '''
        :param pts: [bs,4,n]: x,y,z,intensity
        :return: logits: [bs,n,2],scores for bkg/clutter and object
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,64,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,64,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,64,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,128,n
        out5 = F.relu(self.bn5(self.conv5(out4)))# bs,1024,n
        global_feat = torch.max(out5, 2, keepdim=True)[0] #bs,1024,1

        expand_one_hot_vec = one_hot_vec.view(bs,-1,1)#bs,3,1
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,1027,1
        expand_global_feat_repeat = expand_global_feat.view(bs,-1,1)\
                .repeat(1,1,n_pts)# bs,1027,n
        concat_feat = torch.cat([out2,\
            expand_global_feat_repeat],1)
        # bs, (641024+3)=1091, n

        x = F.relu(self.dbn1(self.dconv1(concat_feat)))#bs,512,n
        x = F.relu(self.dbn2(self.dconv2(x)))#bs,256,n
        x = F.relu(self.dbn3(self.dconv3(x)))#bs,128,n
        x = F.relu(self.dbn4(self.dconv4(x)))#bs,128,n
        x = self.dropout(x)
        x = self.dconv5(x)#bs, 2, n

        seg_pred = x.transpose(2,1).contiguous()#bs, n, 2
        return seg_pred

class PointNetEstimation(nn.Module):
    def __init__(self,n_classes=2):
        '''v1 Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.n_classes = n_classes

        self.fc1 = nn.Linear(512+3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts,one_hot_vec): # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residual,
            and size cluster scores and residual
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,128,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,128,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,256,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,512,n
        global_feat = torch.max(out4, 2, keepdim=False)[0] #bs,512

        expand_one_hot_vec = one_hot_vec.view(bs,-1)#bs,3
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,515

        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))#bs,512
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,256
        box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred

class STNxyz(nn.Module):
    def __init__(self,n_classes=3):
        super(STNxyz, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        #self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(256+n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        init.zeros_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)
    def forward(self, pts,one_hot_vec):
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))# bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))# bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))# bs,256,n
        x = torch.max(x, 2)[0]# bs,256
        expand_one_hot_vec = one_hot_vec.view(bs, -1)# bs,3
        x = torch.cat([x, expand_one_hot_vec],1)#bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))# bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))# bs,128
        x = self.fc3(x)# bs,
        return x

class FrustumPointNetv1(nn.Module):
    def __init__(self,n_classes=3,n_channel=3):
        super(FrustumPointNetv1, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.InsSeg = PointNetInstanceSeg(n_classes=3,n_channel=n_channel)
        self.STN = STNxyz(n_classes=3)
        self.est = PointNetEstimation(n_classes=3)
        self.Loss = FrustumPointNetLoss()
    def forward(self, data_dicts):
        #dict_keys(['point_cloud', 'rot_angle', 'box3d_center', 'size_class', 'size_residual', 'angle_class', 'angle_residual', 'one_hot', 'seg'])

        point_cloud = data_dicts.get('point_cloud')#torch.Size([32, 4, 1024])
        point_cloud = point_cloud[:,:self.n_channel,:]
        one_hot = data_dicts.get('one_hot')#torch.Size([32, 3])
        bs = point_cloud.shape[0]
        # If not None, use to Compute Loss
        seg_label = data_dicts.get('seg')#torch.Size([32, 1024])
        box3d_center_label = data_dicts.get('box3d_center')#torch.Size([32, 3])
        size_class_label = data_dicts.get('size_class')#torch.Size([32, 1])
        size_residual_label = data_dicts.get('size_residual')#torch.Size([32, 3])
        heading_class_label = data_dicts.get('angle_class')#torch.Size([32, 1])
        heading_residual_label = data_dicts.get('angle_residual')#torch.Size([32, 1])

        # 3D Instance Segmentation PointNet
        logits = self.InsSeg(point_cloud, one_hot)#bs,n,2

        # Mask Point Centroid
        object_pts_xyz, mask_xyz_mean, mask = \
                 point_cloud_masking(point_cloud, logits)

        # T-Net
        object_pts_xyz = object_pts_xyz.cuda()
        center_delta = self.STN(object_pts_xyz,one_hot)#(32,3)
        stage1_center = center_delta + mask_xyz_mean#(32,3)

        if(np.isnan(stage1_center.cpu().detach().numpy()).any()):
            ipdb.set_trace()
        object_pts_xyz_new = object_pts_xyz - \
                    center_delta.view(center_delta.shape[0],-1,1).repeat(1,1,object_pts_xyz.shape[-1])

        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new,one_hot)#(32, 59)

        center_boxnet, \
        heading_scores, heading_residual_normalized, heading_residual, \
        size_scores, size_residual_normalized, size_residual = \
                parse_output_to_tensors(box_pred, logits, mask, stage1_center)

        box3d_center = center_boxnet + stage1_center #bs,3

        losses = self.Loss(logits, seg_label, \
                 box3d_center, box3d_center_label, stage1_center, \
                 heading_scores, heading_residual_normalized, \
                 heading_residual, \
                 heading_class_label, heading_residual_label, \
                 size_scores, size_residual_normalized, \
                 size_residual, \
                 size_class_label, size_residual_label)

        for key in losses.keys():
            losses[key] = losses[key]/bs

        with torch.no_grad():
            seg_correct = torch.argmax(logits.detach().cpu(), 2).eq(seg_label.detach().cpu()).numpy()
            seg_accuracy = np.sum(seg_correct) / float(point_cloud.shape[-1])

            iou2ds, iou3ds = compute_box3d_iou( \
                box3d_center.detach().cpu().numpy(),
                heading_scores.detach().cpu().numpy(),
                heading_residual.detach().cpu().numpy(),
                size_scores.detach().cpu().numpy(),
                size_residual.detach().cpu().numpy(),
                box3d_center_label.detach().cpu().numpy(),
                heading_class_label.detach().cpu().numpy(),
                heading_residual_label.detach().cpu().numpy(),
                size_class_label.detach().cpu().numpy(),
                size_residual_label.detach().cpu().numpy())
        metrics = {
            'seg_acc': seg_accuracy,
            'iou2d': iou2ds.mean(),
            'iou3d': iou3ds.mean(),
            'iou3d_0.7': np.sum(iou3ds >= 0.7)/bs
        }
        return losses, metrics


if __name__ == '__main__':
    from provider import FrustumDataset
    dataset = FrustumDataset(npoints=1024, split='val',
        rotate_to_center=True, random_flip=False, random_shift=False, one_hot=True,
        overwritten_data_path='kitti/frustum_caronly_val.pickle',
        gen_ref = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)
    model = FrustumPointNetv1().cuda()
    for batch, data_dicts in enumerate(dataloader):
        data_dicts_var = {key: value.squeeze().cuda() for key, value in data_dicts.items()}
        losses, metrics= model(data_dicts_var)

        print()
        for key,value in losses.items():
            print(key,value)
        print()
        for key,value in metrics.items():
            print(key,value)
        input()
    '''
    total_loss tensor(50.4213, device='cuda:0', grad_fn=<AddBackward0>)
    mask_loss tensor(5.5672, device='cuda:0', grad_fn=<MulBackward0>)
    heading_class_loss tensor(2.5698, device='cuda:0', grad_fn=<MulBackward0>)
    size_class_loss tensor(0.9636, device='cuda:0', grad_fn=<MulBackward0>)
    heading_residual_normalized_loss tensor(9.8356, device='cuda:0', grad_fn=<MulBackward0>)
    size_residual_normalized_loss tensor(4.0655, device='cuda:0', grad_fn=<MulBackward0>)
    stage1_center_loss tensor(4.0655, device='cuda:0', grad_fn=<MulBackward0>)
    corners_loss tensor(26.4575, device='cuda:0', grad_fn=<MulBackward0>)
    
    seg_acc 16.07421875
    iou2d 0.066525616
    iou3d 0.045210287
    iou3d_acc 0.0
    '''
    '''
    data_dicts = {
        'point_cloud': torch.zeros(size=(32,1024,4),dtype=torch.float32).transpose(2, 1),
        'rot_angle': torch.zeros(32).float(),
        'box3d_center': torch.zeros(32,3).float(),
        'size_class': torch.zeros(32),
        'size_residual': torch.zeros(32,3).float(),
        'angle_class': torch.zeros(32).long(),
        'angle_residual': torch.zeros(32).float(),
        'one_hot': torch.zeros(32,3).float(),
        'seg': torch.zeros(32,1024).float()
    }
    data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}
    model = FrustumPointNetv1().cuda()
    losses, metrics= model(data_dicts_var)

    print()
    for key,value in losses.items():
        print(key,value)
    print()
    for key,value in metrics.items():
        print(key,value)
    '''