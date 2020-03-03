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
from model_util_old import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util_old import point_cloud_masking, parse_output_to_tensors
from model_util_old import FrustumPointNetLoss
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
    def __init__(self,n_classes=3,n_channel=4):
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
    def __init__(self,n_classes=3):
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

        self.fc1 = nn.Linear(512+n_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts,one_hot_vec): # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
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
        ###if np.isnan(x.cpu().detach().numpy()).any():
        ###    ipdb.set_trace()
        return x

class FrustumPointNetv1(nn.Module):
    def __init__(self,n_classes=3,n_channel=4):
        super(FrustumPointNetv1, self).__init__()
        self.n_classes = n_classes
        self.InsSeg = PointNetInstanceSeg(n_classes=3,n_channel=n_channel)
        self.STN = STNxyz(n_classes=3)
        self.est = PointNetEstimation(n_classes=3)

    def forward(self, pts, one_hot_vec):#bs,4,n
        # 3D Instance Segmentation PointNet
        logits = self.InsSeg(pts,one_hot_vec)#bs,n,2

        # Mask Point Centroid
        object_pts_xyz, mask_xyz_mean, mask = \
                 point_cloud_masking(pts, logits)###logits.detach()

        # T-Net
        object_pts_xyz = object_pts_xyz.cuda()
        center_delta = self.STN(object_pts_xyz,one_hot_vec)#(32,3)
        stage1_center = center_delta + mask_xyz_mean#(32,3)

        if(np.isnan(stage1_center.cpu().detach().numpy()).any()):
            ipdb.set_trace()
        object_pts_xyz_new = object_pts_xyz - \
                    center_delta.view(center_delta.shape[0],-1,1).repeat(1,1,object_pts_xyz.shape[-1])

        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new,one_hot_vec)#(32, 59)

        center_boxnet, \
        heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals = \
                parse_output_to_tensors(box_pred, logits, mask, stage1_center)

        center = center_boxnet + stage1_center #bs,3
        return logits, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residuals_normalized, heading_residuals, \
            size_scores, size_residuals_normalized, size_residuals, center




if __name__ == '__main__':
    #python models/pointnet.py
    points = torch.zeros(size=(32,4,1024),dtype=torch.float32)
    label = torch.ones(size=(32,3))
    model = FrustumPointNetv1()
    logits, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residuals_normalized, heading_residuals, \
            size_scores, size_residuals_normalized, size_residuals, center \
            = model(points,label)
    print('logits:',logits.shape,logits.dtype)
    print('mask:',mask.shape,mask.dtype)
    print('stage1_center:',stage1_center.shape,stage1_center.dtype)
    print('center_boxnet:',center_boxnet.shape,center_boxnet.dtype)
    print('heading_scores:',heading_scores.shape,heading_scores.dtype)
    print('heading_residuals_normalized:',heading_residuals_normalized.shape,\
          heading_residuals_normalized.dtype)
    print('heading_residuals:',heading_residuals.shape,\
          heading_residuals.dtype)
    print('size_scores:',size_scores.shape,size_scores.dtype)
    print('size_residuals_normalized:',size_residuals_normalized.shape,\
          size_residuals_normalized.dtype)
    print('size_residuals:',size_residuals.shape,size_residuals.dtype)
    print('center:', center.shape,center.dtype)
    '''
    logits: torch.Size([32, 1024, 2]) torch.float32
    mask: torch.Size([32, 1024]) torch.float32
    stage1_center: torch.Size([32, 3]) torch.float32
    center_boxnet: torch.Size([32, 3]) torch.float32
    heading_scores: torch.Size([32, 12]) torch.float32
    heading_residual_normalized: torch.Size([32, 12]) torch.float32
    heading_residual: torch.Size([32, 12]) torch.float32
    size_scores: torch.Size([32, 8]) torch.float32
    size_residuals_normalized: torch.Size([32, 8, 3]) torch.float32
    size_residuals: torch.Size([32, 8, 3]) torch.float32
    center: torch.Size([32, 3]) torch.float32
    '''
    loss = FrustumPointNetLoss()
    mask_label = torch.zeros(32,1024).float()
    center_label = torch.zeros(32,3).float()
    heading_class_label = torch.zeros(32).long()
    heading_residuals_label = torch.zeros(32).float()
    size_class_label = torch.zeros(32).long()
    size_residuals_label = torch.zeros(32,3).float()
    output_loss = loss(logits, mask_label, \
                center, center_label, stage1_center, \
                heading_scores, heading_residuals_normalized, heading_residuals, \
                heading_class_label, heading_residuals_label, \
                size_scores,size_residuals_normalized,size_residuals,
                size_class_label,size_residuals_label)
    print('output_loss',output_loss)
    print()