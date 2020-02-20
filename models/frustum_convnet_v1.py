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
from torch.autograd import Variable
import ipdb
from torch.nn import init
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, parse_output_to_tensors
from model_util import FrustumPointNetLoss

from model_util import g_type2class, g_class2type, g_type2onehotclass
from model_util import g_type_mean_size
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from provider import compute_box3d_iou
from configs.config import cfg

from ops.query_depth_point.query_depth_point import QueryDepthPoint
from ops.pybind11.box_ops_cc import rbbox_iou_3d_pair

def Conv2d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.Conv2d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm2d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.Conv2d(i_c, o_c, k, s, p), nn.ReLU(True))

def init_params(m, method='constant'):
    """
    method: xavier_uniform, kaiming_normal, constant
    """
    if isinstance(m, list):
        for im in m:
            init_params(im, method)
    else:
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data)
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        elif isinstance(method, (int, float)):
            m.weight.data.fill_(method)
        else:
            raise ValueError("unknown method.")
        if m.bias is not None:
            m.bias.data.zero_()

# single scale PointNet module
class PointNetModule(nn.Module):
    def __init__(self, Infea, mlp, dist, nsample, use_xyz=True, use_feature=True):
        super(PointNetModule, self).__init__()
        self.dist = dist
        self.nsample = nsample
        self.use_xyz = use_xyz

        if Infea > 0:
            use_feature = True
        else:
            use_feature = False

        self.use_feature = use_feature

        self.query_depth_point = QueryDepthPoint(dist, nsample)

        if self.use_xyz:
            self.conv1 = Conv2d(Infea + 3, mlp[0], 1)
        else:
            self.conv1 = Conv2d(Infea, mlp[0], 1)

        self.conv2 = Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = Conv2d(mlp[1], mlp[2], 1)

        init_params([self.conv1[0], self.conv2[0], self.conv3[0]], 'kaiming_normal')
        init_params([self.conv1[1], self.conv2[1], self.conv3[1]], 1)

    def forward(self, pc, feat, new_pc=None):
        batch_size = pc.shape[0]
        ipdb.set_trace()
        npoint = new_pc.shape[2]
        k = self.nsample
        indices, num = self.query_depth_point(pc, new_pc)  # b*npoint*nsample

        assert indices.data.max() < pc.shape[2] and indices.data.min() >= 0
        grouped_pc = None
        grouped_feature = None

        if self.use_xyz:
            grouped_pc = torch.gather(
                pc, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, 3, -1)
            ).view(batch_size, 3, npoint, k)

            grouped_pc = grouped_pc - new_pc.unsqueeze(3)

        if self.use_feature:
            grouped_feature = torch.gather(
                feat, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, feat.size(1), -1)
            ).view(batch_size, feat.size(1), npoint, k)

            # grouped_feature = torch.cat([new_feat.unsqueeze(3), grouped_feature], -1)

        if self.use_feature and self.use_xyz:
            grouped_feature = torch.cat([grouped_pc, grouped_feature], 1)
        elif self.use_xyz:
            grouped_feature = grouped_pc.contiguous()

        grouped_feature = self.conv1(grouped_feature)
        grouped_feature = self.conv2(grouped_feature)
        grouped_feature = self.conv3(grouped_feature)
        # output, _ = torch.max(grouped_feature, -1)

        valid = (num > 0).view(batch_size, 1, -1, 1)
        grouped_feature = grouped_feature * valid.float()

        return grouped_feature

# multi-scale PointNet module
class PointNetFeat(nn.Module):
    def __init__(self, input_channel=3, num_vec=0):
        super(PointNetFeat, self).__init__()

        self.num_vec = num_vec
        u = cfg.DATA.HEIGHT_HALF
        assert len(u) == 4
        self.pointnet1 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[0], 32, use_xyz=True, use_feature=True)

        self.pointnet2 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[1], 64, use_xyz=True, use_feature=True)

        self.pointnet3 = PointNetModule(
            input_channel - 3, [128, 128, 256], u[2], 64, use_xyz=True, use_feature=True)

        self.pointnet4 = PointNetModule(
            input_channel - 3, [256, 256, 512], u[3], 128, use_xyz=True, use_feature=True)

    def forward(self, point_cloud, sample_pc, feat=None, one_hot_vec=None):
        pc = point_cloud#torch.Size([32, 3, 1024])
        pc1 = sample_pc[0]
        pc2 = sample_pc[1]
        pc3 = sample_pc[2]
        pc4 = sample_pc[3]
        ipdb.set_trace()
        feat1 = self.pointnet1(pc, feat, pc1)
        feat1, _ = torch.max(feat1, -1)

        feat2 = self.pointnet2(pc, feat, pc2)
        feat2, _ = torch.max(feat2, -1)

        feat3 = self.pointnet3(pc, feat, pc3)
        feat3, _ = torch.max(feat3, -1)

        feat4 = self.pointnet4(pc, feat, pc4)
        feat4, _ = torch.max(feat4, -1)

        if one_hot_vec is not None:
            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat1.shape[-1])
            feat1 = torch.cat([feat1, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat2.shape[-1])
            feat2 = torch.cat([feat2, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat3.shape[-1])
            feat3 = torch.cat([feat3, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat4.shape[-1])
            feat4 = torch.cat([feat4, one_hot], 1)

        return feat1, feat2, feat3, feat4


# FCN
class ConvFeatNet(nn.Module):
    def __init__(self, i_c=128, num_vec=3):
        super(ConvFeatNet, self).__init__()

        self.block1_conv1 = Conv1d(i_c + num_vec, 128, 3, 1, 1)

        self.block2_conv1 = Conv1d(128, 128, 3, 2, 1)
        self.block2_conv2 = Conv1d(128, 128, 3, 1, 1)
        self.block2_merge = Conv1d(128 + 128 + num_vec, 128, 1, 1)

        self.block3_conv1 = Conv1d(128, 256, 3, 2, 1)
        self.block3_conv2 = Conv1d(256, 256, 3, 1, 1)
        self.block3_merge = Conv1d(256 + 256 + num_vec, 256, 1, 1)

        self.block4_conv1 = Conv1d(256, 512, 3, 2, 1)
        self.block4_conv2 = Conv1d(512, 512, 3, 1, 1)
        self.block4_merge = Conv1d(512 + 512 + num_vec, 512, 1, 1)

        self.block2_deconv = DeConv1d(128, 256, 1, 1, 0)
        self.block3_deconv = DeConv1d(256, 256, 2, 2, 0)
        self.block4_deconv = DeConv1d(512, 256, 4, 4, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                # nn.init.xavier_uniform_(m.weight.data)
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3, x4):

        x = self.block1_conv1(x1)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = torch.cat([x, x2], 1)
        x = self.block2_merge(x)
        xx1 = x

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = torch.cat([x, x3], 1)
        x = self.block3_merge(x)
        xx2 = x

        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = torch.cat([x, x4], 1)
        x = self.block4_merge(x)
        xx3 = x

        xx1 = self.block2_deconv(xx1)
        xx2 = self.block3_deconv(xx2)
        xx3 = self.block4_deconv(xx3)

        x = torch.cat([xx1, xx2[:, :, :xx1.shape[-1]], xx3[:, :, :xx1.shape[-1]]], 1)

        return x

class FrustumConvNetv1(nn.Module):
    def __init__(self,n_classes=3,n_channel=4):
        super(FrustumConvNetv1, self).__init__()
        self.n_classes = n_classes
        self.feat_net = PointNetFeat(n_channel, 0)
        #self.InsSeg = PointNetInstanceSeg(n_classes=3,n_channel=n_channel)
        #self.STN = STNxyz(n_classes=3)
        #self.est = PointNetEstimation(n_classes=3)
        #self.Loss = FrustumPointNetLoss()
    def forward(self, data_dicts):
        #dict_keys(['point_cloud', 'rot_angle', 'box3d_center', 'size_class', 'size_residual', 'angle_class', 'angle_residual', 'one_hot', 'seg'])

        point_cloud = data_dicts.get('point_cloud')#torch.Size([32, 4, 1024])
        one_hot = data_dicts.get('one_hot')#torch.Size([32, 3])

        # If not None, use to Compute Loss
        seg_label = data_dicts.get('seg')#torch.Size([32, 1024])
        box3d_center_label = data_dicts.get('box3d_center')#torch.Size([32, 3])
        size_class_label = data_dicts.get('size_class')#torch.Size([32])
        size_residual_label = data_dicts.get('size_residual')#torch.Size([32, 3])
        heading_class_label = data_dicts.get('angle_class')#torch.Size([32])
        heading_residual_label = data_dicts.get('angle_residual')#torch.Size([32])

        center_ref1 = data_dicts.get('center_ref1')#torch.Size([3, 280])
        center_ref2 = data_dicts.get('center_ref2')#torch.Size([3, 140])
        center_ref3 = data_dicts.get('center_ref3')#torch.Size([3, 70])
        center_ref4 = data_dicts.get('center_ref4')#torch.Size([3, 35])

        object_point_cloud_xyz = point_cloud[:,:3,:].contiguous()
        if point_cloud.shape[1] > 3:
            object_point_cloud_i = point_cloud[:,[3],:].contiguous()
        else:
            object_point_cloud_i = None
        ipdb.set_trace()
        feat1, feat2, feat3, feat4 = self.feat_net(
            object_point_cloud_xyz,
            [center_ref1, center_ref2, center_ref3, center_ref4],
            object_point_cloud_i,
            one_hot)
        #feat1:torch.Size([32, 131, 280])
        #feat2:torch.Size([32, 131, 140])
        #feat3:torch.Size([32, 131, 70])
        #feat4:torch.Size([32, 131, 35])
        x = self.conv_net(feat1, feat2, feat3, feat4)







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
        heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals = \
                parse_output_to_tensors(box_pred, logits, mask, stage1_center)

        box3d_center = center_boxnet + stage1_center #bs,3

        losses = self.Loss(logits, seg_label, \
                 box3d_center, box3d_center_label, stage1_center, \
                 heading_scores, heading_residuals_normalized, \
                 heading_residuals, \
                 heading_class_label, heading_residual_label, \
                 size_scores, size_residuals_normalized, \
                 size_residuals, \
                 size_class_label, size_residual_label)

        with torch.no_grad():
            seg_correct = torch.argmax(logits.detach().cpu(), 2).eq(seg_label.detach().cpu()).numpy()
            seg_accuracy = np.sum(seg_correct) / float(point_cloud.shape[-1])

            iou2ds, iou3ds = compute_box3d_iou( \
                box3d_center.detach().cpu().numpy(),
                heading_scores.detach().cpu().numpy(),
                heading_residuals.detach().cpu().numpy(),
                size_scores.detach().cpu().numpy(),
                size_residuals.detach().cpu().numpy(),
                box3d_center_label.detach().cpu().numpy(),
                heading_class_label.detach().cpu().numpy(),
                heading_residual_label.detach().cpu().numpy(),
                size_class_label.detach().cpu().numpy(),
                size_residual_label.detach().cpu().numpy())
            iou3ds_acc = np.sum(iou3ds >= 0.7)
        metrics = {
            'seg_acc': seg_accuracy,
            'iou2d': iou2ds.mean(),
            'iou3d': iou3ds.mean(),
            'iou3d_acc': iou3ds_acc.mean()
        }
        return losses, metrics


if __name__ == '__main__':
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
