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
from model_util import huber_loss

from model_util import g_type2class, g_class2type, g_type2onehotclass
from model_util import g_type_mean_size, g_mean_size_arr
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from model_util import get_box3d_corners_helper
from provider_fusion import compute_box3d_iou
from configs.config import cfg

from ops.query_depth_point.query_depth_point import QueryDepthPoint
from ops.pybind11.box_ops_cc import rbbox_iou_3d_pair
from models.pspnet import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

def get_accuracy(output, target, ignore=None):

    assert output.shape[0] == target.shape[0]
    if ignore is not None:
        assert isinstance(ignore, int)
        keep = (target != ignore).nonzero().view(-1)
        output = output[keep]
        target = target[keep]

    pred = torch.argmax(output, -1)

    correct = (pred.view(-1) == target.view(-1)).float().sum()
    acc = correct * (1.0 / target.view(-1).shape[0])

    return acc

def softmax_focal_loss_ignore(prob, target, alpha=0.25, gamma=2, ignore_idx=-1):
    keep = (target != ignore_idx).nonzero().view(-1)
    num_fg = (target > 0).data.sum()

    target = target[keep]
    prob = prob[keep, :]

    alpha_t = (1 - alpha) * (target == 0).float() + alpha * (target >= 1).float()

    prob_t = prob[range(len(target)), target]
    # alpha_t = alpha_t[range(len(target)), target]
    loss = -alpha_t * (1 - prob_t) ** gamma * torch.log(prob_t + 1e-14)

    loss = loss.sum() / (num_fg + 1e-14)

    return loss

def size_decode(offset, class_mean_size, size_class_label):

    offset_select = torch.gather(offset, 1, size_class_label.view(-1, 1, 1).expand(-1, -1, 3))
    offset_select = offset_select.squeeze(1)

    ex = class_mean_size[size_class_label]

    return offset_select * ex + ex

def size_encode(gt, class_mean_size, size_class_label):
    ex = class_mean_size[size_class_label]
    return (gt - ex) / ex

def center_decode(ex, offset):
    return ex + offset

def center_encode(gt, ex):
    return gt - ex

def angle_decode(ex_res, ex_class_id, num_bins=12, to_label_format=True):

    ex_res_select = torch.gather(ex_res, 1, ex_class_id.unsqueeze(1))
    ex_res_select = ex_res_select.squeeze(1)

    angle_per_class = 2 * np.pi / float(num_bins)

    angle = ex_class_id.float() * angle_per_class + ex_res_select * (angle_per_class / 2)

    if to_label_format:
        flag = angle > np.pi
        angle[flag] = angle[flag] - 2 * np.pi

    return angle

# def angle_encode(gt_angle, num_bins=12):
#     gt_angle = gt_angle % (2 * np.pi)
#     angle_per_class = 2 * np.pi / float(num_bins)

#     gt_class_id = torch.round(gt_angle / angle_per_class).long()
#     gt_res = gt_angle - gt_class_id.float() * angle_per_class

#     gt_res /= angle_per_class
#     print(gt_class_id.min().item(), gt_class_id.max().item())
#     return gt_class_id, gt_res

def angle_encode(gt_angle, num_bins=12):
    gt_angle = gt_angle % (2 * np.pi)
    assert ((gt_angle >= 0) & (gt_angle <= 2 * np.pi)).all()

    angle_per_class = 2 * np.pi / float(num_bins)
    shifted_angle = (gt_angle + angle_per_class / 2) % (2 * np.pi)
    gt_class_id = torch.floor(shifted_angle / angle_per_class).long()
    gt_res = shifted_angle - (gt_class_id.float() * angle_per_class + angle_per_class / 2)

    gt_res /= (angle_per_class / 2)
    return gt_class_id, gt_res

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
        self.mlp = mlp

        if Infea > 0:
            use_feature = True
        else:
            use_feature = False

        self.use_feature = use_feature

        self.query_depth_point = QueryDepthPoint(dist, nsample)

        self.conv1 = Conv2d(Infea + 3, mlp[0], 1)
        self.conv2 = Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = Conv2d(mlp[1], mlp[2], 1)
        self.conv4 = Conv2d(mlp[2], mlp[3], 1)

        init_params([self.conv1[0], self.conv2[0], self.conv3[0], self.conv4[0]], 'kaiming_normal')###
        init_params([self.conv1[1], self.conv2[1], self.conv3[1], self.conv4[1]], 1)###
    """
    def cart2hom(self, pts_3d):
        ''' Input: 3xn points in Cartesian
            Oupput: 4xn points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[1]
        pts_3d_hom = torch.vstack((pts_3d, torch.ones((1,n))))
        return pts_3d_hom
    
    def project_rect_to_image(self, pts_3d_rect, P):
        ''' Input: 3xn points in rect camera coord.
            Output: 2xn points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = torch.dot(pts_3d_rect, np.transpose(P)) # 3xn
        pts_2d[0,:] /= pts_2d[2,:]
        pts_2d[1,:] /= pts_2d[2,:]
        return pts_2d[0:2,:]
    """
    def query_rgb_point(self, pc, img, P):
        pass
        '''
        bs = pc.shape[0]
        n_channel = pc.shape[1]
        n_points = pc.shape[2]
        pc_2d = torch.zeros(bs,n_channel-1,n_points)
        for b in range(bs):
            pc_2d[b,...] = self.project_rect_to_image(pc[b,...].squeeze(), P).unsqueeze(0)
        '''

    def forward(self, pc, feat, img1, img2, P, query_v1, new_pc=None):
        batch_size = pc.shape[0]
        npoint = new_pc.shape[2]
        k = self.nsample
        indices, num = self.query_depth_point(pc, new_pc)  # b*npoint*nsample
        #ipdb> indices.shape torch.Size([32, 140, 64])
        #ipdb> num.shape torch.Size([32, 140])

        # indices_rgb = self.query_rgb_point(pc, img) # torch.Size([32, 140, 64])
        indices_rgb = torch.gather(query_v1, 1, indices.view(batch_size, npoint*k))\
            .view(batch_size,npoint,k)#torch.Size([32, 140, 64])

        assert indices_rgb.data.max() < img1.shape[2]*img1.shape[3]
        assert indices_rgb.data.min() >= 0
        assert indices.data.max() < pc.shape[2] and indices.data.min() >= 0
        grouped_pc = None
        grouped_feature = None
        grouped_rgb = None

        if self.use_xyz:
            grouped_pc = torch.gather(
                pc, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, 3, -1)
            ).view(batch_size, 3, npoint, k)#torch.Size([32, 3, 140, 64])

            grouped_pc = grouped_pc - new_pc.unsqueeze(3)

        if self.use_feature:
            grouped_feature = torch.gather(
                feat, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, feat.size(1), -1)
            ).view(batch_size, feat.size(1), npoint, k)#torch.Size([32, 1, 140, 64])

            # grouped_feature = torch.cat([new_feat.unsqueeze(3), grouped_feature], -1)

        img1 = img1.view(batch_size,self.mlp[0],-1)#torch.Size([32, 32, 46208])
        grouped_rgb1 = torch.gather(
            img1, 2,
            indices_rgb.view(batch_size, 1, npoint * k).expand(-1, self.mlp[0], -1)
        ).view(batch_size, self.mlp[0], npoint, k)#torch.Size([32, 32, 140, 64])

        grouped_rgb2 = None
        if img2 is not None:
            img2 = img2.view(batch_size,self.mlp[1],-1)
            grouped_rgb2 = torch.gather(
                img2, 2,
                indices_rgb.view(batch_size, 1, npoint * k).expand(-1, self.mlp[1], -1)
            ).view(batch_size, self.mlp[1], npoint, k)  # torch.Size([32, 32, 140, 64])

        if self.use_feature and self.use_xyz:
            grouped_feature = torch.cat([grouped_pc, grouped_feature], 1)#torch.Size([32, 4, 140, 64])
        elif self.use_xyz:
            grouped_feature = grouped_pc.contiguous()

        if not self.use_xyz:
            grouped_feature = torch.cat([grouped_rgb1, grouped_rgb2],1)
            return grouped_feature

        point_feature = self.conv1(grouped_feature)#torch.Size([32, 32, 140, 64])
        #grouped_rgb = self.econv1(grouped_rgb)
        #32+32:
        feature1_fusion = torch.cat([point_feature, grouped_rgb1],1)#torch.Size([32, 64, 140, 64]

        point_feature = self.conv2(point_feature)#torch.Size([32, 64, 140, 64])
        #grouped_rgb = self.econv2(grouped_rgb)
        #64+64:
        feature2_fusion = torch.cat([point_feature, grouped_rgb2],1)#[32, 128, 140, 64])

        point_feature = self.conv3(point_feature)#torch.Size([32, 64, 140, 64])
        #grouped_rgb = self.econv3(grouped_rgb)
        point_feature = self.conv4(point_feature)#torch.Size([32, 128, 140, 64])
        #grouped_rgb = self.econv4(grouped_rgb)

        grouped_feature = torch.cat([feature1_fusion,feature2_fusion,point_feature],1)##[32, 64+128+128, 140, 64])
        # output, _ = torch.max(grouped_feature, -1)

        valid = (num > 0).view(batch_size, 1, -1, 1)#torch.Size([32, 1, 140, 1])
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
            input_channel - 3, [32, 64, 64, 128], u[0], 32, use_xyz=cfg.DATA.USE_XYZ, use_feature=True)

        self.pointnet2 = PointNetModule(
            input_channel - 3, [32, 64, 64, 128], u[1], 64, use_xyz=cfg.DATA.USE_XYZ, use_feature=True)

        self.pointnet3 = PointNetModule(
            input_channel - 3, [32, 128, 128, 256], u[2], 64, use_xyz=cfg.DATA.USE_XYZ, use_feature=True)

        self.pointnet4 = PointNetModule(
            input_channel - 3, [32, 256, 256, 512], u[3], 128, use_xyz=cfg.DATA.USE_XYZ, use_feature=True)

        self.econv1 = Conv2d(32, 32, 1)
        self.econv2 = Conv2d(32, 64, 1)
        self.econv3 = Conv2d(64, 64, 1)
        self.econv4 = Conv2d(64, 128, 1)
        self.econv5 = Conv2d(128, 256, 1)

    def forward(self, point_cloud, sample_pc, img, P, query_v1, feat=None, one_hot_vec=None):
        pc = point_cloud#torch.Size([32, 3, 1024])
        pc1 = sample_pc[0]
        pc2 = sample_pc[1]
        pc3 = sample_pc[2]
        pc4 = sample_pc[3]

        img1 = self.econv1(img)
        img2 = self.econv2(img1)
        img3 = self.econv3(img2)
        img4 = self.econv4(img3)
        img5 = self.econv5(img4)
        if cfg.DATA.BLACK_TEST:
            img1 = torch.zeros(img1.shape).cuda()
            img2 = torch.zeros(img2.shape).cuda()
            img3 = torch.zeros(img3.shape).cuda()
            img4 = torch.zeros(img4.shape).cuda()
            img5 = torch.zeros(img5.shape).cuda()

        feat1 = self.pointnet1(pc, feat, img1, img2, P, query_v1, pc1)#32*2+64*2+128
        feat1, _ = torch.max(feat1, -1)

        feat2 = self.pointnet2(pc, feat, img2, img3, P, query_v1, pc2)#32*2+64*2+128
        feat2, _ = torch.max(feat2, -1)

        feat3 = self.pointnet3(pc, feat, img3, img4, P, query_v1, pc3)#32*2+128*2+256
        feat3, _ = torch.max(feat3, -1)

        feat4 = self.pointnet4(pc, feat, img4, img5, P, query_v1, pc4)#32*2+256*2+512
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
def Conv1d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.Conv1d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm1d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.Conv1d(i_c, o_c, k, s, p), nn.ReLU(True))

def DeConv1d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.ConvTranspose1d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm1d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.ConvTranspose1d(i_c, o_c, k, s, p), nn.ReLU(True))

class ConvFeatNet(nn.Module):
    def __init__(self, i_c=320, num_vec=3):
        super(ConvFeatNet, self).__init__()




        self.block1_conv1 = Conv1d(i_c + num_vec, i_c, 3, 1, 1)

        self.block2_conv1 = Conv1d(i_c, i_c, 3, 2, 1)
        self.block2_conv2 = Conv1d(i_c, i_c, 3, 1, 1)
        self.block2_merge = Conv1d(i_c + i_c + num_vec, i_c, 1, 1)

        self.block3_conv1 = Conv1d(i_c, i_c*2, 3, 2, 1)
        self.block3_conv2 = Conv1d(i_c*2, i_c*2, 3, 1, 1)
        self.block3_merge = Conv1d(i_c*2 + i_c*2-64 + num_vec, i_c*2, 1, 1)
        self.block4_conv1 = Conv1d(i_c*2, i_c*4, 3, 2, 1)
        self.block4_conv2 = Conv1d(i_c*4, i_c*4, 3, 1, 1)
        self.block4_merge = Conv1d(i_c*4 + i_c*4-64-128 + num_vec, i_c*4, 1, 1)
        self.block2_deconv = DeConv1d(i_c, i_c*2, 1, 1, 0)
        self.block3_deconv = DeConv1d(i_c*2, i_c*2, 2, 2, 0)
        self.block4_deconv = DeConv1d(i_c*4, i_c*2, 4, 4, 0)

        if not cfg.DATA.USE_XYZ:
            i_c = 32+64
            self.block1_conv1 = Conv1d(i_c + num_vec, i_c, 3, 1, 1)

            self.block2_conv1 = Conv1d(i_c, i_c, 3, 2, 1)
            self.block2_conv2 = Conv1d(i_c, i_c, 3, 1, 1)
            self.block2_merge = Conv1d(i_c + i_c + num_vec, i_c, 1, 1)

            self.block3_conv1 = Conv1d(i_c, i_c * 2, 3, 2, 1)
            self.block3_conv2 = Conv1d(i_c * 2, i_c * 2, 3, 1, 1)
            self.block3_merge = Conv1d(i_c * 2 + i_c * 2 - 32 + num_vec, i_c * 2, 1, 1)
            self.block4_conv1 = Conv1d(i_c * 2, i_c * 4, 3, 2, 1)
            self.block4_conv2 = Conv1d(i_c * 4, i_c * 4, 3, 1, 1)
            self.block4_merge = Conv1d(i_c * 4 + i_c * 4 - 32 - 64 + num_vec, i_c * 4, 1, 1)
            self.block2_deconv = DeConv1d(i_c, i_c * 2, 1, 1, 0)
            self.block3_deconv = DeConv1d(i_c * 2, i_c * 2, 2, 2, 0)
            self.block4_deconv = DeConv1d(i_c * 4, i_c * 2, 4, 4, 0)

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
        '''
        :param x1:torch.Size([32, 323=#32*2+64*2+128+3, 280])
        :param x2:torch.Size([32, 323, 140])
        :param x3:torch.Size([32, 579=#32*2+128*2+256+3, 70])
        :param x4:torch.Size([32, 1091, 35])
        :return:x:torch.Size([32, 768, 140])
        '''
        x = self.block1_conv1(x1)#torch.Size([32, 128, 280])

        x = self.block2_conv1(x)#torch.Size([32, 128, 140])
        x = self.block2_conv2(x)#torch.Size([32, 128, 140])
        x = torch.cat([x, x2], 1)#torch.Size([32, 259, 140])
        x = self.block2_merge(x)#torch.Size([32, 128, 140])
        xx1 = x
        x = self.block3_conv1(x)#torch.Size([32, 256, 70])
        x = self.block3_conv2(x)#torch.Size([32, 256, 70])
        x = torch.cat([x, x3], 1)#torch.Size([32, 515, 70])
        x = self.block3_merge(x)#torch.Size([32, 256, 70])
        xx2 = x

        x = self.block4_conv1(x)#torch.Size([32, 512, 35])
        x = self.block4_conv2(x)#torch.Size([32, 512, 35])
        x = torch.cat([x, x4], 1)#torch.Size([32, 1027, 35])
        x = self.block4_merge(x)#torch.Size([32, 512, 35])
        xx3 = x

        xx1 = self.block2_deconv(xx1)#torch.Size([32, 256, 140])
        xx2 = self.block3_deconv(xx2)#torch.Size([32, 256, 140])
        xx3 = self.block4_deconv(xx3)#torch.Size([32, 256, 140])

        x = torch.cat([xx1, xx2[:, :, :xx1.shape[-1]], xx3[:, :, :xx1.shape[-1]]], 1)#torch.Size([32, 768, 140])

        return x

class FrustumConvNetv1(nn.Module):
    def __init__(self,n_classes=3,n_channel=4):
        super(FrustumConvNetv1, self).__init__()
        self.n_classes = n_classes
        self.feat_net = PointNetFeat(n_channel, 0)
        self.conv_net = ConvFeatNet()

        num_bins = cfg.DATA.NUM_HEADING_BIN
        self.num_bins = num_bins

        output_size = 3 + num_bins * 2 + NUM_SIZE_CLUSTER * 4
        n_feat_channel = 1920 if cfg.DATA.USE_XYZ else 576
        self.reg_out = nn.Conv1d(n_feat_channel, output_size, 1)
        self.cls_out = nn.Conv1d(n_feat_channel, 2, 1)
        nn.init.kaiming_uniform_(self.cls_out.weight, mode='fan_in')###
        nn.init.kaiming_uniform_(self.reg_out.weight, mode='fan_in')###
        self.cls_out.bias.data.zero_()###
        self.reg_out.bias.data.zero_()###

        self.cnn = ModifiedResnet()
    def _slice_output(self, output):
        '''
        :param output: torch.Size([99, 39])
        :return:
        '''

        batch_size = output.shape[0]

        num_bins = self.num_bins

        center = output[:, 0:3].contiguous()#torch.Size([99, 3])

        heading_scores = output[:, 3:3 + num_bins].contiguous()#torch.Size([99, 12])

        heading_res_norm = output[:, 3 + num_bins:3 + num_bins * 2].contiguous()#torch.Size([99, 12])

        size_scores = output[:, 3 + num_bins * 2:3 + num_bins * 2 + NUM_SIZE_CLUSTER].contiguous()#torch.Size([99, 3])

        size_res_norm = output[:, 3 + num_bins * 2 + NUM_SIZE_CLUSTER:].contiguous()#torch.Size([99, 9])
        size_res_norm = size_res_norm.view(batch_size, NUM_SIZE_CLUSTER, 3)#torch.Size([99, 3, 3])

        return center, heading_scores, heading_res_norm, size_scores, size_res_norm

    def get_center_loss(self, pred_offsets, gt_offsets):

        center_dist = torch.norm(gt_offsets - pred_offsets, 2, dim=-1)
        center_loss = huber_loss(center_dist, delta=3.0)

        return center_loss

    def get_heading_loss(self, heading_scores, heading_res_norm, heading_class_label, heading_res_norm_label):

        heading_class_loss = F.cross_entropy(heading_scores, heading_class_label)

        # b, NUM_HEADING_BIN -> b, 1
        heading_res_norm_select = torch.gather(heading_res_norm, 1, heading_class_label.view(-1, 1))

        heading_res_norm_loss = huber_loss(
            heading_res_norm_select.squeeze(1) - heading_res_norm_label, delta=1.0)

        return heading_class_loss, heading_res_norm_loss

    def get_size_loss(self, size_scores, size_res_norm, size_class_label, size_res_label_norm):
        batch_size = size_scores.shape[0]
        size_class_loss = F.cross_entropy(size_scores, size_class_label)

        # b, NUM_SIZE_CLUSTER, 3 -> b, 1, 3
        size_res_norm_select = torch.gather(size_res_norm, 1,
                                            size_class_label.view(batch_size, 1, 1).expand(
                                                batch_size, 1, 3))

        size_norm_dist = torch.norm(
            size_res_label_norm - size_res_norm_select.squeeze(1), 2, dim=-1)

        size_res_norm_loss = huber_loss(size_norm_dist, delta=1.0)

        return size_class_loss, size_res_norm_loss

    def get_corner_loss(self, preds, gts):

        center_label, heading_label, size_label = gts
        center_preds, heading_preds, size_preds = preds

        corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label)
        corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label + np.pi, size_label)

        corners_3d_pred = get_box3d_corners_helper(center_preds, heading_preds, size_preds)

        # N, 8, 3
        corners_dist = torch.min(
            torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1).mean(-1),
            torch.norm(corners_3d_pred - corners_3d_gt_flip, 2, dim=-1).mean(-1))
        # corners_dist = torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1)
        corners_loss = huber_loss(corners_dist, delta=1.0)

        return corners_loss, corners_3d_gt

    def forward(self, data_dicts):
        #dict_keys(['point_cloud', 'rot_angle', 'box3d_center', 'size_class', 'size_residual', 'angle_class', 'angle_residual', 'one_hot', 'label', 'center_ref1', 'center_ref2', 'center_ref3', 'center_ref4'])

        image = data_dicts.get('image')#torch.Size([32, 3, 400, 200])
        out_image = self.cnn(image)#torch.Size([32, 32, 400, 200])
        P = data_dicts.get('P')#torch.Size([32, 3, 4])
        query_v1 = data_dicts.get('query_v1')#torch.Size([32, 1024])

        point_cloud = data_dicts.get('point_cloud')#torch.Size([32, 4, 1024])
        one_hot = data_dicts.get('one_hot')#torch.Size([32, 3])
        ref_label = data_dicts.get('ref_label')#torch.Size([32, 140])
        bs = point_cloud.shape[0]

        # If not None, use to Compute Loss
        #seg_label = data_dicts.get('seg')#torch.Size([32, 1024])
        box3d_center_label = data_dicts.get('box3d_center')#torch.Size([32, 3])
        size_class_label = data_dicts.get('size_class')#torch.Size([32])
        #size_residual_label = data_dicts.get('size_residual')  # torch.Size([32, 3])###
        #heading_class_label = data_dicts.get('angle_class')  # torch.Size([32])###
        #heading_residual_label = data_dicts.get('angle_residual')  # torch.Size([32])###

        box3d_size_label = data_dicts.get('box3d_size')###not residual
        box3d_heading_label = data_dicts.get('box3d_heading')###not residual


        center_ref1 = data_dicts.get('center_ref1')#torch.Size([32, 3, 280])
        center_ref2 = data_dicts.get('center_ref2')#torch.Size([32, 3, 140])
        center_ref3 = data_dicts.get('center_ref3')#torch.Size([32, 3, 70])
        center_ref4 = data_dicts.get('center_ref4')#torch.Size([32, 3, 35])

        object_point_cloud_xyz = point_cloud[:,:3,:].contiguous()
        if point_cloud.shape[1] > 3:
            object_point_cloud_i = point_cloud[:,[3],:].contiguous()#torch.Size([32, 1, 1024])
        else:
            object_point_cloud_i = None

        mean_size_array = torch.from_numpy(g_mean_size_arr).type_as(point_cloud)

        feat1, feat2, feat3, feat4 = self.feat_net(
            object_point_cloud_xyz,
            [center_ref1, center_ref2, center_ref3, center_ref4],
            out_image,
            P,
            query_v1,
            object_point_cloud_i,
            one_hot)
        #feat1:torch.Size([32, 131, 280])
        #feat2:torch.Size([32, 131, 140])
        #feat3:torch.Size([32, 131, 70])
        #feat4:torch.Size([32, 131, 35])
        x = self.conv_net(feat1, feat2, feat3, feat4)##torch.Size([32, 768, 140])

        cls_scores = self.cls_out(x)#torch.Size([32, 2, 140])
        outputs = self.reg_out(x)#torch.Size([32, 39, 140])

        num_out = outputs.shape[2]
        output_size = outputs.shape[1]
        # b, c, n -> b, n, c
        cls_scores = cls_scores.permute(0, 2, 1).contiguous().view(-1, 2)#torch.Size([4480, 2])
        outputs = outputs.permute(0, 2, 1).contiguous().view(-1, output_size)#torch.Size([4480, 39])

        center_ref2 = center_ref2.permute(0, 2, 1).contiguous().view(-1, 3)#torch.Size([4480, 3])

        cls_probs = F.softmax(cls_scores, -1)#torch.Size([4480, 2])


        if box3d_center_label is None: #no label == test mode or from rgb detection -> return output
            det_outputs = self._slice_output(outputs)  # torch.Size([4480, 39])
            center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = det_outputs

            heading_probs = F.softmax(heading_scores, -1)  # torch.Size([4480, 12])
            size_probs = F.softmax(size_scores, -1)  # torch.Size([4480, 3])

            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)

            center_preds = center_boxnet + center_ref2

            heading_preds = angle_decode(heading_res_norm, heading_pred_label)
            size_preds = size_decode(size_res_norm, mean_size_array, size_pred_label)

            # corner_preds = get_box3d_corners_helper(center_preds, heading_preds, size_preds)

            cls_probs = cls_probs.view(bs, -1, 2)
            center_preds = center_preds.view(bs, -1, 3)

            size_preds = size_preds.view(bs, -1, 3)
            heading_preds = heading_preds.view(bs, -1)

            outputs = (cls_probs, center_preds, heading_preds, size_preds)

            return outputs


        fg_idx = (ref_label.view(-1) == 1).nonzero().view(-1)#torch.Size([99])

        assert fg_idx.numel() != 0

        outputs = outputs[fg_idx, :]#torch.Size([99, 39])
        center_ref2 = center_ref2[fg_idx]#torch.Size([99, 3])

        det_outputs = self._slice_output(outputs)
        center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = det_outputs
        #(99,3+12+12+3+3x3)

        heading_probs = F.softmax(heading_scores, -1)#torch.Size([99, 12])
        size_probs = F.softmax(size_scores, -1)#torch.Size([99, 3])
        # cls_loss = F.cross_entropy(cls_scores, mask_label, ignore_index=-1)
        cls_loss = softmax_focal_loss_ignore(cls_probs, ref_label.view(-1), ignore_idx=-1)

        heading_probs = F.softmax(heading_scores, -1)
        size_probs = F.softmax(size_scores, -1)
        # cls_loss = F.cross_entropy(cls_scores, mask_label, ignore_index=-1)
        cls_loss = softmax_focal_loss_ignore(cls_probs, ref_label.view(-1), ignore_idx=-1)


        # prepare label
        center_label = box3d_center_label.unsqueeze(1).expand(-1, num_out, -1)\
            .contiguous().view(-1, 3)[fg_idx]#torch.Size([99, 3])
        size_label = box3d_size_label.unsqueeze(1).expand(-1, num_out, -1)\
            .contiguous().view(-1, 3)[fg_idx]#torch.Size([99, 3])
        heading_label = box3d_heading_label.view(-1,1).expand(-1, num_out)\
            .contiguous().view(-1)[fg_idx]#torch.Size([99])
        size_class_label = size_class_label.view(-1,1).expand(-1, num_out)\
            .contiguous().view(-1)[fg_idx]#torch.Size([99])

        # encode regression targets
        center_gt_offsets = center_encode(center_label, center_ref2)#torch.Size([99, 3])
        heading_class_label, heading_res_norm_label = angle_encode(heading_label)#torch.Size([99]),torch.Size([99])
        size_res_label_norm = size_encode(size_label, mean_size_array, size_class_label)#torch.Size([99, 3])

        # loss calculation

        # center_loss
        center_loss = self.get_center_loss(center_boxnet, center_gt_offsets)

        # heading loss
        heading_class_loss, heading_res_norm_loss = self.get_heading_loss(
            heading_scores, heading_res_norm, heading_class_label, heading_res_norm_label)

        # size loss
        size_class_loss, size_res_norm_loss = self.get_size_loss(
            size_scores, size_res_norm, size_class_label, size_res_label_norm)

        # corner loss regulation
        center_preds = center_decode(center_ref2, center_boxnet)
        heading = angle_decode(heading_res_norm, heading_class_label)
        size = size_decode(size_res_norm, mean_size_array, size_class_label)

        corners_loss, corner_gts = self.get_corner_loss(
            (center_preds, heading, size),
            (center_label, heading_label, size_label)
        )

        BOX_LOSS_WEIGHT = cfg.LOSS.BOX_LOSS_WEIGHT
        CORNER_LOSS_WEIGHT = cfg.LOSS.CORNER_LOSS_WEIGHT
        HEAD_REG_WEIGHT = cfg.LOSS.HEAD_REG_WEIGHT
        SIZE_REG_WEIGHT = cfg.LOSS.SIZE_REG_WEIGHT

        # Weighted sum of all losses
        loss = cls_loss + \
            BOX_LOSS_WEIGHT * (center_loss +
                               heading_class_loss + size_class_loss +
                               HEAD_REG_WEIGHT * heading_res_norm_loss +
                               SIZE_REG_WEIGHT * size_res_norm_loss +
                               CORNER_LOSS_WEIGHT * corners_loss)

        # some metrics to monitor training status

        with torch.no_grad():

            # accuracy
            cls_prec = get_accuracy(cls_probs, ref_label.view(-1))
            heading_prec = get_accuracy(heading_probs, heading_class_label.view(-1))
            size_prec = get_accuracy(size_probs, size_class_label.view(-1))

            # iou metrics
            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)

            heading_preds = angle_decode(heading_res_norm, heading_pred_label)
            size_preds = size_decode(size_res_norm, mean_size_array, size_pred_label)

            corner_preds = get_box3d_corners_helper(center_preds, heading_preds, size_preds)
            overlap = rbbox_iou_3d_pair(corner_preds.detach().cpu().numpy(), corner_gts.detach().cpu().numpy())

            iou2ds, iou3ds = overlap[:, 0], overlap[:, 1]
            iou2d_mean = iou2ds.mean()
            iou3d_mean = iou3ds.mean()
            iou3d_gt_mean = (iou3ds >= cfg.IOU_THRESH).mean()
            iou2d_mean = torch.tensor(iou2d_mean).type_as(cls_prec)
            iou3d_mean = torch.tensor(iou3d_mean).type_as(cls_prec)
            iou3d_gt_mean = torch.tensor(iou3d_gt_mean).type_as(cls_prec)

        losses = {
            'total_loss': loss,
            'cls_loss': cls_loss,
            'center_loss': center_loss,
            'heading_class_loss': heading_class_loss,
            'heading_residual_normalized_loss': heading_res_norm_loss,
            'size_class_loss': size_class_loss,
            'size_residual_normalized_loss': size_res_norm_loss,
            'corners_loss': corners_loss
        }
        metrics = {
            'cls_acc': cls_prec,
            'head_acc': heading_prec,
            'size_acc': size_prec,
            'iou2d': iou2d_mean,
            'iou3d': iou3d_mean,
            'iou3d_' + str(cfg.IOU_THRESH): iou3d_gt_mean
        }

        return losses, metrics

if __name__ == '__main__':
    from provider_fusion import FrustumDataset
    dataset = FrustumDataset(npoints=1024, split='val',
        rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True,
        overwritten_data_path='kitti/frustum_caronly_wimage_val.pickle',
        gen_ref = True, with_image=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)
    model = FrustumConvNetv1(n_classes=2, n_channel=3).cuda()
    for batch, data_dicts in enumerate(dataloader):
        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}
        # dict_keys(['point_cloud', 'rot_angle', 'box3d_center', 'one_hot',
        # 'ref_label', 'center_ref1', 'center_ref2', 'center_ref3', 'center_ref4',
        # 'size_class', 'box3d_size', 'box3d_heading', 'image', 'P', 'query_v1'])
        losses, metrics= model(data_dicts_var)

        print()
        for key,value in losses.items():
            print(key,value)
        print()
        for key,value in metrics.items():
            print(key,value)
