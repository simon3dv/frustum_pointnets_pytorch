import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import numpy as np
import torch
import ipdb
import torch.nn as nn
import torch.nn.functional as F


# -----------------
# Global Constants
# -----------------

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3 #8 # one cluster for each type
NUM_OBJECT_POINT = 512

#g_type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3,
#              'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
g_type2class={'Car':0,'Pedestrian':1,'Cyclist':2}
g_class2type = {g_type2class[t]:t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
'''
g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                    'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Misc': np.array([3.64300781,1.54298177,1.92320313])}
'''
'''
#trainval kitti
g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127])}
'''
#train_kitti
g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127])}

g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]


def parse_output_to_tensors(box_pred, logits, mask, stage1_center):
    '''
    :param box_pred: (bs,59)
    :param logits: (bs,1024,2)
    :param mask: (bs,1024)
    :param stage1_center: (bs,3)
    :return:
        center_boxnet:(bs,3)
        heading_scores:(bs,12)
        heading_residual_normalized:(bs,12),-1 to 1
        heading_residual:(bs,12)
        size_scores:(bs,8)
        size_residual_normalized:(bs,8)
        size_residual:(bs,8)
    '''
    bs = box_pred.shape[0]
    # center
    center_boxnet = box_pred[:, :3]#0:3
    c = 3

    # heading
    heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]#3:3+12
    c += NUM_HEADING_BIN
    heading_residual_normalized = \
        box_pred[:, c:c + NUM_HEADING_BIN]#3+12 : 3+2*12
    heading_residual = \
        heading_residual_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    # size
    size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]#3+2*12 : 3+2*12+8
    c += NUM_SIZE_CLUSTER
    size_residual_normalized = \
        box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER].contiguous() #[32,24] 3+2*12+8 : 3+2*12+4*8
    size_residual_normalized = \
        size_residual_normalized.view(bs,NUM_SIZE_CLUSTER,3)#[32,8,3]
    size_residual = size_residual_normalized * \
                     torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs,1,1).cuda()
    return center_boxnet,\
            heading_scores, heading_residual_normalized, heading_residual,\
            size_scores, size_residual_normalized, size_residual

def point_cloud_masking(pts, logits, xyz_only=True):
    '''
    :param pts: bs,c,n in frustum
    :param logits: bs,n,2
    :param xyz_only: bool
    :return:
    '''
    bs = pts.shape[0]
    n_pts = pts.shape[2]
    # Binary Classification for each point
    mask = logits[:, :, 0] < logits[:, :, 1]  # (bs, n)
    mask = mask.unsqueeze(1).float()# (bs, 1, n)
    mask_count = mask.sum(2,keepdim=True).repeat(1, 3, 1)  # (bs, 3, 1)
    pts_xyz = pts[:, :3, :]  # (bs,3,n)
    mask_xyz_mean = (mask.repeat(1, 3, 1) * pts_xyz).sum(2,keepdim=True)  # (bs, 3, 1)
    mask_xyz_mean = mask_xyz_mean / torch.clamp(mask_count,min=1)  # (bs, 3, 1)
    mask = mask.squeeze()  # (bs,n)
    pts_xyz_stage1 = pts_xyz - mask_xyz_mean.repeat(1, 1, n_pts)

    if xyz_only:
        pts_stage1 = pts_xyz_stage1
    else:
        pts_features = pts[:, 3:, :]
        pts_stage1 = torch.cat([pts_xyz_stage1, pts_features], dim=-1)
    object_pts, _ = gather_object_pts(pts_stage1, mask, NUM_OBJECT_POINT)
    #(32,512,3) (32,512)
    object_pts = object_pts.reshape(bs, NUM_OBJECT_POINT, -1)
    object_pts = object_pts.float().view(bs,3,-1)
    return object_pts, mask_xyz_mean.squeeze(), mask

def gather_object_pts(pts, mask, n_pts=NUM_OBJECT_POINT):
    '''
    :param pts: (bs,c,1024)
    :param mask: (bs,1024)
    :param n_pts: max number of points of an object
    :return:
        object_pts:(bs,c,n_pts)
        indices:(bs,n_pts)
    '''
    bs = pts.shape[0]
    indices = torch.zeros((bs, n_pts), dtype=torch.int64)  # (bs, 512)
    object_pts = torch.zeros((bs, pts.shape[1], n_pts))

    for i in range(bs):
        pos_indices = torch.where(mask[i, :] > 0.5)[0]  # (653,)
        if len(pos_indices) > 0:
            if len(pos_indices) > n_pts:
                choice = np.random.choice(len(pos_indices),
                                          n_pts, replace=False)
            else:
                choice = np.random.choice(len(pos_indices),
                                          n_pts - len(pos_indices), replace=True)
                choice = np.concatenate(
                    (np.arange(len(pos_indices)), choice))
            np.random.shuffle(choice)  # (512,)
            indices[i, :] = pos_indices[choice]
            object_pts[i,:,:] = pts[i,:,indices[i,:]]
        ###else?
    return object_pts, indices


def get_box3d_corners_helper(centers, headings, sizes):
    """ Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    #print '-----', centers
    N = centers.shape[0]
    l = sizes[:,0].view(N,1)
    w = sizes[:,1].view(N,1)
    h = sizes[:,2].view(N,1)
    #print l,w,h
    x_corners = torch.cat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], dim=1) # (N,8)
    y_corners = torch.cat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], dim=1) # (N,8)
    z_corners = torch.cat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], dim=1) # (N,8)
    corners = torch.cat([x_corners.view(N,1,8), y_corners.view(N,1,8),\
                            z_corners.view(N,1,8)], dim=1) # (N,3,8)

    ###ipdb.set_trace()
    #print x_corners, y_corners, z_corners
    c = torch.cos(headings).cuda()
    s = torch.sin(headings).cuda()
    ones = torch.ones([N], dtype=torch.float32).cuda()
    zeros = torch.zeros([N], dtype=torch.float32).cuda()
    row1 = torch.stack([c,zeros,s], dim=1) # (N,3)
    row2 = torch.stack([zeros,ones,zeros], dim=1)
    row3 = torch.stack([-s,zeros,c], dim=1)
    R = torch.cat([row1.view(N,1,3), row2.view(N,1,3), \
                      row3.view(N,1,3)], axis=1) # (N,3,3)
    #print row1, row2, row3, R, N
    corners_3d = torch.bmm(R, corners) # (N,3,8)
    corners_3d +=centers.view(N,3,1).repeat(1,1,8) # (N,3,8)
    corners_3d = torch.transpose(corners_3d,1,2) # (N,8,3)
    return corners_3d

def get_box3d_corners(center, heading_residual, size_residual):
    """
    Inputs:
        center: (bs,3)
        heading_residual: (bs,NH)
        size_residual: (bs,NS,3)
    Outputs:
        box3d_corners: (bs,NH,NS,8,3) tensor
    """
    bs = center.shape[0]
    heading_bin_centers = torch.from_numpy(\
            np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN)).float() # (12,) (NH,)
    headings = heading_residual + heading_bin_centers.view(1,-1).cuda() # (bs,12)

    mean_sizes = torch.from_numpy(g_mean_size_arr).float().view(1,NUM_SIZE_CLUSTER,3).cuda()\
                 + size_residual.cuda() # (1,8,3)+(bs,8,3) = (bs,8,3)
    sizes = mean_sizes + size_residual # (bs,8,3)
    sizes = sizes.view(bs,1,NUM_SIZE_CLUSTER,3)\
                .repeat(1,NUM_HEADING_BIN,1,1).float() # (B,12,8,3)
    headings = headings.view(bs,NUM_HEADING_BIN,1).repeat(1,1,NUM_SIZE_CLUSTER) # (bs,12,8)
    centers = center.view(bs,1,1,3).repeat(1,NUM_HEADING_BIN, NUM_SIZE_CLUSTER,1) # (bs,12,8,3)
    N = bs*NUM_HEADING_BIN*NUM_SIZE_CLUSTER
    ###ipdb.set_trace()
    corners_3d = get_box3d_corners_helper(centers.view(N,3),headings.view(N),\
                                    sizes.view(N,3))
    ###ipdb.set_trace()
    return corners_3d.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3)#[32, 12, 8, 8, 3]

def huber_loss(error, delta=1.0):#(32,), ()
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)

class FrustumPointNetLoss(nn.Module):
    def __init__(self):
        super(FrustumPointNetLoss, self).__init__()

    def forward(self, logits, mask_label, \
                center, center_label, stage1_center, \
                heading_scores, heading_residual_normalized, heading_residual, \
                heading_class_label, heading_residual_label, \
                size_scores,size_residual_normalized,size_residual,
                size_class_label,size_residual_label,
                corner_loss_weight=10.0,box_loss_weight=1.0):
        '''
        1.InsSeg
        logits: torch.Size([32, 1024, 2]) torch.float32
        mask_label: [32, 1024]
        2.Center
        center: torch.Size([32, 3]) torch.float32
        stage1_center: torch.Size([32, 3]) torch.float32
        center_label:[32,3]
        3.Heading
        heading_scores: torch.Size([32, 12]) torch.float32
        heading_residual_snormalized: torch.Size([32, 12]) torch.float32
        heading_residual: torch.Size([32, 12]) torch.float32
        heading_class_label:(32)
        heading_residual_label:(32)
        4.Size
        size_scores: torch.Size([32, 8]) torch.float32
        size_residual_normalized: torch.Size([32, 8, 3]) torch.float32
        size_residual: torch.Size([32, 8, 3]) torch.float32
        size_class_label:(32)
        size_residual_label:(32,3)
        5.Corner
        6.Weight
        corner_loss_weight: float scalar
        box_loss_weight: float scalar

        '''
        bs = logits.shape[0]
        # 3D Instance Segmentation PointNet Loss
        logits = F.log_softmax(logits.view(-1,2),dim=1)#torch.Size([32768, 2])
        mask_label = mask_label.view(-1).long()#torch.Size([32768])
        mask_loss = F.nll_loss(logits, mask_label)#tensor(0.6361, grad_fn=<NllLossBackward>)

        # Center Regression Loss
        center_dist = torch.norm(center-center_label,dim=1)#(32,)
        center_loss = huber_loss(center_dist, delta=2.0)

        stage1_center_dist = torch.norm(center-stage1_center,dim=1)#(32,)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

        # Heading Loss
        heading_class_loss = F.nll_loss(F.log_softmax(heading_scores,dim=1), \
                                        heading_class_label.long())#tensor(2.4505, grad_fn=<NllLossBackward>)
        hcls_onehot = torch.eye(NUM_HEADING_BIN)[heading_class_label.long()].cuda()  # 32,12
        heading_residual_normalized_label = \
            heading_residual_label / (np.pi / NUM_HEADING_BIN)  # 32,
        heading_residual_normalized_dist = torch.sum( \
            heading_residual_normalized * hcls_onehot.float(), dim=1)  # 32,
        ### Only compute reg loss on gt label
        heading_residual_normalized_loss = \
            huber_loss(heading_residual_normalized_dist -
                       heading_residual_normalized_label, delta=1.0)###fix,2020.1.14
        # Size loss
        size_class_loss = F.nll_loss(F.log_softmax(size_scores,dim=1),\
                    size_class_label.long())#tensor(2.0240, grad_fn=<NllLossBackward>)

        scls_onehot = torch.eye(NUM_SIZE_CLUSTER)[size_class_label.long()].cuda()  # 32,8
        scls_onehot_repeat = scls_onehot.view(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3)  # 32,8,3
        predicted_size_residual_normalized_dist = torch.sum( \
            size_residual_normalized * scls_onehot_repeat.cuda(), dim=1)#32,3
        mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).float().cuda() \
            .view(1, NUM_SIZE_CLUSTER, 3)  # 1,8,3
        mean_size_label = torch.sum(scls_onehot_repeat * mean_size_arr_expand, dim=1)# 32,3
        size_residual_label_normalized = size_residual_label / mean_size_label.cuda()

        size_normalized_dist = torch.norm(size_residual_label_normalized-\
                    predicted_size_residual_normalized_dist,dim=1)#32
        size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)#tensor(11.2784, grad_fn=<MeanBackward0>)

        # Corner Loss
        corners_3d = get_box3d_corners(center,\
                    heading_residual,size_residual).cuda()#(bs,NH,NS,8,3)(32, 12, 8, 8, 3)
        gt_mask = hcls_onehot.view(bs,NUM_HEADING_BIN,1).repeat(1,1,NUM_SIZE_CLUSTER) * \
                  scls_onehot.view(bs,1,NUM_SIZE_CLUSTER).repeat(1,NUM_HEADING_BIN,1)# (bs,NH=12,NS=8)
        corners_3d_pred = torch.sum(\
            gt_mask.view(bs,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,1,1)\
            .float().cuda() * corners_3d,\
            dim=[1, 2]) # (bs,8,3)
        heading_bin_centers = torch.from_numpy(\
            np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float().cuda()  # (NH,)
        heading_label = heading_residual_label.view(bs,1) + \
                        heading_bin_centers.view(1,NUM_HEADING_BIN)  #(bs,1)+(1,NH)=(bs,NH)

        heading_label = torch.sum(hcls_onehot.float() * heading_label, 1)
        mean_sizes = torch.from_numpy(g_mean_size_arr)\
                    .float().view(1,NUM_SIZE_CLUSTER,3).cuda()#(1,NS,3)
        size_label = mean_sizes + \
                     size_residual_label.view(bs,1,3) #(1,NS,3)+(bs,1,3)=(bs,NS,3)
        size_label = torch.sum(\
           scls_onehot.view(bs,NUM_SIZE_CLUSTER,1).float() * size_label, axis=[1])  # (B,3)

        corners_3d_gt = get_box3d_corners_helper( \
            center_label, heading_label, size_label)  # (B,8,3)
        corners_3d_gt_flip = get_box3d_corners_helper( \
            center_label, heading_label + np.pi, size_label)  # (B,8,3)

        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                  torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
        corners_loss = huber_loss(corners_dist, delta=1.0)

        # Weighted sum of all losses
        total_loss = mask_loss + box_loss_weight * (center_loss + \
                    heading_class_loss + size_class_loss + \
                    heading_residual_normalized_loss * 20 + \
                    size_residual_normalized_loss * 20 + \
                    stage1_center_loss + \
                    corner_loss_weight * corners_loss)
        ###total_loss = mask_loss
        #tensor(306.7591, grad_fn=<AddBackward0>)
        ###if np.isnan(total_loss.item()) or total_loss > 10000.0:
        ###    ipdb.set_trace()
        losses = {
            'total_loss': total_loss,
            'mask_loss': mask_loss,
            'mask_loss': box_loss_weight * center_loss,
            'heading_class_loss': box_loss_weight * heading_class_loss,
            'size_class_loss': box_loss_weight * size_class_loss,
            'heading_residual_normalized_loss': box_loss_weight * heading_residual_normalized_loss * 20,
            'size_residual_normalized_loss': box_loss_weight * size_residual_normalized_loss * 20,
            'stage1_center_loss': box_loss_weight * size_residual_normalized_loss * 20,
            'corners_loss': box_loss_weight * corners_loss * corner_loss_weight,
        }
        return losses
        '''
        return total_loss, mask_loss, center_loss, heading_class_loss, \
            size_class_loss, heading_residual_normalized_loss, \
            size_residual_normalized_loss, stage1_center_loss, \
            corners_loss
        '''
