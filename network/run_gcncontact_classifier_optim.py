# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import openmesh as om
import trimesh

import networkx as nx

DEEPCONTACT_BIN_WEIGHTS_FILE = '/scratch/minjay/coma_taxonomy_prediction/data/class_bin_weights.out'
DEEPCONTACT_NUM_BINS = 10

TAXONOMY_CONDITION = False
LATENT_CONDITION = False
GAN_HAND = False

CLASSIFIER_PRETRAINED = False
TRAIN2STEP = '2step' #2step # single # test
CE_STEP = 10

START_epoch11 = False
STEPAB = False

#EXP_NAME = 'NIA_AUG_TRAINING_taxonomy_condition_encoder_joint_fix'
EXP_NAME = 'Final2Step_0.2_reweight_TT_10'

TAX_WEIGHT_AMP = 0.2

#NonPreSingleJoCoR_0.0
#NonPreSingleJoCoR_0.2
#Pre2StepCE_0.0
#PreSingleCE_0.0
#Pre2StepJoCoR_0.0
#Pre2StepJoCoR_0.2
#PreSingleJoCoR_0.0
#PreSingleJoCoR_0.2

# NonPreSingleJoCoR_0.2

# 'NIA_AUG_TRANS_0.01_ROT_0.1_incep_classifier_0.3_rate_2step_clamp_fix_remove_recon_onlyCE'
# 'NIA_AUG_TRANS_0.01_ROT_0.1_incep_classifier_0.3_rate_2step'
#'NIA_AUG_TRAINING_noise30_reweight_mano_param'
#'NIA_AUG_TRAINING_noise30_only_reweight'
#EXP_NAME = 'NIA_TRAINING_noise00'
#EXP_NAME = 'NIA_AUG_TRAINING_noise30'
#EXP_NAME = 'NIA_TRAINING_noise30'

MODEL_LOAD_PATH = '/scratch/minjay/s2contact/NIA_AUG_TRAINING_non_taxonomy_joint_fix/checkpoint_035.pt'

print(MODEL_LOAD_PATH)

from scipy.sparse import csgraph

#'/scratch/minjay/s2contact/NIA_AUG_TRAINING_noise00_reweight_mano_param/checkpoint_010.pt'
#'/scratch/minjay/s2contact/NIA_AUG_TRAINING_noise30_reweight_mano_param/checkpoint_010.pt'
#'/scratch/minjay/s2contact/NIA_TRAINING_noise00/checkpoint_025.pt'
#'/scratch/minjay/s2contact/NIA_TRAINING_noise30/checkpoint_025.pt'

# '/scratch/minjay/s2contact/NIA_AUG_TRAINING_non_taxonomy_joint_fix/checkpoint_035.pt'
# '/scratch/minjay/s2contact/NIA_AUG_TRAINING_taxonomy_condition_encoder_joint_fix/checkpoint_035.pt'
#'/scratch/minjay/s2contact/NIA_AUG_TRAINING_taxonomy_condition_joint_fix/checkpoint_035.pt'

IMAGE_SAVE_PATH = 'comparem3/'

print(EXP_NAME)
# 'GANHAND_AUG_TRAINING_taxonomy_condition_encoder_joint_fix'
# 'GANHAND_AUG_TRAINING_non_taxonomy'
# 'GANHAND_AUG_TRAINING_taxonomy_condition_encoder'
# 'NIA_AUG_TRAINING_taxonomy_condition_encoder_surfacecut'
# 'GANHAND_AUG_TRAINING_taxonomy_condition_encoder

COMA_PATH = '/scratch/minjay/coma_taxonomy_prediction/out/corp_learning_in_ch_4_latent_128_noise_30/checkpoints/checkpoint_120.pt'
#COMA_PATH = '/scratch/minjay/coma_taxonomy_prediction/out/corp_learning_in_ch_4_forget20/checkpoints/checkpoint_120.pt'
#COMA_PATH = '/scratch/minjay/coma_taxonomy_prediction/out/corp_learning_in_ch_4_forget00/checkpoints/checkpoint_120.pt'
#WEIGHT_PATH = '/scratch/minjay/s2contact/train_00_error_weight.pickle'
WEIGHT_PATH = '/scratch/minjay/s2contact/train_30_error_weight.pickle'



SURFACE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 57, 58, 60, 61, 62, 63, 64, 
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 93, 94, 95, 96, 97, 98, 99, 100, 101, 
            102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 
            122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 137, 138, 139, 140, 141, 
            142, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 159, 161, 162, 164, 165, 166, 
            167, 168, 169, 170, 171, 172, 173, 174, 177, 185, 188, 189, 194, 195, 196, 197, 198, 199, 
            221, 222, 223, 224, 225, 228, 237, 238, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 
            254, 255, 257, 258, 262, 263, 264, 265, 266, 267, 268, 271, 273, 275, 277, 278, 280, 281, 
            284, 285, 288, 293, 297, 298, 299, 300, 301, 302, 304, 309, 317, 320, 321, 322, 323, 324, 
            325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 340, 341, 342, 343, 
            344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 361, 362, 
            367, 368, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 384, 385, 386, 387, 
            390, 391, 392, 393, 394, 396, 397, 398, 401, 402, 403, 409, 410, 411, 412, 413, 414, 431, 
            432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 444, 448, 449, 452, 453, 454, 455, 
            456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 473, 474, 
            479, 480, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 494, 495, 496, 497, 500, 501, 
            502, 503, 504, 506, 507, 508, 510, 511, 512, 513, 514, 520, 521, 522, 523, 524, 525, 535, 
            536, 537, 539, 540, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 
            556, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 
            575, 576, 577, 578, 579, 580, 581, 582, 583, 585, 586, 591, 592, 594, 595, 596, 597, 598, 
            599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 612, 613, 614, 615, 618, 619, 620, 
            621, 622, 624, 625, 626, 629, 630, 631, 637, 638, 639, 640, 641, 642, 643, 656, 657, 659, 
            660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 
            678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 
            696, 697, 698, 699, 700, 701, 703, 704, 710, 711, 712, 713, 714, 715, 717, 718, 730, 732, 
            733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 748, 749, 750, 751, 
            753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 
            771, 772, 773, 774, 775, 776, 777]

NON_SURFACE = list(set([i for i in range(778)]) - set(SURFACE))

from network.loader import ContactDBDataset
from network.deepcontact_net import DeepContactNet
import glob
import argparse
from network.optimize_pose import optimize_pose
from network.visualize import show_optimization
import pickle
from network.hand_object import HandObject
import network.util as util
from tqdm import tqdm
import network.arguments as arguments
import time
import torch
import os
from torch.utils.data import Dataset, DataLoader
import pytorch3d
import numpy as np
import inspect
import torch.nn.functional as F

sys.path.append('/scratch/minjay/ContactOpt')

from manopth.manolayer import ManoLayer

import contactopt.util as util
from contactopt.util import SAMPLE_VERTS_NUM
from contactopt.hand_object import HandObject
import multiprocessing
import threading
import glob
import open3d as o3d
import os

import numpy as np
import pickle
import open3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import random
import sys
from open3d import visualization as o3dv
from manopth import rodrigues_layer
from manopth.manolayer import ManoLayer
from manopth.tensutils import th_posemap_axisang, make_list, th_pack, subtract_flat_id, th_with_zeros
import torch
from torch.utils.tensorboard import SummaryWriter

from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import visualization as o3dv
from pytorch3d.structures import Meshes
import torch.nn as nn

sys.path.append('/scratch/minjay/coma_taxonomy_prediction')

from models import AE
from psbody.mesh import Mesh
from utils_coma import train_eval, mesh_sampling
import os.path as osp

from pytorch3d.structures import Meshes, Pointclouds

COMMON_KEY = [1, 2, 3, 4, 7, 9, 11, 12, 14, 17, 18, 19, 20, 22, 26, 27, 28, 29, 31, 33]

if GAN_HAND :
    TAX_DICT =   {1:1,2:2,3:3,4:4,6:5,7:6,8:7,9:8,11:9,12:10,14:11,15:12,17:13,18:14,19:15,20:16,22:17,26:18,27:19,28:20,29:21,31:22,33:23}

else :
    TAX_DICT   = {1:1,2:2,3:3,4:4,5:5,7:6,9:7,10:8,11:9,12:10,13:11,14:12,16:13,17:14,18:15,19:16,20:17,22:18,23:19,24:20,25:21,26:22,27:23,28:24,29:25,30:26,31:27,33:28}
    TAX_DICT_R = {1:1,2:2,3:3,4:4,5:5,6:7,7:9,8:10,9:11,10:12,11:13,12:14,13:16,14:17,15:18,16:19,17:20,18:22,19:23,20:24,21:25,22:26,23:27,24:28,25:29,26:30,27:31,28:33}

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3


from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pytorch3d import _C

class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):

        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


point_face_distance = _PointFaceDistance.apply

def point_mesh_face_distance(
    meshes: Meshes,
    pcls: Pointclouds,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area)

    return point_to_face

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduction = 'none')

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def loss_select(y_1, y_2, t, forget_rate, co_lambda=0.8):

    # (logits1, logits2, labels, self.rate_schedule[epoch], ind, self.co_lambda)

    loss_pick_1 = F.cross_entropy(y_1, t, reduction = 'none') 
    loss_pick_2 = F.cross_entropy(y_2, t, reduction = 'none') 
    
    loss_pick = loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2, reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False)
    
    if torch.isnan(loss_pick).any() :
        import pdb; pdb.set_trace()

    #sorted index
    ind_sorted = torch.argsort(loss_pick)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    # exchange
    loss = torch.mean( torch.clamp(loss_pick[ind_update], max=20.0 ))

    return loss, ind_update

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]

def export_obj(vertices, faces, file_name):
    """
    Export a list of vertices and faces to a Wavefront OBJ file.

    Parameters:
    - vertices: List of tuples representing the vertices (x, y, z).
    - faces: List of tuples representing the vertex indices of each face.
              Indices are 1-based (following the OBJ format).
    - file_name: String representing the name of the file to save.
    """
    with open(file_name, 'w') as file:
        # Write vertices
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write faces
        for face in faces:
            face_str = ' '.join(str(index) for index in face)
            file.write(f"f {face_str}\n")


def val_to_class_0_9(val):

    """

    Converts a contact value [0-1] to a class assignment

    :param val: tensor (batch, verts)

    :return: class assignment (batch, verts)

    """

    expanded = torch.floor( (val) * DEEPCONTACT_NUM_BINS)

    return torch.clamp(expanded, min=0, max=DEEPCONTACT_NUM_BINS - 1).long() # Cut off potential 1.0 inputs?

def class_to_val(raw_scores):

    """

    Finds the highest softmax for each class

    :param raw_scores: tensor (batch, verts, classes)

    :return: highest class (batch, verts)

    """

    cls = torch.argmax(raw_scores, dim=2)

    val = (cls + 0.5) / DEEPCONTACT_NUM_BINS

    return val

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10, power=0.75, init_lr=0.001,weight_decay=0.0005, max_iter=10000):
    #10000
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    #max_iter = 10000
    gamma = 10.0
    lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return optimizer

def vis_contactMap(gt_ho):

    hand, obj = gt_ho.get_o3d_meshes(hand_contact=True, obj_contact=True, normalize_pos=True)

    geom_list = [hand,obj]

    o3dv.draw_geometries(geom_list,window_name=str('contactmap'))

def vis_contactMap_two(gt_ho,gt_pca):

    hand, obj = gt_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)

    hand2, _ = gt_pca.get_o3d_meshes(hand_contact=True, normalize_pos=True,hot=True)

    geom_list = [hand,hand2,obj]

    o3dv.draw_geometries(geom_list,window_name=str('contactmap'))

def vis_contactMap_nc(gt_ho,gt_pca):

    hand = gt_ho.get_o3d_meshes_hand_only(hand_contact=False, normalize_pos=False,hot=True)

    hand2 = gt_pca.get_o3d_meshes_hand_only(hand_contact=False, normalize_pos=False)

    geom_list = [hand,hand2]

    o3dv.draw_geometries(geom_list,window_name=str('contactmap'))

def vis_contactMap_nc_single(gt_ho):

    hand = gt_ho.get_o3d_meshes_hand_only(hand_contact=False, normalize_pos=False,hot=True)

    geom_list = [hand]

    o3dv.draw_geometries(geom_list,window_name=str('contactmap'))

def vis_contactMap_hand(gt_ho):

    hand = gt_ho.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

    # hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
    geom_list = [hand]

    o3dv.draw_geometries(geom_list,window_name=str('contactmap'))

def vis_contactMap_surface(gt_ho,lst):

    hand = gt_ho.get_o3d_meshes_surface(lst,hand_contact=True, normalize_pos=False,hot=True)

    geom_list = [hand]

    o3dv.draw_geometries(geom_list,window_name=str('contactmap'))

def mesh_gaussian_filter(contact_map, adjacency_matrix, sigma):
    laplacian = csgraph.laplacian(adjacency_matrix, normed=True)
    identity = np.eye(len(adjacency_matrix))
    diffusion = np.linalg.inv(identity + sigma * laplacian)
    return diffusion @ contact_map @ diffusion.T

class DictDataset(Dataset):
    def __init__(self, pickle_path):
        
        if os.path.isfile(pickle_path) :

            with open(pickle_path, 'rb') as handle :
                self.dict = pickle.load(handle)
            
        else :
            
            print("ERROR Loading GanHand")

            exit(0)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, index):

        # dict_keys(['taxonomy', 'aug_pose', 'aug_rot', 'aug_trans', 'hand_beta', 
        # 'obj_sampled_idx', 'hand_feats', 'obj_feats', 'hand_verts', 'obj_verts', 
        # 'hand_contact', 'obj_contact', 'hand_contact_aug', 'obj_normals_aug', 'object_full_face', 'object_full_verts'])

        sample = self.dict[index]

        if TAXONOMY_CONDITION :
        # Number of classes (adjust this based on your specific case)
            if GAN_HAND :
                num_classes = 23
            else :
                num_classes = 28

            # Generate one-hot encoding
            import torch.nn.functional as F

            device = sample['hand_feats'].device

            hand_feats_aug = sample['hand_feats']
            obj_feats_aug  = sample['obj_feats']

            one_hot_encoding = F.one_hot(torch.tensor([sample['taxonomy']]), num_classes=num_classes)
            one_hot_encoding_hand = one_hot_encoding.repeat(778,1).to(device)
            one_hot_encoding_obj = one_hot_encoding.repeat(2048,1).to(device)

            hand_feats_aug_new = torch.cat((hand_feats_aug, one_hot_encoding_hand), dim=1)
            obj_feats_aug_new = torch.cat((obj_feats_aug, one_hot_encoding_obj), dim=1)

            sample['hand_feats']   = hand_feats_aug_new
            sample['obj_feats']    = obj_feats_aug_new
            sample['idx'] = index

        return sample

        # Get data by key
        #data_point = self.data_dict[self.keys[index]]
        #return data_point

    @staticmethod
    def collate_fn(batch):
        sample = dict()
        batch_keys = batch[0].keys()
        
        skip_keys = ['obj_normals_aug','object_full_face','object_full_verts']    # These will be manually collated

        # For each not in skip_keys, use default torch collator
        for key in [k for k in batch_keys if k not in skip_keys]:
            sample[key] = torch.utils.data._utils.collate.default_collate([d[key] for d in batch])

        obj_normals_aug_all = [sample['obj_normals_aug'] for sample in batch]
        object_full_face_all = [sample['object_full_face'] for sample in batch]
        object_full_verts_all = [sample['object_full_verts'] for sample in batch]

        sample['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded(obj_normals_aug_all, pad_value=-1)
        sample['object_full_face'] = pytorch3d.structures.utils.list_to_padded(object_full_face_all, pad_value=-1)
        sample['object_full_verts'] = pytorch3d.structures.utils.list_to_padded(object_full_verts_all, pad_value=-1)

        return sample

class DictDataset_PLUG_COMA(Dataset):
    def __init__(self, pickle_path):
        
        if os.path.isfile(pickle_path) :

            with open(pickle_path, 'rb') as handle :
                self.dict = pickle.load(handle)
            
        else :
            
            print("ERROR Loading GanHand")

            exit(0)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, index):

        # dict_keys(['taxonomy', 'aug_pose', 'aug_rot', 'aug_trans', 'hand_beta', 
        # 'obj_sampled_idx', 'hand_feats', 'obj_feats', 'hand_verts', 'obj_verts', 
        # 'hand_contact', 'obj_contact', 'hand_contact_aug', 'obj_normals_aug', 'object_full_face', 'object_full_verts'])

        sample = self.dict[index]

        device = sample['hand_feats'].device

        if TAXONOMY_CONDITION :
        # Number of classes (adjust this based on your specific case)
            if GAN_HAND :
                num_classes = 23
            else :
                num_classes = 28

            # Generate one-hot encoding
            import torch.nn.functional as F

            hand_feats_aug = sample['hand_feats']
            obj_feats_aug  = sample['obj_feats']

            one_hot_encoding = F.one_hot(torch.tensor([sample['taxonomy']]), num_classes=num_classes)
            one_hot_encoding_hand = one_hot_encoding.repeat(778,1).to(device)
            one_hot_encoding_obj = one_hot_encoding.repeat(2048,1).to(device)

            hand_feats_aug_new = torch.cat((hand_feats_aug, one_hot_encoding_hand), dim=1)
            obj_feats_aug_new = torch.cat((obj_feats_aug, one_hot_encoding_obj), dim=1)

            sample['hand_feats']   = hand_feats_aug_new
            sample['obj_feats']    = obj_feats_aug_new
        
        sample['index'] = index
        mano_verts = sample['hand_origin_verts']

        x_coords = mano_verts[:, 0]
        y_coords = mano_verts[:, 1]
        z_coords = mano_verts[:, 2]

        x_min, x_max = torch.min(x_coords), torch.max(x_coords)
        y_min, y_max = torch.min(y_coords), torch.max(y_coords)
        z_min, z_max = torch.min(z_coords), torch.max(z_coords)

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2

        mano_verts = mano_verts - torch.Tensor([x_center, y_center, z_center]).to(device)

        sample['mano_verts'] = mano_verts

        
        return sample

        # Get data by key
        #data_point = self.data_dict[self.keys[index]]
        #return data_point

    @staticmethod
    def collate_fn(batch):
        sample = dict()
        batch_keys = batch[0].keys()
        
        skip_keys = ['obj_normals_aug','object_full_face','object_full_verts']    # These will be manually collated

        # For each not in skip_keys, use default torch collator
        for key in [k for k in batch_keys if k not in skip_keys]:
            sample[key] = torch.utils.data._utils.collate.default_collate([d[key] for d in batch])

        obj_normals_aug_all = [sample['obj_normals_aug'] for sample in batch]
        object_full_face_all = [sample['object_full_face'] for sample in batch]
        object_full_verts_all = [sample['object_full_verts'] for sample in batch]

        sample['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded(obj_normals_aug_all, pad_value=-1)
        sample['object_full_face'] = pytorch3d.structures.utils.list_to_padded(object_full_face_all, pad_value=-1)
        sample['object_full_verts'] = pytorch3d.structures.utils.list_to_padded(object_full_verts_all, pad_value=-1)

        return sample

class GANHAND(Dataset):
    def __init__(self,setup):

        self.dataset_pkl_name = setup + '.pkl'
        
        self.dict = {}

        self.setup = setup

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mano_layer = ManoLayer(side='right', mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=False, flat_hand_mean=True,
                            center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(self.device)

        self.mano_faces = self.mano_layer.th_faces
        
        self.ncomps = 45

        self.mano_run = ManoLayer(mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=True, ncomps=self.ncomps, side='right', flat_hand_mean=True, center_idx=0).to(self.device)

        self.handoripath = f'/scratch/minjay/GANHAND_EXTRACT_FOR_TRAINING/{self.setup}_hand_ori.pickle'

        self.obj_face_pkl = f'/scratch/minjay/GANHAND_EXTRACT_FOR_TRAINING/obj_face.pickle'

        self.hoi_dict_path = f'/scratch/minjay/GANHAND_EXTRACT_FOR_TRAINING/{self.setup}_hand_info_.pickle'


        if os.path.isfile(self.dataset_pkl_name) :

            with open(self.obj_face_pkl, 'rb') as handle:
                self.obj_faces_dict = pickle.load(handle)
            
            print('obj face dict loaded')

            with open(self.hoi_dict_path, 'rb') as handle:
                self.dict = pickle.load(handle)

            print('dataset dict loaded')

            with open(self.handoripath, 'rb') as handle:
                self.handori_dict = pickle.load(handle)

            print('hand param loaded')

        else :
            
            print("ERROR Loading GanHand")

            exit(0)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):

        sample = {}

        ho_info    = self.dict[idx]
        
        hand_dict  = self.handori_dict[idx]

        obj_faces  = self.obj_faces_dict[int(ho_info['obj_ids'])]

        ho_gt = HandObject()
        ho_gt.hand_verts   = ho_info['hand_verts'].to(self.device) #* 1000
        ho_gt.closed_faces = self.mano_faces
        ho_gt.obj_verts    = ho_info['obj_verts'].to(self.device)
        ho_gt.obj_faces    = obj_faces.to(self.device)

        sample['object_full_face'] = ho_gt.obj_faces 
        sample['object_full_verts'] = ho_gt.obj_verts
        #ho_gt.hand_joints  = ho_info['hand_joints'].to(self.device)

        gt_contact_hand = ho_info['hand_contact']
    
        ho_gt.obj_contact  = ho_info['obj_contact']
        ho_gt.hand_contact  = gt_contact_hand

        ho_gt.hand_beta = hand_dict['hand_beta']
        ho_gt.hand_pose = hand_dict['hand_pose'][:,:self.ncomps]
        sample['taxonomy']     = TAX_DICT[int(ho_info['taxonomy'])] - 1

        ho_gt.hand_rot  = hand_dict['hand_mTc'] # Rotation

        aug_trans=0.05
        aug_rot=0.1 #.1
        aug_pca=0.3 #.2 # 3

        aug_t = np.random.randn(3) * aug_trans
        aug_p = np.concatenate((np.random.randn(3) * aug_rot, np.random.randn(self.ncomps) * aug_pca)).astype(np.float32)

        ho_gt.calc_dist_contact(hand=True, obj=True)
        
        ho_gt.hand_contact[NON_SURFACE] = 0.05
        #ho_aug.calc_dist_contact(hand=True, obj=True)

        ho_aug = HandObject()

        ho_aug.load_from_ho_new_trans(ho_gt, self.mano_run, hand_dict['trans'], self.device, aug_p, aug_t)
        
        ######## ---- ########
        sample['aug_pose']  = ho_aug.hand_pose
        sample['aug_rot']   = ho_aug.hand_rot
        sample['aug_trans'] = ho_aug.hand_trans.float()
        ######## ---- ########

        ho_aug.closed_faces = self.mano_faces

        obj_sampled_idx = np.random.randint(0, len(ho_gt.obj_verts), 2048)

        ######## ---- ########
        sample['obj_sampled_idx'] = obj_sampled_idx
        ######## ---- ########

        hand_feats_aug, obj_feats_aug = ho_aug.generate_pointnet_features(obj_sampled_idx,self.device)

        #print(sample['taxonomy'])
        #vis_contactMap(ho_gt)

        ######## ---- ########
        sample['obj_normals_aug'] = ho_aug.obj_normals
        ######## ---- ########

        if TAXONOMY_CONDITION :
        # Number of classes (adjust this based on your specific case)
            if GAN_HAND :
                num_classes = 23
            else :
                num_classes = 28

            # Generate one-hot encoding
            import torch.nn.functional as F

            one_hot_encoding = F.one_hot(torch.tensor([sample['taxonomy']]), num_classes=num_classes)
            one_hot_encoding_hand = one_hot_encoding.repeat(778,1).to(self.device)
            one_hot_encoding_obj = one_hot_encoding.repeat(2048,1).to(self.device)

            hand_feats_aug_new = torch.cat((hand_feats_aug, one_hot_encoding_hand), dim=1)
            obj_feats_aug_new = torch.cat((obj_feats_aug, one_hot_encoding_obj), dim=1)

            sample['hand_feats']   = hand_feats_aug_new
            sample['obj_feats']    = obj_feats_aug_new

        else :            
            sample['hand_feats']   = hand_feats_aug
            sample['obj_feats']    = obj_feats_aug


        obj_verts = ho_aug.obj_verts[obj_sampled_idx, :] 
        obj_contact = ho_gt.obj_contact[obj_sampled_idx, :] 

        sample['hand_verts']   = ho_aug.hand_verts
        sample['obj_verts']    = obj_verts

        #sample['hand_feats']   = hand_feats_aug
        #sample['obj_feats']    = obj_feats_aug
        
        sample['hand_contact'] = ho_gt.hand_contact
        sample['obj_contact']  = obj_contact

        ho_aug.calc_dist_contact(hand=True, obj=False)

        sample['hand_contact_aug'] = ho_aug.hand_contact

        return sample  # Dummy label, replace as needed

    @staticmethod
    def collate_fn(batch):
        sample = dict()
        batch_keys = batch[0].keys()
        
        skip_keys = ['obj_normals_aug','object_full_face','object_full_verts']    # These will be manually collated

        # For each not in skip_keys, use default torch collator
        for key in [k for k in batch_keys if k not in skip_keys]:
            sample[key] = torch.utils.data._utils.collate.default_collate([d[key] for d in batch])

        obj_normals_aug_all = [sample['obj_normals_aug'] for sample in batch]
        object_full_face_all = [sample['object_full_face'] for sample in batch]
        object_full_verts_all = [sample['object_full_verts'] for sample in batch]

        sample['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded(obj_normals_aug_all, pad_value=-1)
        sample['object_full_face'] = pytorch3d.structures.utils.list_to_padded(object_full_face_all, pad_value=-1)
        sample['object_full_verts'] = pytorch3d.structures.utils.list_to_padded(object_full_verts_all, pad_value=-1)

        return sample


class CustomDataset(Dataset):
    def __init__(self,dirpath,setup):

        #self.dataset_pkl_name = setup + '1000.pkl'
        self.dataset_pkl_name = setup + '.pkl'
        
        self.dict = {}

        self.directory_path = os.path.join(dirpath,setup)

        self.obj_face_pkl =  os.path.join(dirpath,'obj_face_dict.pickle')

        self.setup = setup

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mano_layer = ManoLayer(side='right', mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=False, flat_hand_mean=True,
                            center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(self.device)

        self.mano_faces = self.mano_layer.th_faces

        hands_components = self.mano_layer.th_selected_comps
        
        self.ncomps = 45

        #selected_components = hands_components[:ncomps]
        self.pca_inv = torch.Tensor(torch.inverse(hands_components)).cpu()

        #smpl_data['hands_man']
        self.mano_run = ManoLayer(mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=True, ncomps=self.ncomps, side='right', flat_hand_mean=True, center_idx=0).to(self.device)
        # mano_model = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=15, side='right', flat_hand_mean=False)

        with open(self.obj_face_pkl, 'rb') as handle:
                obj_face_dict = pickle.load(handle)

        self.obj_faces_dict = obj_face_dict

        self.taxonomyfixpath = f'/scratch/minjay/NIA_EXTRACT_FOR_TRAINING/{self.setup}_taxonomy_.pickle'

        self.filterpath = f'/scratch/minjay/NIA_EXTRACT_FOR_TRAINING/{self.setup}_fileavail_.pickle'

        self.handoripath = f'/scratch/minjay/NIA_EXTRACT_FOR_TRAINING/{self.setup}_hand_info_.pickle'

        self.surfacefilterpath = f'/scratch/minjay/NIA_EXTRACT_FOR_TRAINING/{self.setup}_hand_surface_filter_.pickle'

        with open(WEIGHT_PATH, 'rb') as handle:
                self.weight_dict = pickle.load(handle)

        with open(self.filterpath, 'rb') as handle:
                self.file_names = pickle.load(handle)

        if os.path.isfile(self.dataset_pkl_name) :
            
            with open(self.surfacefilterpath, 'rb') as handle:
                self.file_names = pickle.load(handle)

            with open(self.taxonomyfixpath, 'rb') as handle:
                self.taxonomyfixed = pickle.load(handle)

            print('taxonomy loaded')

            with open(self.dataset_pkl_name, 'rb') as handle:
                self.dict = pickle.load(handle)

            print('dataset dict loaded')

            with open(self.handoripath, 'rb') as handle:
                self.handori_dict = pickle.load(handle)

            print('hand data loaded')

            #self.file_names = self.dict['file_name']

        else :
            
            with open(self.filterpath, 'rb') as handle:
                self.file_names1 = pickle.load(handle)

            # /scratch/minjay/NIA_EXTRACT_FOR_TRAINING/train

            self.file_names2 = [f for f in os.listdir(self.directory_path) if f.endswith('.pickle')]

            file_name1 = set(self.file_names1)
            file_name2 = set(self.file_names2)

            self.file_names = list(file_name1 & file_name2)
            #self.file_names = [f for f in os.listdir(self.directory_path) if f.endswith('.pickle')][:1000]

            for file_name in tqdm(self.file_names) :

                file = os.path.join(self.directory_path,file_name)

                with open(file, 'rb') as handle:
                    temp_dict = pickle.load(handle)

                #dict_keys(['taxonomy', 'obj_verts', 'obj_contact', 'hand_verts', 'hand_contact', 'hand_joints', 'obj_ids'])
                #           int         [3886, 3]    (3886, 1):nd.array [778, 3]  (778, 1):nd.array [21, 3]      '21': str

                self.dict[int(file_name.split('.')[0])] = temp_dict

            self.dict['file_name'] = self.file_names

            with open(self.dataset_pkl_name, 'wb') as handle:
                pickle.dump(self.dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(self.taxonomyfixpath, 'rb') as handle:
                self.taxonomyfixed = pickle.load(handle)

            with open(self.handoripath, 'rb') as handle:
                self.handori_dict = pickle.load(handle)


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        sample = {}

        file_name = self.file_names[idx]

        ho_info    = self.dict[int(file_name.split('.')[0])]
        
        hand_dict = self.handori_dict[int(file_name.split('.')[0])]

        ho_gt = HandObject()

        obj_faces  = self.obj_faces_dict[int(ho_info['obj_ids'])]

        ho_gt.hand_verts   = ho_info['hand_verts'].to(self.device) 
        ho_gt.closed_faces = self.mano_faces
        ho_gt.obj_verts    = ho_info['obj_verts'].to(self.device)
        ho_gt.obj_faces    = obj_faces.to(self.device)
        ho_gt.hand_joints  = ho_info['hand_joints'].to(self.device)

        sample['object_full_face'] = ho_gt.obj_faces 
        sample['object_full_verts'] = ho_gt.obj_verts
        sample['obj_id'] = int(ho_info['obj_ids'])

        gt_contact_hand = ho_info['hand_contact']

        ho_gt.obj_contact  = ho_info['obj_contact']
        ho_gt.hand_contact  = gt_contact_hand

        ho_gt.hand_beta = hand_dict['hand_beta']

        ho_gt.hand_pose = (hand_dict['hand_pose']).mm(self.pca_inv)[:,:self.ncomps]

        ho_gt.hand_rot  = hand_dict['hand_mTc'] # Rotation

        sample['taxonomy']     = TAX_DICT[int(self.taxonomyfixed[int(file_name.split('.')[0])])] - 1

        sample['hand_joints_gt'] = ho_gt.hand_joints
        sample['hand_pose_gt'] = hand_dict['hand_pose'][0]

        aug_trans=0.01
        aug_rot=0.1 #.1
        aug_pca=0 #.3 #.2 # 3

        aug_t = np.random.randn(3) * aug_trans
        aug_p = np.concatenate((np.random.randn(3) * aug_rot, np.random.randn(self.ncomps) * aug_pca)).astype(np.float32)

        ho_gt.calc_dist_contact(hand=True, obj=True)
        ho_gt.hand_contact[NON_SURFACE] = 0.05

        ho_aug = HandObject()

        ho_aug.load_from_ho_new(ho_gt, self.mano_run, self.device, aug_p, aug_t)

        ######## ---- ########
        sample['aug_pose']  = ho_aug.hand_pose
        sample['aug_rot']   = ho_aug.hand_rot
        sample['aug_trans'] = ho_aug.hand_trans.float()
        sample['hand_beta'] = ho_aug.hand_beta
        sample['idx'] = idx
        ######## ---- ########

        ho_aug.closed_faces = self.mano_faces

        obj_sampled_idx = np.random.randint(0, len(ho_gt.obj_verts), 2048)

        ######## ---- ########
        sample['obj_sampled_idx'] = obj_sampled_idx
        ######## ---- ########

        hand_feats_aug, obj_feats_aug = ho_aug.generate_pointnet_features(obj_sampled_idx,self.device)
        hand_feats_gt, obj_feats_gt = ho_gt.generate_pointnet_features(obj_sampled_idx,self.device)

        ######## ---- ########
        sample['obj_normals_aug'] = ho_aug.obj_normals
        ######## ---- ########
        
        sample['hand_feats']   = hand_feats_aug
        sample['obj_feats']    = obj_feats_aug

        obj_verts = ho_aug.obj_verts[obj_sampled_idx, :] 
        obj_contact = ho_gt.obj_contact[obj_sampled_idx, :] 


        sample['hand_verts']   = ho_aug.hand_verts
        sample['obj_verts']    = obj_verts

        mano_verts = ho_aug.hand_verts

        x_coords = mano_verts[:, 0]
        y_coords = mano_verts[:, 1]
        z_coords = mano_verts[:, 2]

        x_min, x_max = torch.min(x_coords), torch.max(x_coords)
        y_min, y_max = torch.min(y_coords), torch.max(y_coords)
        z_min, z_max = torch.min(z_coords), torch.max(z_coords)

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2

        mano_verts = mano_verts - torch.Tensor([x_center, y_center, z_center]).to(self.device)

        sample['mano_verts'] = mano_verts

        mano_verts_gt = ho_gt.hand_verts

        x_coords = mano_verts_gt[:, 0]
        y_coords = mano_verts_gt[:, 1]
        z_coords = mano_verts_gt[:, 2]

        x_min, x_max = torch.min(x_coords), torch.max(x_coords)
        y_min, y_max = torch.min(y_coords), torch.max(y_coords)
        z_min, z_max = torch.min(z_coords), torch.max(z_coords)

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2

        mano_verts_gt = mano_verts_gt - torch.Tensor([x_center, y_center, z_center]).to(self.device)

        sample['mano_verts'] = mano_verts

        sample['hand_verts_gt'] = ho_gt.hand_verts
        sample['obj_verts_gt'] =  ho_gt.obj_verts[obj_sampled_idx, :] 
        sample['hand_feats_gt'] = hand_feats_gt
        sample['obj_feats_gt'] = obj_feats_gt
        sample['mano_verts_gt'] = mano_verts_gt

        #sample['hand_feats']   = hand_feats_aug
        #sample['obj_feats']    = obj_feats_aug
        
        sample['hand_contact'] = ho_gt.hand_contact
        sample['obj_contact']  = obj_contact

        ho_aug.calc_dist_contact(hand=True, obj=False)

        sample['hand_contact_aug'] = ho_aug.hand_contact

        return sample  # Dummy label, replace as needed

    @staticmethod
    def collate_fn(batch):
        sample = dict()
        batch_keys = batch[0].keys()
        
        skip_keys = ['obj_normals_aug','object_full_face','object_full_verts']    # These will be manually collated

        # For each not in skip_keys, use default torch collator
        for key in [k for k in batch_keys if k not in skip_keys]:
            sample[key] = torch.utils.data._utils.collate.default_collate([d[key] for d in batch])

        obj_normals_aug_all = [sample['obj_normals_aug'] for sample in batch]
        object_full_face_all = [sample['object_full_face'] for sample in batch]
        object_full_verts_all = [sample['object_full_verts'] for sample in batch]

        sample['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded(obj_normals_aug_all, pad_value=-1)
        sample['object_full_face'] = pytorch3d.structures.utils.list_to_padded(object_full_face_all, pad_value=-1)
        sample['object_full_verts'] = pytorch3d.structures.utils.list_to_padded(object_full_verts_all, pad_value=-1)

        return sample


class DexYCB(Dataset):
    def __init__(self,setup):

        #self.dataset_pkl_name = setup + '1000.pkl'
        self.dataset_pkl_name = f'/scratch/minjay/ContactOpt/dexycb_{setup}_2024.pkl'
        
        self.dict = {}

        self.setup = setup

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mano_layer = ManoLayer(side='right', mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=False, flat_hand_mean=True,
                            center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(self.device)

        self.mano_faces = self.mano_layer.th_faces

        with open(self.dataset_pkl_name, 'rb') as handle:
            self.dict = pickle.load(handle)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):

        sample = {}

        ho_gt = HandObject()

        hoi_info = self.dict[idx]

        obj_faces  = hoi_info['obj_face'] #FIXME:

        ho_gt.hand_verts   = hoi_info['hand_verts_gt'].to(self.device) #FIXME: 
        ho_gt.closed_faces = self.mano_faces
        ho_gt.obj_verts    = hoi_info['obj_verts_gt'].to(self.device) #FIXME:
        ho_gt.obj_faces    = obj_faces.to(self.device) 
        ho_gt.hand_joints  = hoi_info['joints_gt'].to(self.device) #FIXME:

        sample['object_full_face'] = ho_gt.obj_faces  #FIXME:
        sample['object_full_verts'] = ho_gt.obj_verts #FIXME:

        gt_contact_hand = hoi_info['hand_contact_gt'] #FIXME:
        gt_contact_hand[SURFACE] = 0.05

        ho_gt.obj_contact  = hoi_info['obj_contact_gt'] #FIXME:
        ho_gt.hand_contact  = gt_contact_hand

        sample['taxonomy'] = hoi_info['taxonomy'] - 1
        sample['hand_joints_gt'] = ho_gt.hand_joints
        sample['idx'] = idx

        ho_aug = HandObject()

        ho_aug.hand_verts   = hoi_info['hand_verts_pred'].to(self.device) #FIXME: 
        ho_aug.closed_faces = self.mano_faces
        ho_aug.obj_verts    = hoi_info['obj_verts_pred'].to(self.device) #FIXME:
        ho_aug.obj_faces    = obj_faces.to(self.device) 
        ho_aug.hand_joints  = hoi_info['joints_pred'].to(self.device) #FIXME:

        obj_sampled_idx = np.random.randint(0, len(ho_gt.obj_verts), 2048)

        sample['obj_sampled_idx'] = obj_sampled_idx

        hand_feats_aug, obj_feats_aug = ho_aug.generate_pointnet_features(obj_sampled_idx,self.device)

        sample['obj_normals_aug'] = ho_aug.obj_normals

        if TAXONOMY_CONDITION :

            num_classes = 33

            # Generate one-hot encoding
            import torch.nn.functional as F

            one_hot_encoding = F.one_hot(torch.tensor([sample['taxonomy']]), num_classes=num_classes) #FIXME:
            one_hot_encoding_hand = one_hot_encoding.repeat(778,1).to(self.device)
            one_hot_encoding_obj = one_hot_encoding.repeat(2048,1).to(self.device)

            hand_feats_aug_new = torch.cat((hand_feats_aug, one_hot_encoding_hand), dim=1)
            obj_feats_aug_new = torch.cat((obj_feats_aug, one_hot_encoding_obj), dim=1)

            sample['hand_feats']   = hand_feats_aug_new
            sample['obj_feats']    = obj_feats_aug_new

        else :            
            sample['hand_feats']   = hand_feats_aug
            sample['obj_feats']    = obj_feats_aug

        obj_verts = ho_aug.obj_verts[obj_sampled_idx, :] 
        obj_contact = ho_gt.obj_contact[obj_sampled_idx, :] 

        sample['hand_verts']   = ho_aug.hand_verts
        sample['obj_verts']    = obj_verts

        # mano_verts = ho_aug.hand_verts

        # x_coords = mano_verts[:, 0]
        # y_coords = mano_verts[:, 1]
        # z_coords = mano_verts[:, 2]

        # x_min, x_max = torch.min(x_coords), torch.max(x_coords)
        # y_min, y_max = torch.min(y_coords), torch.max(y_coords)
        # z_min, z_max = torch.min(z_coords), torch.max(z_coords)

        # x_center = (x_max + x_min) / 2
        # y_center = (y_max + y_min) / 2
        # z_center = (z_max + z_min) / 2

        # mano_verts = mano_verts - torch.Tensor([x_center, y_center, z_center]).to(self.device)

        # sample['mano_verts'] = mano_verts
        
        sample['hand_contact'] = ho_gt.hand_contact
        sample['obj_contact']  = obj_contact

        ho_aug.calc_dist_contact(hand=True, obj=False)

        sample['hand_contact_aug'] = ho_aug.hand_contact

        return sample  # Dummy label, replace as needed

    @staticmethod
    def collate_fn(batch):
        sample = dict()
        batch_keys = batch[0].keys()
        
        skip_keys = ['obj_normals_aug','object_full_face','object_full_verts']    # These will be manually collated

        # For each not in skip_keys, use default torch collator
        for key in [k for k in batch_keys if k not in skip_keys]:
            sample[key] = torch.utils.data._utils.collate.default_collate([d[key] for d in batch])

        obj_normals_aug_all = [sample['obj_normals_aug'] for sample in batch]
        object_full_face_all = [sample['object_full_face'] for sample in batch]
        object_full_verts_all = [sample['object_full_verts'] for sample in batch]

        sample['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded(obj_normals_aug_all, pad_value=-1)
        sample['object_full_face'] = pytorch3d.structures.utils.list_to_padded(object_full_face_all, pad_value=-1)
        sample['object_full_verts'] = pytorch3d.structures.utils.list_to_padded(object_full_verts_all, pad_value=-1)

        return sample

class ContactPose(Dataset):
    def __init__(self,dirpath,setup):

        print("Loading ContactPose Dataset")

        self.dataset_pkl_name = os.path.join(dirpath,f'perturbed_contactpose_{setup}.pkl')

        self.dict = {}

        self.directory_path = os.path.join(dirpath,setup)

        #self.obj_face_pkl =  os.path.join(dirpath,'obj_face_dict.pickle')

        self.setup = setup

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mano_layer = ManoLayer(side='right', mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=False, flat_hand_mean=True,
                            center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(self.device)



        self.mano_faces = self.mano_layer.th_faces

        # with open(self.obj_face_pkl, 'rb') as handle:
        #         obj_face_dict = pickle.load(handle)

        # self.obj_faces_dict = obj_face_dict

        if os.path.isfile(self.dataset_pkl_name) :

            with open(self.dataset_pkl_name, 'rb') as handle:
                dict_ = pickle.load(handle)

            self.dict = dict_

        else :

            print(self.dataset_pkl_name)
            print('Wrong Path for pkl files')



    def __len__(self):
        return len(self.dict)    
    
    def __getitem__(self, idx):

        sample = {}

        ho_gt = HandObject()

        ho_gt = self.dict[idx]['ho_gt']

        ho_aug = self.dict[idx]['ho_aug']

        # Testing input as a AUG #
        ##############################
        # Testing input as a AUG #
                
        obj_sampled_idx = self.dict[idx]['obj_sampled_idx']

        obj_verts = ho_aug.obj_verts[obj_sampled_idx, :] 
        obj_contact = ho_gt.obj_contact[obj_sampled_idx, :] 

        sample['hand_verts']   = torch.Tensor(ho_aug.hand_verts).to(self.device)
        sample['obj_verts']    = torch.Tensor(obj_verts).to(self.device)
        sample['hand_feats']   = torch.Tensor(self.dict[idx]['hand_feats_aug']).to(self.device)
        sample['obj_feats']    = torch.Tensor(self.dict[idx]['obj_feats_aug']).to(self.device)
        sample['taxonomy']     = 0
        sample['hand_contact'] = ho_gt.hand_contact
        sample['obj_contact']  = obj_contact

        return sample
    
    

def get_newest_checkpoint(args):
    """
    Finds the newest model checkpoint file, sorted by the date of the file
    :return: Model with loaded weights
    """
    list_of_files = glob.glob('checkpoints/*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    print('Loading checkpoint file:', latest_file)

    model = DeepContactNet(model=args.model)
    model.load_state_dict(torch.load(latest_file))
    return model

def run(model, train_loader, test_loader, epochs, optimizer_dict, writer, exp_name, device, args_o):

    train_losses, test_losses = [], []

    s_writer = SummaryWriter(f'runs/{exp_name}')

    epoch_division_idx = 0

    f1_score_hand = 0
    f1_score_obj = 0

    test_hand_contact_loss=0
    test_obj_contact_loss=0
    f1_score_hand=0
    f1_score_obj=0
    precision_hand=0
    recall_hand=0
    precision_obj=0
    recall_obj=0
    
    model_dict = model.state_dict()

    pointnet_init = '/scratch/minjay/s2contact/Generator_pretrained_surfacecut/checkpoint_200.pt'

    model_init = '/scratch/minjay/s2contact/Pre2StepJoCoR_0.2/checkpoint_045.pt'

    print(model_init)
    # '/scratch/minjay/s2contact/Pre2StepCE_0.0/checkpoint_145.pt'
    # '/scratch/minjay/s2contact/Pre2StepJoCoR_0.2/checkpoint_045.pt'
    # '/scratch/minjay/s2contact/Pre2StepJoCoR_0.0/checkpoint_200.pt'
    # '/scratch/minjay/s2contact/Generator_pretrained_surfacecut/checkpoint_200.pt'
    
    #'/scratch/minjay/s2contact/Generator_pretrained_surfacecut_classifier/checkpoint_025.pt'

    if CLASSIFIER_PRETRAINED :

        pretrained_dict = torch.load(model_init)['model_state_dict']

        model.load_state_dict(pretrained_dict)

    else :

        pretrained_dict = torch.load(pointnet_init)['model_state_dict']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'pointnet' in k}

        model_dict.update(pretrained_dict)

        model.load_state_dict(model_dict)

    '''

    ### PRETRAIN G

    
    for epoch in range(1, epochs + 1):

        t = time.time()

        optmizer_g = optimizer_dict['opt_g']
        inv_lr_scheduler(optimizer_dict['param_lr_g'], optmizer_g, epoch, init_lr=optimizer_dict['train_lr'],max_iter=optimizer_dict['max_iter'])

        train(model, optmizer_g, train_loader, device, writer, s_writer, epoch, epochs)
                
        t_duration = time.time() - t

        if epoch % 5 == 0 :
            test_hand_contact_loss, test_obj_contact_loss , f1_score_hand, f1_score_obj, precision_hand, recall_hand, precision_obj, recall_obj = test(model, test_loader, epoch, 0, 20, device)

        info = {
            'current_epoch': epoch,
            't_duration' : t_duration,
            'epochs': epochs,
            'f1_score_hand' : f1_score_hand,
            'tax_acc' : 0,
            'f1_score_obj' : f1_score_obj,
            'test_hand_contact_loss' : test_hand_contact_loss,
            'test_obj_contact_loss' : test_obj_contact_loss,
            'precision_hand' : precision_hand,
            'recall_hand' : recall_hand,
            'precision_obj' : precision_obj,
            'recall_obj' :recall_obj
        }

        writer.print_info_test(info)
        writer.s_writer_test(info,s_writer,epoch)

        if epoch % 5 == 0 :
            writer.save_checkpoint(model, optmizer_g, epoch, exp_name)

    ##### PRETRAIN C1, C2

    '''

    '''
    jo_exponent = 1
    num_gradual = 20

    rate_schedule = np.ones(epochs) * args_o.jo_noise_rate
    rate_schedule[:num_gradual] = np.linspace(0,  args_o.jo_noise_rate ** jo_exponent, num_gradual)

    epochs = 100

    for epoch in range(1, epochs + 1):

        forget_rate = rate_schedule[epoch-1]

        optmizer_c1 = optimizer_dict['opt_c1']
        inv_lr_scheduler(optimizer_dict['param_lr_f1'], optmizer_c1, epoch, init_lr=optimizer_dict['train_lr'],max_iter=optimizer_dict['max_iter'])

        optmizer_c2 = optimizer_dict['opt_c2']
        inv_lr_scheduler(optimizer_dict['param_lr_f2'], optmizer_c2, epoch, init_lr=optimizer_dict['train_lr'],max_iter=optimizer_dict['max_iter'])

        train_A_step(model, optmizer_c1, optmizer_c2, forget_rate, train_loader, device, writer, s_writer, epoch, epochs)

        t = time.time()
                
        t_duration = time.time() - t

        info = {
            'current_epoch': epoch,
            't_duration' : t_duration,
            'epochs': epochs,
            'f1_score_hand' : f1_score_hand,
            'tax_acc' : 0,
            'f1_score_obj' : f1_score_obj,
            'test_hand_contact_loss' : test_hand_contact_loss,
            'test_obj_contact_loss' : test_obj_contact_loss,
            'precision_hand' : precision_hand,
            'recall_hand' : recall_hand,
            'precision_obj' : precision_obj,
            'recall_obj' :recall_obj
        }

        writer.print_info_test(info)
        writer.s_writer_test(info,s_writer,epoch)

        if epoch % 5 == 0 :
            writer.save_checkpoint(model, optmizer_c1, epoch, exp_name)
    
    '''

    jo_exponent = 1
    num_gradual = 20

    rate_schedule = np.ones(epochs) * args_o.jo_noise_rate
    rate_schedule[:num_gradual] = np.linspace(0,  args_o.jo_noise_rate ** jo_exponent, num_gradual)

    start_1 = 1

    if START_epoch11 == True :
        import pdb; pdb.set_trace()
        start_1 = 11

        if STEPAB == True :
            import pdb; pdb.set_trace()
            
            model_path = '/scratch/minjay/s2contact/NIA_AUG_TRANS_0.01_ROT_0.1_incep_classifier_0.3_rate_2step_clamp_fix_remove_recon/checkpoint_010.pt'
            pretrained_dict = torch.load(model_path)['model_state_dict']
            model.load_state_dict(pretrained_dict)

        else :
            import pdb; pdb.set_trace()
            model_path = '/scratch/minjay/s2contact/NIA_AUG_TRANS_0.01_ROT_0.1_incep_classifier_0.3_rate_single_clamp_fix_remove_recon/checkpoint_010.pt'
            pretrained_dict = torch.load(model_path)['model_state_dict']
            model.load_state_dict(pretrained_dict)
            
    for epoch in range(start_1, epochs + 1):

        forget_rate = rate_schedule[epoch-1]

        t = time.time()

        optmizer_g = optimizer_dict['opt_g']
        inv_lr_scheduler(optimizer_dict['param_lr_g'], optmizer_g,  epoch, init_lr=optimizer_dict['train_lr'],max_iter=optimizer_dict['max_iter'])

        optmizer_c1 = optimizer_dict['opt_c1']
        inv_lr_scheduler(optimizer_dict['param_lr_f1'], optmizer_c1, epoch, init_lr=optimizer_dict['train_lr'],max_iter=optimizer_dict['max_iter'])

        optmizer_c2 = optimizer_dict['opt_c2']
        inv_lr_scheduler(optimizer_dict['param_lr_f2'], optmizer_c2, epoch, init_lr=optimizer_dict['train_lr'],max_iter=optimizer_dict['max_iter'])

        if TRAIN2STEP == '2step' :

            iteration = 100

            len_train_loader = len(train_loader)

            mok, left = divmod(len_train_loader, iteration)

            model.train()

            for idx, sample in enumerate(tqdm(train_loader)) :

                B_total_contact_loss_hand = 0
                B_total_contact_loss_obj = 0
                B_total_taxonomy_loss = 0

                A_total_loss_s = 0
                A_total_recon = 0
                A_train_correct = 0
                A_train_correct2 = 0

                iteration_A_info = train_A_step_v2(model, optmizer_c1, optmizer_c2, optmizer_g, forget_rate, idx, sample, device, writer, s_writer, epoch, epochs)

                A_total_loss_s += iteration_A_info['loss_s']
                A_total_recon  += iteration_A_info['total_recon']
                A_train_correct += iteration_A_info['train_acc1']
                A_train_correct2 += iteration_A_info['train_acc2']

                writer.s_writer_train_stepA_iteration(iteration_A_info, s_writer, (epoch-1) * iteration  + idx )

                for train_B_step_iter in range(4) :
                    iteration_B_info = train_B_step_target_v2(model, optmizer_g, idx, sample, device, writer, s_writer, epoch, epochs)

                B_total_contact_loss_hand += iteration_B_info['train_contact_loss_hand']
                B_total_contact_loss_obj  += iteration_B_info['train_contact_loss_obj']
                B_total_taxonomy_loss += iteration_B_info['train_taxonomy_loss']

                writer.s_writer_train_stepB_iteration_V2(iteration_B_info, s_writer, (epoch-1) * iteration  + idx )

                if idx % iteration == 0 and idx > 0 :
                    
                    info = {
                        'epochs' : epochs,
                        'current_epoch': (epoch-1) * mok + idx // iteration ,
                        'train_contact_loss_obj' : B_total_contact_loss_obj / iteration,
                        'train_contact_loss_hand' : B_total_contact_loss_hand / iteration,
                        'train_taxonomy_loss' : B_total_taxonomy_loss / iteration,
                        'total_contact_loss_gt' : 0,
                        'mok': mok 
                    }
                    
                    writer.print_info_train_stepB_V2(info)
                    writer.s_writer_train_stepB_V2(info,s_writer,info['current_epoch'])

                    train_acc1 = float(A_train_correct)  / float(iteration)
                    train_acc2 = float(A_train_correct2) / float(iteration)
                    total_loss_s = float(A_total_loss_s) / float(iteration)            

                    info = {
                        'epochs' : epochs,
                        'current_epoch': (epoch-1) * mok + idx // iteration ,
                        'train_acc1' : train_acc1,
                        'train_acc2' : train_acc2,
                        'total_loss_s' : total_loss_s,
                        'total_recon' : A_total_recon / iteration, 
                        'mok': mok 
                    }

                    writer.print_info_train_stepA(info)
                    writer.s_writer_train_stepA(info,s_writer,info['current_epoch'])

                    break

            #train_B_step_source_target(model, optmizer_g, train_loader, device, writer, s_writer, epoch, epochs)
            #train_A_step_source(model, optmizer_c1, optmizer_c2, forget_rate, train_loader, device, writer, s_writer, epoch, epochs)

            #train_A_step(model, optmizer_c1, optmizer_c2, forget_rate, train_loader, device, writer, s_writer, epoch, epochs)
            #train_B_step(model, optmizer_g, train_loader, device, writer, s_writer, epoch, epochs)

        elif  TRAIN2STEP == 'single' :
            train_AB_step(model, optmizer_c1, optmizer_c2, optmizer_g, forget_rate, train_loader, device, writer, s_writer, epoch, epochs)

        t_duration = time.time() - t

        test_hand_contact_loss, test_obj_contact_loss , f1_score_hand, f1_score_obj, precision_hand, recall_hand, precision_obj, recall_obj = test(model, test_loader, epoch, 0, 5, device)

        info = {
            'current_epoch': epoch,
            't_duration' : t_duration,
            'epochs': epochs,
            'f1_score_hand' : f1_score_hand,
            'tax_acc' : 0,
            'f1_score_obj' : f1_score_obj,
            'test_hand_contact_loss' : test_hand_contact_loss,
            'test_obj_contact_loss' : test_obj_contact_loss,
            'precision_hand' : precision_hand,
            'recall_hand' : recall_hand,
            'precision_obj' : precision_obj,
            'recall_obj' :recall_obj
        }

        writer.print_info_test(info)
        writer.s_writer_test(info,s_writer,epoch)

        if epoch % 5 == 0 :
            writer.save_checkpoint(model, optmizer_g, epoch, exp_name)
    
    s_writer.close()


def final_run(model, test_loader, optimizer, scheduler, writer, exp_name, device):

    train_losses, test_losses = [], []

    s_writer = SummaryWriter(f'runs/{exp_name}')

    print(test_loader)

    test_hand_contact_loss, test_obj_contact_loss , f1_score_hand, f1_score_obj, precision_hand, recall_hand, precision_obj, recall_obj = test_final(model, test_loader, device)

    info = {
        'current_epoch': 0,
        't_duration' : 0,
        'epochs': 0,
        'train_contact_loss_obj' : 0,
        'train_contact_loss_hand' : 0,
        'train_taxonomy_loss' : 0,
        'f1_score_hand' : f1_score_hand,
        'tax_acc' : 0,
        'f1_score_obj' : f1_score_obj,
        'test_hand_contact_loss' : test_hand_contact_loss,
        'test_obj_contact_loss' : test_obj_contact_loss,
        'precision_hand' : precision_hand,
        'recall_hand' : recall_hand,
        'precision_obj' : precision_obj,
        'recall_obj' :recall_obj
    }

    writer.print_info(info)

    s_writer.close()

def train_A_step(model, opt_c1, opt_c2, forget_rate, train_loader, device, writer, s_writer, epoch, epochs):

    train_correct = 0
    train_correct2 = 0

    total_loss_s = 0
    total_recon = 0

    model.train()

    loss_l2 = torch.nn.MSELoss(reduction='mean').to(device)
    criterion_taxonomy = nn.CrossEntropyLoss().to(device)

    iteration = 100

    len_train_loader = len(train_loader)

    mok, left = divmod(len_train_loader, iteration)

    model.C1.train()
    model.C2.train()

    for idx, sample in enumerate(tqdm(train_loader)) :

        opt_c1.zero_grad()
        opt_c2.zero_grad()

        out = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

        gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

        labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

        ## TRAIN C1

        out_c1, logit_c1, coma_input_c1 = model.forward_C1(sample['mano_verts'],out['contact_hand'])

        #recon_loss_c1 = loss_l2(out_c1, coma_input_c1)

        ## TRAIN C2

        out_c2, logit_c2, coma_input_c2 = model.forward_C2(sample['mano_verts'],out['contact_hand'])

        #recon_loss_c2 = loss_l2(out_c2, coma_input_c2)

        if epoch > CE_STEP:
            loss_s, ind_update = loss_select(logit_c1, logit_c2, labels, forget_rate)
        else:
            loss_s = criterion_taxonomy(logit_c1, labels) + criterion_taxonomy(logit_c2, labels)

        prec1 = accuracy(logit_c1, labels, topk=(1,))

        train_correct += prec1

        prec2 = accuracy(logit_c2, labels, topk=(1,))

        train_correct2 += prec2

        loss =  loss_s #+ recon_loss_c1 +  recon_loss_c2 

        loss.backward()

        opt_c1.step()
        opt_c2.step()

        total_loss_s += loss_s.item()
        #total_recon += recon_loss_c1.item() + recon_loss_c2.item()

        iteration_info = {
                'train_acc1' : prec1,
                'train_acc2' : prec2,
                'total_loss_s' : loss_s.item(),
                'total_recon'  : 0, #recon_loss_c1.item() + recon_loss_c2.item(),
                'mok': mok 
        }

        writer.s_writer_train_stepA_iteration(iteration_info, s_writer, (epoch-1) * iteration  + idx )
        

        if idx % iteration == 0 and idx > 0 :

            train_acc1 = float(train_correct)  / float(iteration)
            train_acc2 = float(train_correct2) / float(iteration)

            total_loss_s = float(total_loss_s) / float(iteration)
            #total_recon  = float(total_recon) / float(iteration)
            

            info = {
                'epochs' : epochs,
                'current_epoch': (epoch-1) * mok + idx // iteration ,
                'train_acc1' : train_acc1,
                'train_acc2' : train_acc2,
                'total_loss_s' : total_loss_s,
                'total_recon' : 0, #total_recon,
                'mok': mok 
            }
            
            writer.print_info_train_stepA(info)
            writer.s_writer_train_stepA(info,s_writer,info['current_epoch'])

            break
                
    return 0

def train_A_step_source(model, opt_c1, opt_c2, forget_rate, train_loader, device, writer, s_writer, epoch, epochs):

    train_correct = 0
    train_correct2 = 0

    total_loss_s = 0
    total_recon = 0

    model.train()

    loss_l2 = torch.nn.MSELoss(reduction='mean').to(device)
    criterion_taxonomy = nn.CrossEntropyLoss().to(device)
    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)

    criterion_hand = torch.nn.NLLLoss(weight=bin_weights)
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights)

    iteration = 100

    len_train_loader = len(train_loader)

    mok, left = divmod(len_train_loader, iteration)

    model.C1.train()
    model.C2.train()

    for idx, sample in enumerate(tqdm(train_loader)) :

        opt_c1.zero_grad()
        opt_c2.zero_grad()

        ## LABELS 

        gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

        labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

        ## SOURCE

        out_gt = model(sample['hand_verts_gt'], sample['hand_feats_gt'].to(torch.float32), sample['obj_verts_gt'], sample['obj_feats_gt'].to(torch.float32))

        ## TRAIN C1

        out_c1, logit_c1, coma_input_c1 = model.forward_C1(sample['mano_verts_gt'],out_gt['contact_hand'])

        #recon_loss_c1 = loss_l2(out_c1, coma_input_c1)

        ## TRAIN C2

        out_c2, logit_c2, coma_input_c2 = model.forward_C2(sample['mano_verts_gt'],out_gt['contact_hand'])

        #recon_loss_c2 = loss_l2(out_c2, coma_input_c2)

        if epoch > CE_STEP:
            loss_s, ind_update = loss_select(logit_c1, logit_c2, labels, forget_rate)
        else:
            loss_s = criterion_taxonomy(logit_c1, labels) + criterion_taxonomy(logit_c2, labels)

        prec1 = accuracy(logit_c1, labels, topk=(1,))

        train_correct += prec1

        prec2 = accuracy(logit_c2, labels, topk=(1,))

        train_correct2 += prec2

        loss =  loss_s #+ recon_loss_c1 +  recon_loss_c2 

        loss.backward()

        opt_c1.step()
        opt_c2.step()

        total_loss_s += loss_s.item()
        #total_recon += recon_loss_c1.item() + recon_loss_c2.item()

        iteration_info = {
                'train_acc1' : prec1,
                'train_acc2' : prec2,
                'total_loss_s' : loss_s.item(),
                'total_recon'  : 0, #recon_loss_c1.item() + recon_loss_c2.item(),
                'mok': mok 
        }

        writer.s_writer_train_stepA_iteration(iteration_info, s_writer, (epoch-1) * iteration  + idx )
        
        if idx % iteration == 0 and idx > 0 :

            train_acc1 = float(train_correct)  / float(iteration)
            train_acc2 = float(train_correct2) / float(iteration)

            total_loss_s = float(total_loss_s) / float(iteration)
            #total_recon  = float(total_recon) / float(iteration)
            

            info = {
                'epochs' : epochs,
                'current_epoch': (epoch-1) * mok + idx // iteration ,
                'train_acc1' : train_acc1,
                'train_acc2' : train_acc2,
                'total_loss_s' : total_loss_s,
                'total_recon' : 0, #total_recon,
                'mok': mok 
            }
            
            writer.print_info_train_stepA(info)
            writer.s_writer_train_stepA(info,s_writer,info['current_epoch'])

            break
                
    return 0

def train_A_step_v2(model, opt_c1, opt_c2, opt_g, forget_rate, idx, sample, device, writer, s_writer, epoch, epochs):

    loss_l2 = torch.nn.MSELoss(reduction='mean').to(device)
    criterion_taxonomy = nn.CrossEntropyLoss().to(device)
    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)

    criterion_hand = torch.nn.NLLLoss(weight=bin_weights)
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights)

    opt_c1.zero_grad()
    opt_c2.zero_grad()
    opt_g.zero_grad()

    ## LABELS 

    gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

    labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

    gt_contact_map_hand = val_to_class_0_9(sample['hand_contact']).squeeze(2).long().to(device)
    gt_contact_map_obj  = val_to_class_0_9(sample['obj_contact']).squeeze(2).long().to(device)

    ## SOURCE
    #out_gt = model(sample['hand_verts_gt'], sample['hand_feats_gt'].to(torch.float32), sample['obj_verts_gt'], sample['obj_feats_gt'].to(torch.float32))
    out_gt = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

    ## TRAIN C1
    out_c1, logit_c1, coma_input_c1 = model.forward_C1(sample['mano_verts_gt'],out_gt['contact_hand'])

    ## TRAIN C2
    out_c2, logit_c2, coma_input_c2 = model.forward_C2(sample['mano_verts_gt'],out_gt['contact_hand'])

    if epoch > CE_STEP:
        loss_s, ind_update = loss_select(logit_c1, logit_c2, labels, forget_rate)
    else:
        loss_s = criterion_taxonomy(logit_c1, labels) + criterion_taxonomy(logit_c2, labels)

    prec1 = accuracy(logit_c1, labels, topk=(1,))
    prec2 = accuracy(logit_c2, labels, topk=(1,))

    contact_hand_from_gt = out_gt['contact_hand'].to(device)
    contact_obj_from_gt  = out_gt['contact_obj'].to(device)

    contact_classify_loss_hand_gt = criterion_hand(contact_hand_from_gt.permute(0, 2, 1), gt_contact_map_hand)
    contact_classify_loss_obj_gt  = criterion_obj(contact_obj_from_gt.permute(0, 2, 1), gt_contact_map_obj)

    loss =  loss_s * TAX_WEIGHT_AMP + contact_classify_loss_hand_gt + contact_classify_loss_obj_gt

    loss.backward()

    opt_g.step()
    opt_c1.step()
    opt_c2.step()
    
    iteration_info = {
            'train_acc1' : prec1,
            'train_acc2' : prec2,
            'loss_s' : loss_s.item() * TAX_WEIGHT_AMP,
            'total_recon'  : contact_classify_loss_hand_gt.item() + contact_classify_loss_obj_gt.item(),
    }
                
    return iteration_info

def train_B_step(model, optimizer, train_loader, device, writer, s_writer, epoch, epochs):

    model.train()

    total_contact_loss_hand = 0
    total_contact_loss_obj = 0
    total_taxonomy_loss = 0

    criterion_taxonomy = nn.CrossEntropyLoss().to(device)
    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)

    criterion_hand = torch.nn.NLLLoss(weight=bin_weights)
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights)

    iteration = 100

    len_train_loader = len(train_loader)

    mok, left = divmod(len_train_loader, iteration)

    model.pointnet.train()

    for idx, sample in enumerate(tqdm(train_loader)) :

        optimizer.zero_grad()

        out = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

        contact_hand = out['contact_hand'].to(device)
        contact_obj  = out['contact_obj'].to(device)

        gt_contact_map_hand = val_to_class_0_9(sample['hand_contact']).squeeze(2).long().to(device)
        gt_contact_map_obj  = val_to_class_0_9(sample['obj_contact']).squeeze(2).long().to(device)

        contact_classify_loss_hand = criterion_hand(contact_hand.permute(0, 2, 1), gt_contact_map_hand)
        contact_classify_loss_obj = criterion_obj(contact_obj.permute(0, 2, 1), gt_contact_map_obj)

        out_c1, logit_c1, coma_input_c1 = model.forward_C1(sample['mano_verts'],out['contact_hand'])
        out_c2, logit_c2, coma_input_c2 = model.forward_C2(sample['mano_verts'],out['contact_hand'])

        gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

        labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

        loss_s = criterion_taxonomy(logit_c1, labels) + criterion_taxonomy(logit_c2, labels)

        loss =  contact_classify_loss_hand + contact_classify_loss_obj + loss_s

        loss.backward()

        optimizer.step()

        total_contact_loss_hand += contact_classify_loss_hand.item()
        total_contact_loss_obj  += contact_classify_loss_obj.item()
        total_taxonomy_loss += loss_s.item()

        iteration_info = {
                'train_contact_loss_obj' : contact_classify_loss_obj.item(),
                'train_contact_loss_hand' : contact_classify_loss_hand.item(),
                'train_taxonomy_loss' : loss_s.item(),
        }

        writer.s_writer_train_stepB_iteration(iteration_info, s_writer, (epoch-1) * iteration  + idx )

        if idx % iteration == 0 and idx > 0 :
            
            info = {
                'epochs' : epochs,
                'current_epoch': (epoch-1) * mok + idx // iteration ,
                'train_contact_loss_obj' : total_contact_loss_obj / iteration,
                'train_contact_loss_hand' : total_contact_loss_hand / iteration,
                'train_taxonomy_loss' : total_taxonomy_loss / iteration,
                'mok': mok 
            }
            
            writer.print_info_train_stepB(info)
            writer.s_writer_train_stepB(info,s_writer,info['current_epoch'])

            total_contact_loss_hand = 0
            total_contact_loss_obj = 0

            break
                
    return 0

def train_B_step_source_target(model, optimizer, train_loader, device, writer, s_writer, epoch, epochs):

    model.train()

    total_contact_loss_hand = 0
    total_contact_loss_obj = 0
    total_taxonomy_loss = 0

    total_contact_loss_gt = 0

    criterion_taxonomy = nn.CrossEntropyLoss().to(device)
    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)

    criterion_hand = torch.nn.NLLLoss(weight=bin_weights)
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights)

    iteration = 100

    len_train_loader = len(train_loader)

    mok, left = divmod(len_train_loader, iteration)

    model.pointnet.train()

    for idx, sample in enumerate(tqdm(train_loader)) :

        optimizer.zero_grad()

        out = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

        contact_hand = out['contact_hand'].to(device)
        contact_obj  = out['contact_obj'].to(device)

        gt_contact_map_hand = val_to_class_0_9(sample['hand_contact']).squeeze(2).long().to(device)
        gt_contact_map_obj  = val_to_class_0_9(sample['obj_contact']).squeeze(2).long().to(device)

        contact_classify_loss_hand = criterion_hand(contact_hand.permute(0, 2, 1), gt_contact_map_hand)
        contact_classify_loss_obj = criterion_obj(contact_obj.permute(0, 2, 1), gt_contact_map_obj)

        out_c1, logit_c1, coma_input_c1 = model.forward_C1(sample['mano_verts'],out['contact_hand'])
        out_c2, logit_c2, coma_input_c2 = model.forward_C2(sample['mano_verts'],out['contact_hand'])

        gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

        labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

        loss_s = criterion_taxonomy(logit_c1, labels) + criterion_taxonomy(logit_c2, labels)

        loss =  contact_classify_loss_hand + contact_classify_loss_obj + loss_s

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        out_gt = model(sample['hand_verts_gt'], sample['hand_feats_gt'].to(torch.float32), sample['obj_verts_gt'], sample['obj_feats_gt'].to(torch.float32))
        
        contact_hand_from_gt = out_gt['contact_hand'].to(device)
        contact_obj_from_gt  = out_gt['contact_obj'].to(device)

        contact_classify_loss_hand_gt = criterion_hand(contact_hand_from_gt.permute(0, 2, 1), gt_contact_map_hand)
        contact_classify_loss_obj_gt  = criterion_obj(contact_obj_from_gt.permute(0, 2, 1), gt_contact_map_obj)

        loss_gt =  contact_classify_loss_hand_gt + contact_classify_loss_obj_gt

        loss_gt.backward()

        optimizer.step()

        total_contact_loss_hand += contact_classify_loss_hand.item()
        total_contact_loss_obj  += contact_classify_loss_obj.item()
        total_taxonomy_loss += loss_s.item()
        total_contact_loss_gt += loss_gt.item()

        iteration_info = {
                'train_contact_loss_obj' : contact_classify_loss_obj.item(),
                'train_contact_loss_hand' : contact_classify_loss_hand.item(),
                'train_taxonomy_loss' : loss_s.item(),
                'loss_gt' : loss_gt.item()
        }

        writer.s_writer_train_stepB_iteration_V2(iteration_info, s_writer, (epoch-1) * iteration  + idx )

        if idx % iteration == 0 and idx > 0 :
            
            info = {
                'epochs' : epochs,
                'current_epoch': (epoch-1) * mok + idx // iteration ,
                'train_contact_loss_obj' : total_contact_loss_obj / iteration,
                'train_contact_loss_hand' : total_contact_loss_hand / iteration,
                'train_taxonomy_loss' : total_taxonomy_loss / iteration,
                'total_contact_loss_gt' : total_contact_loss_gt / iteration,
                'mok': mok 
            }
            
            writer.print_info_train_stepB_V2(info)
            writer.s_writer_train_stepB_V2(info,s_writer,info['current_epoch'])

            total_contact_loss_hand = 0
            total_contact_loss_obj = 0

            break
                
    return 0

def train_AB_step(model, opt_c1, opt_c2, opt_g, forget_rate, train_loader, device, writer, s_writer, epoch, epochs):

    model.train()

    total_contact_loss_hand = 0
    total_contact_loss_obj = 0

    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)

    criterion_hand = torch.nn.NLLLoss(weight=bin_weights,reduction = 'none')
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights, reduction = 'none')

    loss_l2 = torch.nn.MSELoss(reduction='mean').to(device)
    criterion_taxonomy = nn.CrossEntropyLoss().to(device)

    train_correct = 0
    train_correct2 = 0

    total_loss_s = 0
    total_recon = 0

    iteration = 100

    len_train_loader = len(train_loader)

    mok, left = divmod(len_train_loader, iteration)

    model.C1.train()
    model.C2.train()
    model.pointnet.train()

    for idx, sample in enumerate(tqdm(train_loader)) :

        opt_c1.zero_grad()
        opt_c2.zero_grad()
        opt_g.zero_grad()

        out = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

        ## TRAIN G

        contact_hand = out['contact_hand'].to(device)
        contact_obj  = out['contact_obj'].to(device)

        gt_contact_map_hand = val_to_class_0_9(sample['hand_contact']).squeeze(2).long().to(device)
        gt_contact_map_obj  = val_to_class_0_9(sample['obj_contact']).squeeze(2).long().to(device)

        contact_classify_loss_hand = criterion_hand(contact_hand.permute(0, 2, 1), gt_contact_map_hand)
        contact_classify_loss_obj = criterion_obj(contact_obj.permute(0, 2, 1), gt_contact_map_obj)

        ## TRAIN Classifier

        gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

        labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

        ## TRAIN C1

        out_c1, logit_c1, coma_input_c1 = model.forward_C1(sample['mano_verts'],out['contact_hand'])

        #recon_loss_c1 = loss_l2(out_c1, coma_input_c1)

        ## TRAIN C2

        out_c2, logit_c2, coma_input_c2 = model.forward_C2(sample['mano_verts'],out['contact_hand'])

        #recon_loss_c2 = loss_l2(out_c2, coma_input_c2)

        #recon_loss = recon_loss_c1 + recon_loss_c2

        #TODO: ind_update -> filter Contact Loss and CE loss

        if epoch > CE_STEP:

            loss_s, ind_update = loss_select(logit_c1, logit_c2, labels, forget_rate)

            loss_contact_hand = torch.mean(contact_classify_loss_hand[ind_update]) 
            loss_contact_obj = torch.mean(contact_classify_loss_obj[ind_update])

        else :
            loss_s = criterion_taxonomy(logit_c1, labels) + criterion_taxonomy(logit_c2, labels)

            loss_contact_hand = torch.mean(contact_classify_loss_hand) 
            loss_contact_obj = torch.mean(contact_classify_loss_obj)

        loss_contact = loss_contact_hand + loss_contact_obj

        prec1 = accuracy(logit_c1, labels, topk=(1,))

        train_correct += prec1

        prec2 = accuracy(logit_c2, labels, topk=(1,))

        train_correct2 += prec2

        loss =   loss_s + loss_contact # + recon_loss

        loss.backward()

        opt_g.step()
        opt_c1.step()
        opt_c2.step()
        

        total_loss_s += loss_s.item()
        total_recon += 0 #recon_loss.item()

        total_contact_loss_hand += loss_contact_hand.item()
        total_contact_loss_obj  += loss_contact_obj.item()

        iteration_info = {
                'train_acc1' : prec1,
                'train_acc2' : prec2,
                'total_loss_s' : loss_s.item(),
                'total_recon'  : 0, #recon_loss.item(),
                'train_contact_loss_obj' : loss_contact_hand.item(),
                'train_contact_loss_hand' : loss_contact_hand.item()
        }

        writer.s_writer_train_stepAB_iteration(iteration_info, s_writer, (epoch-1) * iteration  + idx )

        if idx % iteration == 0 and idx > 0 :

            info = {
                'epochs' : epochs,
                'current_epoch': (epoch-1) * mok + idx // iteration ,
                'train_acc1' : train_correct.item() / iteration,
                'train_acc2' : train_correct2.item() / iteration,
                'total_loss_s' : total_loss_s / iteration,
                'total_recon' : 0, #total_recon / iteration ,
                'train_contact_loss_obj' : total_contact_loss_obj / iteration,
                'train_contact_loss_hand' : total_contact_loss_hand / iteration,
                'mok': mok 
            }

            writer.print_info_train_stepAB(info)
            writer.s_writer_train_stepAB(info,s_writer,info['current_epoch'])

            break
                
    return 0


def train(model, optimizer, train_loader, device, writer, s_writer, epoch, epochs):

    model.train()

    total_contact_loss_hand = 0
    total_contact_loss_obj = 0

    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)

    criterion_hand = torch.nn.NLLLoss(weight=bin_weights)
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights)

    iteration = 100

    len_train_loader = len(train_loader)

    mok, left = divmod(len_train_loader, iteration)

    model.pointnet.train()

    for idx, sample in enumerate(tqdm(train_loader)) :

        optimizer.zero_grad()

        out = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

        contact_hand = out['contact_hand'].to(device)
        contact_obj  = out['contact_obj'].to(device)

        gt_contact_map_hand = val_to_class_0_9(sample['hand_contact']).squeeze(2).long().to(device)
        gt_contact_map_obj  = val_to_class_0_9(sample['obj_contact']).squeeze(2).long().to(device)

        contact_classify_loss_hand = criterion_hand(contact_hand.permute(0, 2, 1), gt_contact_map_hand)
        contact_classify_loss_obj = criterion_obj(contact_obj.permute(0, 2, 1), gt_contact_map_obj)

        loss =  contact_classify_loss_hand + contact_classify_loss_obj

        loss.backward()

        optimizer.step()

        total_contact_loss_hand += contact_classify_loss_hand.item()
        total_contact_loss_obj  += contact_classify_loss_obj.item()

        if idx % iteration == 0 and idx > 0 :
            
            info = {
                'epochs' : epochs,
                'current_epoch': (epoch-1) * mok + idx // iteration ,
                'train_contact_loss_obj' : total_contact_loss_obj / iteration,
                'train_contact_loss_hand' : total_contact_loss_hand / iteration,
                'train_taxonomy_loss' : 0,
                'joint_loss' : 0,
                'mano_pose_loss' : 0,
                'mok': mok 
            }
            
            writer.print_info_train(info)
            writer.s_writer_train(info,s_writer,info['current_epoch'])

            total_contact_loss_hand = 0
            total_contact_loss_obj = 0

            break
                
    return 0

def train_w_coma(model, coma, optimizer, train_loader, device, writer, s_writer, epoch, epochs):

    model.train()

    total_contact_loss_hand = 0
    total_contact_loss_obj = 0
    total_taxonomy_loss = 0

    total_joint_loss = 0
    total_mano_param_loss = 0

    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)

    criterion_hand = torch.nn.NLLLoss(weight=bin_weights,reduction='none')
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights,reduction='none')

    CE = nn.CrossEntropyLoss(reduction='none')
    loss_l2 = torch.nn.MSELoss()

    # a = iter(train_loader)

    #count_taken = 0
    ## 아래에 Count_taken 수정해줘야 함

    # for i in tqdm(range(start_idx,end_idx)) :

    #     sample = next(a)
    iteration = 100

    len_train_loader = len(train_loader)

    mok, left = divmod(len_train_loader, iteration)

    MANO = ManoLayer(mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=True, ncomps=15, side='right', flat_hand_mean=True, center_idx=0).to(device)

    th_selected_comps = MANO.th_selected_comps

    for idx, sample in enumerate(tqdm(train_loader)) :

        #count_taken += 1
        
        optimizer.zero_grad()

        if LATENT_CONDITION :
            
            #out = model.forward_onehot(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))
            out = model.forward_onehot_w_coma(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

            ##### TODO: MANO 
        else :            
            out = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

        hand_pose = out['mano_param']

        hand_rot = sample['aug_rot'][:,0,:].to(device)
        hand_beta = sample['hand_beta'][:,0,:].to(device)
        hand_trans = sample['aug_trans'][:,0,:].to(device)

        mano_param = torch.cat([hand_rot, hand_pose], dim=1)

        pred_verts, pred_joints = MANO(mano_param,hand_beta,th_trans=hand_trans)

        pred_verts = pred_verts / 1000
        pred_joints = pred_joints / 1000

        x_coords = pred_verts[:, :, 0]
        y_coords = pred_verts[:, :, 1]
        z_coords = pred_verts[:, :, 2]

        x_min, x_max = torch.min(x_coords, dim=1, keepdim=True)[0], torch.max(x_coords, dim=1, keepdim=True)[0]
        y_min, y_max = torch.min(y_coords, dim=1, keepdim=True)[0], torch.max(y_coords, dim=1, keepdim=True)[0]
        z_min, z_max = torch.min(z_coords, dim=1, keepdim=True)[0], torch.max(z_coords, dim=1, keepdim=True)[0]

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2

        xyz_center = torch.cat((x_center, y_center, z_center),dim=1).to(device)
        xyz_center_repeat = xyz_center.unsqueeze(1).repeat(1, 778, 1)

        norm_verts = pred_verts - xyz_center_repeat
        
        coma_out = model.forward_coma(out['contact_hand'].to(device),norm_verts)

        th_full_hand_pose = hand_pose.mm(th_selected_comps)

        joint_loss = loss_l2(pred_joints, sample['hand_joints_gt'] / 1000)
        mano_param_loss = loss_l2(th_full_hand_pose,sample['hand_pose_gt'].to(device))

        contact_hand = out['contact_hand'].to(device)
        contact_obj  = out['contact_obj'].to(device)
        #logit  = coma_out['logit'].to(device)

        gt_contact_map_hand = val_to_class_0_9(sample['hand_contact']).squeeze(2).long().to(device)
        gt_contact_map_obj  = val_to_class_0_9(sample['obj_contact']).squeeze(2).long().to(device)

        contact_classify_loss_hand = criterion_hand(contact_hand.permute(0, 2, 1), gt_contact_map_hand)
        contact_classify_loss_obj = criterion_obj(contact_obj.permute(0, 2, 1), gt_contact_map_obj)
        
        #gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 
        #labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

        #loss_ce = CE(logit, labels)     

        weight_loss = sample['weight'].to(device)

        #taxonomy_loss = (loss_ce * weight_loss).mean()

        contact_classify_loss_hand = (weight_loss.unsqueeze(1) * contact_classify_loss_hand).mean()
        contact_classify_loss_obj = (weight_loss.unsqueeze(1) * contact_classify_loss_obj).mean()

        loss =  contact_classify_loss_hand + contact_classify_loss_obj + mano_param_loss + joint_loss

        loss.backward()
        
        total_taxonomy_loss += 0 #taxonomy_loss.item()
        total_contact_loss_hand += contact_classify_loss_hand.item()
        total_contact_loss_obj  += contact_classify_loss_obj.item()
        total_joint_loss += joint_loss.item()
        total_mano_param_loss += mano_param_loss.item()


        loss.item()

        optimizer.step()

        if idx % iteration == 0 and idx > 0 :
            
            info = {
                'epochs' : epochs,
                'current_epoch': (epoch-1) * mok + idx // iteration ,
                'train_contact_loss_obj' : total_contact_loss_obj / iteration,
                'train_contact_loss_hand' : total_contact_loss_hand / iteration,
                'train_taxonomy_loss' : 0, #total_taxonomy_loss / iteration,
                'joint_loss' : total_joint_loss / iteration,
                'maon_pose_loss' : total_mano_param_loss / iteration,
                'mok': mok 
            }
            
            writer.print_info_train(info)
            writer.s_writer_train(info,s_writer,info['current_epoch'])

            total_contact_loss_hand = 0
            total_contact_loss_obj = 0
            total_taxonomy_loss = 0
            total_mano_param_loss = 0 
            total_joint_loss = 0
        
    #count_taken = len(train_loader)
        
    return 0


def train_B_step_target_v2(model, optimizer, idx, sample, device, writer, s_writer, epoch, epochs):

    criterion_taxonomy = nn.CrossEntropyLoss().to(device)
    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)

    criterion_hand = torch.nn.NLLLoss(weight=bin_weights)
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights)
    
    optimizer.zero_grad()

    out = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

    contact_hand = out['contact_hand'].to(device)
    contact_obj  = out['contact_obj'].to(device)

    gt_contact_map_hand = val_to_class_0_9(sample['hand_contact']).squeeze(2).long().to(device)
    gt_contact_map_obj  = val_to_class_0_9(sample['obj_contact']).squeeze(2).long().to(device)

    contact_classify_loss_hand = criterion_hand(contact_hand.permute(0, 2, 1), gt_contact_map_hand)
    contact_classify_loss_obj = criterion_obj(contact_obj.permute(0, 2, 1), gt_contact_map_obj)

    out_c1, logit_c1, coma_input_c1 = model.forward_C1(sample['mano_verts'],out['contact_hand'])
    out_c2, logit_c2, coma_input_c2 = model.forward_C2(sample['mano_verts'],out['contact_hand'])

    gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

    labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

    loss_s = criterion_taxonomy(logit_c1, labels) + criterion_taxonomy(logit_c2, labels)

    loss =  contact_classify_loss_hand + contact_classify_loss_obj + loss_s * TAX_WEIGHT_AMP

    loss.backward()

    optimizer.step()

    ## MOVE TO STEP A ##

    #optimizer.zero_grad()

    # out_gt = model(sample['hand_verts_gt'], sample['hand_feats_gt'].to(torch.float32), sample['obj_verts_gt'], sample['obj_feats_gt'].to(torch.float32))
    
    # contact_hand_from_gt = out_gt['contact_hand'].to(device)
    # contact_obj_from_gt  = out_gt['contact_obj'].to(device)

    # contact_classify_loss_hand_gt = criterion_hand(contact_hand_from_gt.permute(0, 2, 1), gt_contact_map_hand)
    # contact_classify_loss_obj_gt  = criterion_obj(contact_obj_from_gt.permute(0, 2, 1), gt_contact_map_obj)

    # loss_gt =  contact_classify_loss_hand_gt + contact_classify_loss_obj_gt

    # loss_gt.backward()

    # optimizer.step()

    iteration_info = {
            'train_contact_loss_obj' : contact_classify_loss_obj.item(),
            'train_contact_loss_hand' : contact_classify_loss_hand.item(),
            'train_taxonomy_loss' : loss_s.item() * TAX_WEIGHT_AMP,
            'loss_gt' : 0
    }
                
    return iteration_info

def test_final(model, test_loader, device) :
    #test(model, test_loader, test_loader_hoi, epoch, meshdata, exp_name, loss_weight, device):
    
    model.eval()

    total_contact_loss_hand = 0
    total_contact_loss_obj = 0

    total_f1_score_hand = 0
    total_f1_score_obj = 0

    total_precision_hand = 0
    total_recall_hand = 0
    total_precision_obj = 0
    total_recall_obj = 0

    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)
    criterion_hand = torch.nn.NLLLoss(weight=bin_weights)
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights)
    
    with torch.no_grad():
        
        for sample in tqdm(test_loader) :
            
            out = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

            contact_hand = out['contact_hand'].to(device)
            contact_obj  = out['contact_obj'].to(device)

            gt_contact_map_hand = val_to_class_0_9(sample['hand_contact']).squeeze(2).long().to(device)
            gt_contact_map_obj  = val_to_class_0_9(sample['obj_contact']).squeeze(2).long().to(device)

            contact_classify_loss_hand = criterion_hand(contact_hand.permute(0, 2, 1), gt_contact_map_hand)
            contact_classify_loss_obj = criterion_obj(contact_obj.permute(0, 2, 1), gt_contact_map_obj)

            total_contact_loss_hand += contact_classify_loss_hand.item()
            total_contact_loss_obj  += contact_classify_loss_obj.item()

            contact_pred_map_hand = contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)
            contact_pred_map_obj = contact_obj.permute(0, 2, 1).cpu().data.numpy().argmax(1)

            contact_gt_map_hand = gt_contact_map_hand.cpu().data.numpy()
            contact_gt_map_obj = gt_contact_map_obj.cpu().data.numpy()
            
            mask1_hand = contact_pred_map_hand > 1
            mask2_hand = contact_gt_map_hand > 1

            mask3_hand = contact_pred_map_hand == 0
            mask4_hand = contact_gt_map_hand == 0

            mask1_obj = contact_pred_map_obj > 1
            mask2_obj = contact_gt_map_obj > 1

            mask3_obj = contact_pred_map_obj == 0
            mask4_obj = contact_gt_map_obj == 0

            contact_pred_mask_hand = contact_pred_map_hand > 1
            contact_gt_mask_hand = contact_gt_map_hand > 1

            contact_pred_mask_obj = contact_pred_map_obj > 1
            contact_gt_mask_obj = contact_gt_map_obj > 1

            TP_hand = np.sum(np.logical_and(contact_pred_mask_hand == True, contact_gt_mask_hand == True))
            FP_hand = np.sum(np.logical_and(contact_pred_mask_hand == True, contact_gt_mask_hand == False))
            FN_hand = np.sum(np.logical_and(contact_pred_mask_hand == False, contact_gt_mask_hand == True))

            TP_obj = np.sum(np.logical_and(contact_pred_mask_obj == True, contact_gt_mask_obj == True))
            FP_obj = np.sum(np.logical_and(contact_pred_mask_obj == True, contact_gt_mask_obj == False))
            FN_obj = np.sum(np.logical_and(contact_pred_mask_obj == False, contact_gt_mask_obj == True))
            #precision = (contact_pred_map[mask_TP_and_FP] == contact_gt_map[mask_TP_and_FP]).mean()

            precision_hand = TP_hand / (TP_hand + FP_hand) if (TP_hand + FP_hand) > 0 else 0
            recall_hand = TP_hand / (TP_hand + FN_hand) if (TP_hand + FN_hand) > 0 else 0

            total_precision_hand += precision_hand
            total_recall_hand += recall_hand

            precision_obj = TP_obj / (TP_obj + FP_obj) if (TP_obj + FP_obj) > 0 else 0
            recall_obj = TP_obj / (TP_obj + FN_obj) if (TP_obj + FN_obj) > 0 else 0

            total_precision_obj += precision_obj
            total_recall_obj += recall_obj

            f1_score_hand = 2 * (precision_hand * recall_hand) / (precision_hand + recall_hand)
            f1_score_obj = 2 * (precision_obj * recall_obj) / (precision_obj + recall_obj)

            f1_score_hand = torch.tensor(f1_score_hand,dtype=torch.float32)
            f1_score_obj = torch.tensor(f1_score_obj,dtype=torch.float32)

            total_f1_score_hand += f1_score_hand.item()

            total_f1_score_obj += f1_score_obj.item()

    count_taken = len(test_loader)
                        
    return  total_contact_loss_hand / count_taken, total_contact_loss_obj / count_taken,  total_f1_score_hand / count_taken, total_f1_score_obj / count_taken, total_precision_hand / count_taken, total_recall_hand / count_taken, total_precision_obj / count_taken, total_recall_obj / count_taken

def test(model, test_loader, epoch, start_idx, end_idx, device) :
    #test(model, test_loader, test_loader_hoi, epoch, meshdata, exp_name, loss_weight, device):
    
    model.eval()

    total_contact_loss_hand = 0
    total_contact_loss_obj = 0

    total_f1_score_hand = 0
    total_f1_score_obj = 0

    total_precision_hand = 0
    total_recall_hand = 0
    total_precision_obj = 0
    total_recall_obj = 0

    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)
    criterion_hand = torch.nn.NLLLoss(weight=bin_weights)
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights)

    #total_taxonomy_loss = 0
    #total_taxonomy_acr = 0

    a = iter(test_loader)

    count_taken = 0

    lst_precision_hand = []
    lst_recall_hand = []

    lst_precision_obj = []
    lst_recall_obj = []
    
    with torch.no_grad():
        
        for i in tqdm(range(start_idx,end_idx)) :

            sample = next(a)
            count_taken += 1
                        
            out = model(sample['hand_verts'].to(device), sample['hand_feats'].to(torch.float32).to(device), sample['obj_verts'].to(device), sample['obj_feats'].to(torch.float32).to(device))
                
            contact_hand = out['contact_hand'].to(device)
            contact_obj  = out['contact_obj'].to(device)

            gt_contact_map_hand = val_to_class_0_9(sample['hand_contact'].to(device)).squeeze(2).long().to(device)
            gt_contact_map_obj  = val_to_class_0_9(sample['obj_contact'].to(device)).squeeze(2).long().to(device)

            contact_classify_loss_hand = criterion_hand(contact_hand.permute(0, 2, 1), gt_contact_map_hand)
            contact_classify_loss_obj = criterion_obj(contact_obj.permute(0, 2, 1), gt_contact_map_obj)

            total_contact_loss_hand += contact_classify_loss_hand.item()
            total_contact_loss_obj  += contact_classify_loss_obj.item()

            contact_pred_map_hand = contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)
            contact_pred_map_obj = contact_obj.permute(0, 2, 1).cpu().data.numpy().argmax(1)

            contact_gt_map_hand = gt_contact_map_hand.cpu().data.numpy()
            contact_gt_map_obj = gt_contact_map_obj.cpu().data.numpy()
            
            contact_pred_mask_hand = contact_pred_map_hand > 1
            contact_gt_mask_hand = contact_gt_map_hand > 1

            contact_pred_mask_obj = contact_pred_map_obj > 1
            contact_gt_mask_obj = contact_gt_map_obj > 1

            #import pdb; pdb.set_trace()
            #for i in range(32) : np.sum(contact_gt_mask_hand[i][NON_SURFACE])

            if TRAIN2STEP == 'test' :
            
                TP_hand = np.sum(np.logical_and(contact_pred_mask_hand == True, contact_gt_mask_hand == True),axis=1)
                FP_hand = np.sum(np.logical_and(contact_pred_mask_hand == True, contact_gt_mask_hand == False),axis=1)
                FN_hand = np.sum(np.logical_and(contact_pred_mask_hand == False, contact_gt_mask_hand == True),axis=1)

                TP_obj = np.sum(np.logical_and(contact_pred_mask_obj == True, contact_gt_mask_obj == True),axis=1)
                FP_obj = np.sum(np.logical_and(contact_pred_mask_obj == True, contact_gt_mask_obj == False),axis=1)
                FN_obj = np.sum(np.logical_and(contact_pred_mask_obj == False, contact_gt_mask_obj == True),axis=1)

                for tp_hand, fp_hand, fn_hand in zip(TP_hand,FP_hand,FN_hand) : 

                    precision_hand = tp_hand / (tp_hand + fp_hand) if (tp_hand + fp_hand) > 0 else 0
                    recall_hand = tp_hand / (tp_hand + fn_hand) if (tp_hand + fn_hand) > 0 else 0

                    lst_precision_hand.append(precision_hand)
                    lst_recall_hand.append(recall_hand)

                for tp_obj, fp_obj, fn_obj in zip(TP_obj,FP_obj,FN_obj) : 

                    precision_obj = tp_obj / (tp_obj + fp_obj) if (tp_obj + fp_obj) > 0 else 0
                    recall_obj = tp_obj / (tp_obj + fn_obj) if (tp_obj + fn_obj) > 0 else 0

                    lst_precision_obj.append(precision_obj)
                    lst_recall_obj.append(recall_obj)

            else :
                TP_hand = np.sum(np.logical_and(contact_pred_mask_hand == True, contact_gt_mask_hand == True))
                FP_hand = np.sum(np.logical_and(contact_pred_mask_hand == True, contact_gt_mask_hand == False))
                FN_hand = np.sum(np.logical_and(contact_pred_mask_hand == False, contact_gt_mask_hand == True))

                TP_obj = np.sum(np.logical_and(contact_pred_mask_obj == True, contact_gt_mask_obj == True))
                FP_obj = np.sum(np.logical_and(contact_pred_mask_obj == True, contact_gt_mask_obj == False))
                FN_obj = np.sum(np.logical_and(contact_pred_mask_obj == False, contact_gt_mask_obj == True))
            
                precision_hand = TP_hand / (TP_hand + FP_hand) if (TP_hand + FP_hand) > 0 else 0
                recall_hand = TP_hand / (TP_hand + FN_hand) if (TP_hand + FN_hand) > 0 else 0

                total_precision_hand += precision_hand
                total_recall_hand += recall_hand

                precision_obj = TP_obj / (TP_obj + FP_obj) if (TP_obj + FP_obj) > 0 else 0
                recall_obj = TP_obj / (TP_obj + FN_obj) if (TP_obj + FN_obj) > 0 else 0

                total_precision_obj += precision_obj
                total_recall_obj += recall_obj

                f1_score_hand = 2 * (precision_hand * recall_hand) / (precision_hand + recall_hand)
                f1_score_obj = 2 * (precision_obj * recall_obj) / (precision_obj + recall_obj)

                f1_score_hand = torch.tensor(f1_score_hand,dtype=torch.float32)
                f1_score_obj = torch.tensor(f1_score_obj,dtype=torch.float32)

                total_f1_score_hand += f1_score_hand.item()

                total_f1_score_obj += f1_score_obj.item()

    # print(np.array(lst_precision_hand).mean())
    # print(np.array(lst_recall_hand).mean())

    # print(np.array(lst_precision_obj).mean())
    # print(np.array(lst_recall_obj).mean())

    return  total_contact_loss_hand / count_taken, total_contact_loss_obj / count_taken,  total_f1_score_hand / count_taken, total_f1_score_obj / count_taken, total_precision_hand / count_taken, total_recall_hand / count_taken, total_precision_obj / count_taken, total_recall_obj / count_taken

def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                        spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))

def run_contactopt(args):
    """
    Actually run ContactOpt approach. Estimates target contact with DeepContact,
    then optimizes it. Performs random restarts if selected.
    Saves results to a pkl file.
    :param args: input settings
    """

    contactopt_args = args

    args_o = args

    print('Start loading traing_dataset')

    #train_dataset = CustomDataset('/scratch/minjay/NIA_EXTRACT_FOR_TRAINING','train')
    
    #train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=CustomDataset.collate_fn)

    print('Start loading test_dataset')
    # test_dataset = GANHAND('test')
    test_dataset = CustomDataset('/scratch/minjay/NIA_EXTRACT_FOR_TRAINING','test')

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=CustomDataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeepContactNet(model=args.model).to(device)

    args_o.work_dir = '/scratch/minjay/coma_taxonomy_prediction'
    args_o.data_fp = osp.join(args_o.work_dir, 'data', args_o.dataset)
    
    transform_fp = osp.join(args_o.data_fp, 'transform.pkl')

    if not osp.exists(transform_fp):
        print('Generating transform matrices...')
        mesh = Mesh(filename=template_fp)
        ds_factors = [4, 2, 2, 2]
        _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
        tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}
        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))
    else:
        with open(transform_fp, 'rb') as f: tmp = pickle.load(f, encoding='latin1')

    edge_index_list = [to_edge_index(adj).to(device) for adj in tmp['adj']]
    down_transform_list = [
        to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]

    C1 = AE(args_o.in_channels,
           args_o.out_channels,
           args_o.output_channels,
           args_o.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args_o.K).to(device)

    C2 = AE(args_o.in_channels,
           args_o.out_channels,
           args_o.output_channels,
           args_o.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args_o.K).to(device)

    C1_13 = AE(args_o.in_channels,
           args_o.out_channels,
           13,
           args_o.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args_o.K).to(device)

    C2_13 = AE(args_o.in_channels,
           args_o.out_channels,
           13,
           args_o.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args_o.K).to(device)


    import copy

    model_copy = copy.deepcopy(model)
    

    model.add_coma_model(C1,C2)
    model_copy.add_coma_model(C1_13,C2_13)
    model_copy2 = copy.deepcopy(model_copy)

    #coma_checkpoint = torch.load(COMA_PATH)

    # print('############## pretrained_model_loaded #################')

    # coma_model.eval()

    # for para in coma_model.parameters(): para.requires_grad = False

    # coma_model.load_state_dict(coma_checkpoint['model_state_dict_m1'])

    #model.load_state_dict(torch.load(MODEL_LOAD_PATH)['model_state_dict'])    
    
    ############################ args ############################

    epochs = 200
    lr_decay = 0.99
    decay_step = 1
    optimizer = 'SGD'
    weight_decay = 0.0005
    lr = 1e-3
    exp_name = EXP_NAME
    sgd_momentum = 0.9

    G = model.pointnet
    #conv1~4, lin1, mlp,  mano_forward

    opt_g  = torch.optim.SGD(list(model.pointnet.parameters()), lr=0.1,
                    momentum=sgd_momentum, weight_decay=0.0005, 
                    nesterov=True)

    opt_c1 = torch.optim.SGD(list(model.C1.parameters()), lr=1.0,
                    momentum=sgd_momentum, weight_decay=0.0005,
                    nesterov=True)

    opt_c2 = torch.optim.SGD(list(model.C2.parameters()), lr=1.0,
                    momentum=sgd_momentum, weight_decay=0.0005,
                    nesterov=True)
    ############################ args ############################

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
        
    param_lr_f1 = []
    for param_group in opt_c1.param_groups:
        param_lr_f1.append(param_group["lr"])

    param_lr_f2 = []
    for param_group in opt_c2.param_groups:
        param_lr_f2.append(param_group["lr"])

    max_iter = 200
    train_lr = 0.01

    optimizer_dict = {}
    optimizer_dict['max_iter'] = max_iter
    optimizer_dict['train_lr'] = train_lr

    optimizer_dict['param_lr_g'] = param_lr_g
    optimizer_dict['param_lr_f1'] = param_lr_f1
    optimizer_dict['param_lr_f2'] = param_lr_f2

    optimizer_dict['opt_g'] = opt_g
    optimizer_dict['opt_c1'] = opt_c1
    optimizer_dict['opt_c2'] = opt_c2
    
    from utils import utils, writer

    args = None

    writer = writer.Writer(exp_name,args)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,50], gamma=0.1) 


    ##################### TODO: LOAD DICT DATASET ########################
    test_50_dataset = DictDataset('test_dataset_sampled_nia_50_trans_rot.pickle')
    #test_50_dataset = DictDataset('test_dataset_sampled_nia_50_no_aug.pickle')
    
    test_50_dataloader = DataLoader(test_50_dataset, batch_size=1, shuffle=False, collate_fn=DictDataset.collate_fn)
    ##################### TODO: LOAD DICT DATASET ########################


    #run(model, train_dataloader, test_dataloader, epochs, optimizer_dict, writer, exp_name, device, args_o)

    #import pdb; pdb.set_trace()
    
    ### TRAINING END ###

    ####### TEST #########

    model_path = '/scratch/minjay/s2contact/Generator_pretrained_surfacecut/checkpoint_200.pt' 
    model_path_1 = '/scratch/minjay/s2contact/Final2Step_0.2_reweight_TT_10/checkpoint_200.pt' #/scratch/minjay/s2contact/Pre2StepJoCoR_0.0/checkpoint_200.pt'
    model_path_2 = '/scratch/minjay/s2contact/Final2Step_0.0_reweight_TT/checkpoint_200.pt' #/scratch/minjay/s2contact/Pre2StepCE_0.0/checkpoint_145.pt'

    #'/scratch/minjay/s2contact/NIA_AUG_TRANS_0.01_ROT_0.1_incep_classifier_0.3_rate_single_clamp_fix_remove_recon_onlyCE/checkpoint_060.pt'
    #'/scratch/minjay/s2contact/NIA_AUG_TRANS_0.01_ROT_0.1_incep_classifier_0.3_rate_2step_clamp_fix_remove_recon_onlyCE/checkpoint_030.pt'
    #'/scratch/minjay/s2contact/NIA_AUG_TRANS_0.01_ROT_0.1_incep_classifier_0.3_rate_single_clamp_fix_remove_recon_onlyCE/checkpoint_040.pt'
    pretrained_dict = torch.load(model_path)['model_state_dict']
    model_dict_1 = torch.load(model_path_1)['model_state_dict']
    model_dict_2 = torch.load(model_path_2)['model_state_dict']

    model.load_state_dict(pretrained_dict)
    model_copy.load_state_dict(model_dict_1)
    model_copy2.load_state_dict(model_dict_2)


    model.eval()
    model_copy.eval()
    model_copy2.eval()

    count_taken = 0
    batch_index = 0
    
    mano_run = ManoLayer(mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=True, ncomps=45, side='right', flat_hand_mean=True, center_idx=0).to(device)
    
    hand_contact_per_object  = [ 0 for i in range(30) ]
    hand_contact_per_object_lst  = [ list() for i in range(30) ]

    BP = False

    test_idx_lst = [573, 8666,12589, 606, 16539, 7677, 10560, 14111, 301, 8680, 10650, 4380, 6993, 6845, 5801, 7195, 7881, 978, 5964, 6394, 1190, 2104,5365, 7051, 16005, 7602, 9339, 11051, 12034, 6329, 6569, 7251, 7303, 8041, 3290, 5599, 7243, 15666, 12092, 1224,3503, 4029, 5779, 13459, 83, 1174, 2191, 5254, 14138, 11141, 12628, 13462, 16290, 16591]
    
    hand_contact_per_taxonomy_gt        = [ [ list() for i in range(28) ] for i in range(30)]
    hand_contact_per_taxonomy_baseline  = [ [ list() for i in range(28) ] for i in range(30)]
    hand_contact_per_taxonomy_model1    = [ [ list() for i in range(28) ] for i in range(30)]
    hand_contact_per_taxonomy_model2    = [ [ list() for i in range(28) ] for i in range(30)]

    total_l1_surface = [0,0,0]
    total_recall = [0,0,0]
    total_precision = [0,0,0]
    total_f1 = [0,0,0]
    total_recall_surface = [0,0,0]
    total_precision_surface = [0,0,0]
    total_f1_surface = [0,0,0]

    
    for sample in tqdm(test_dataloader) : # test_50_dataloader # test_dataloader
        
        # break

        
        # if not (sample['idx'].item() in test_idx_lst) :
        #      continue
            
        taxonomy_tmp = TAX_DICT_R[sample['taxonomy'][batch_index].item() + 1 ]
        modify_taxonomy = sample['taxonomy'][batch_index].item()
        obj_item =  sample['obj_id'].item() - 1 

        if not (TAX_DICT_R[sample['taxonomy'][batch_index].item() + 1 ] == 28  and  sample['obj_id'].item() == 19) :
               continue
        if BP :
            break

        if len(hand_contact_per_object_lst[obj_item]) >= 150 :

                if  hand_contact_per_object[obj_item] == 0 :

                    for i in hand_contact_per_object_lst : print(len(i),end =" ")

                hand_contact_per_object[obj_item] = 1

                if all(hand_contact_per_object) :
                    BP = True 
                    break

                continue
        
        if hand_contact_per_object_lst[obj_item].count(taxonomy_tmp) < 50 :

            hand_contact_per_object_lst[obj_item].append(taxonomy_tmp)

        else :
            continue

        sample = util.dict_to_device(sample, device)

        out = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))
        out_o = model_copy(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))
        out_o2 = model_copy2(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

        _, logit_1, _ = model_copy.forward_C1(sample['mano_verts'],out_o['contact_hand'])
        _, logit_2, _ = model_copy2.forward_C1(sample['mano_verts'],out_o2['contact_hand'])

        contact_hand = out['contact_hand'].to(device)
        contact_obj  = out['contact_obj'].to(device)

        aug_ho = HandObject()

        aug_ho.hand_verts   = sample['hand_verts'][batch_index].to(device) #* 1000
        aug_ho.closed_faces = mano_run.th_faces
        aug_ho.obj_verts    = sample['object_full_verts'][batch_index].to(device) # torch.Size([9892, 3]
        aug_ho.obj_faces    = sample['object_full_face'][batch_index].to(device) # [16384, 3]

        #aug_ho.hand_contact = sample['hand_contact'][0]
        aug_ho.obj_contact  = util.class_to_val(out['contact_obj'])[batch_index].unsqueeze(1).cpu()
        aug_ho.hand_contact = util.class_to_val(out['contact_hand'])[batch_index].unsqueeze(1).cpu()

        gt_ho = HandObject()

        gt_ho.hand_verts   = sample['hand_verts_gt'][batch_index].to(device) #* 1000
        gt_ho.closed_faces = mano_run.th_faces
        gt_ho.obj_verts    = sample['object_full_verts'][batch_index].to(device) # torch.Size([9892, 3]
        gt_ho.obj_faces    = sample['object_full_face'][batch_index].to(device) # [16384, 3]

        hand_pcd = Pointclouds(points=gt_ho.hand_verts.cpu().unsqueeze(0)*100)

        obj_mesh = Meshes(verts=sample['object_full_verts'].cpu()*100, faces=sample['object_full_face'].cpu())

        inter_dist = point_mesh_face_distance(obj_mesh, hand_pcd)

        inter_dist = np.array(inter_dist.unsqueeze(1))

        gt_ho.calc_dist_contact(hand=True, obj=True)

        # #aug_ho.hand_contact = sample['hand_contact'][0]
        # gt_ho.obj_contact  = util.class_to_val(sample['obj_contact' ])[batch_index].unsqueeze(1).cpu()
        # gt_ho.hand_contact = util.class_to_val(sample['hand_contact'])[batch_index].unsqueeze(1).cpu()
        
    
        ## AUG OBJ CONTACT ##
        obj_contact_target  = util.class_to_val(out['contact_obj']).unsqueeze(2).cpu()

        mesh_gt = Meshes(verts=sample['object_full_verts'].cpu(), faces=sample['object_full_face'].cpu())

        obj_contact_upscale = util.upscale_contact(mesh_gt, sample['obj_sampled_idx'].cpu(), obj_contact_target)
        
        aug_ho.obj_contact = obj_contact_upscale[batch_index]

        hand_mesh_aug, obj_mesh_aug = aug_ho.get_o3d_meshes(hand_contact=False,obj_contact=False)   

        hand_mesh_aug_pred, obj_mesh_aug_pred = aug_ho.get_o3d_meshes(hand_contact=True,obj_contact=True)            
        
        hand_mesh_gt, obj_mesh_gt = gt_ho.get_o3d_meshes(hand_contact=True,obj_contact=True)

        import torch.nn.functional as F
        print('taxonomy : ' + str(TAX_DICT_R[sample['taxonomy'][batch_index].item() + 1]))
        print('model1 : ',    str(TAX_DICT_R[F.softmax(logit_1, dim=1).argmax().item() + 1]))
        print('model2 : ',    str(TAX_DICT_R[F.softmax(logit_2, dim=1).argmax().item() + 1]))
        print(sample['idx'].item())
    
        ho_flat = HandObject()

        pose_tensor = torch.Tensor(np.concatenate((np.array([[-1.5,0,0]]), np.zeros((1,48))), axis=1)).to(device)
        hand_verts, _ = mano_run(pose_tensor)
        # 
        ho_flat.hand_verts   = (hand_verts/1000)[0].to(device) # [778, 3]
        ho_flat.closed_faces =  mano_run.th_faces # torch.Size([1538, 3])
        ho_flat.obj_verts    = sample['object_full_verts'][0].to(device) # torch.Size([9892, 3]
        ho_flat.obj_faces    = sample['object_full_face'][0].to(device) # [16384, 3]

        ho_flat.hand_contact =  aug_ho.hand_contact.cpu() # (778, 1)
        hand_00_mesh = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
        
        ho_flat.hand_contact = util.class_to_val(out_o['contact_hand'])[batch_index].unsqueeze(1).cpu()
        hand_00_mesh_o = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

        ho_flat.hand_contact = util.class_to_val(out_o2['contact_hand'])[batch_index].unsqueeze(1).cpu()
        hand_00_mesh_o2 = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

        inter_dist = np.clip(0,0.9,inter_dist)

        ho_flat.hand_contact = 1 - inter_dist # (778, 1)

        hand_gt_contact = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

        gt_hand_contact = torch.Tensor(1 - inter_dist)
        baseline_contact = aug_ho.hand_contact.cpu()
        model1_contact = util.class_to_val(out_o['contact_hand'])[batch_index].unsqueeze(1).cpu()
        model2_contact = util.class_to_val(out_o2['contact_hand'])[batch_index].unsqueeze(1).cpu()
        
        hand_contact_per_taxonomy_baseline[obj_item][modify_taxonomy].append(baseline_contact)
        hand_contact_per_taxonomy_model1[obj_item][modify_taxonomy].append(model1_contact)
        hand_contact_per_taxonomy_model2[obj_item][modify_taxonomy].append(model2_contact)
        hand_contact_per_taxonomy_gt[obj_item][modify_taxonomy].append(gt_hand_contact)

        obj_mesh_gt.translate((0.2, 0.2, 0))
        hand_mesh_gt.translate((0.2, 0.2, 0))
        hand_00_mesh_o.translate((-0.2, 0, 0))
        hand_00_mesh_o2.translate((-0.4, 0, 0))
        hand_00_mesh.translate((-0.6, 0, 0))
        hand_gt_contact.translate((-0.8, 0, 0))
        hand_mesh_aug_pred.translate((0.4, 0.4, 0))
        obj_mesh_aug_pred.translate((0.4, 0.4, 0))

        ###############################
        # gt_hand_contact_mask = np.array(gt_hand_contact > 0.3)
        # baseline_contact_mask = np.array(baseline_contact > 0.3)
        # model1_contact_mask = np.array(model1_contact > 0.3)
        # model2_contact_mask = np.array(model2_contact > 0.3)

        # lst_temp = [baseline_contact_mask, model1_contact_mask, model2_contact_mask]

        # for idx, pred_mask in enumerate(lst_temp) :

        #     TP_hand = np.sum(np.logical_and(pred_mask == True, gt_hand_contact_mask == True),axis=0)
        #     FP_hand = np.sum(np.logical_and(pred_mask == True, gt_hand_contact_mask == False),axis=0)
        #     FN_hand = np.sum(np.logical_and(pred_mask == False, gt_hand_contact_mask == True),axis=0)

        #     precision_hand = TP_hand / (TP_hand + FP_hand) if (TP_hand + FP_hand) > 0 else 0
        #     recall_hand = TP_hand / (TP_hand + FN_hand) if (TP_hand + FN_hand) > 0 else 0

        #     f1_score = 2 * (precision_hand * recall_hand) / (precision_hand + recall_hand)

        #     total_f1[idx] += f1_score
        #     total_precision[idx] += precision_hand
        #     total_recall[idx] += recall_hand

        #     TP_hand = np.sum(np.logical_and(pred_mask == True, gt_hand_contact_mask == True)[SURFACE],axis=0)
        #     FP_hand = np.sum(np.logical_and(pred_mask == True, gt_hand_contact_mask == False)[SURFACE],axis=0)
        #     FN_hand = np.sum(np.logical_and(pred_mask == False, gt_hand_contact_mask == True)[SURFACE],axis=0)

        #     precision_hand = TP_hand / (TP_hand + FP_hand) if (TP_hand + FP_hand) > 0 else 0
        #     recall_hand = TP_hand / (TP_hand + FN_hand) if (TP_hand + FN_hand) > 0 else 0

        #     f1_score = 2 * (precision_hand * recall_hand) / (precision_hand + recall_hand)

        #     total_f1_surface[idx] += f1_score
        #     total_precision_surface[idx] += precision_hand
        #     total_recall_surface[idx] += recall_hand

        # lst_temp = [baseline_contact, model1_contact, model2_contact]

        # for idx, pred_contact in enumerate(lst_temp) :

        #     pred_contact = np.clip(0.1,1.0,pred_contact)
            
        #     l1_loss = F.l1_loss(pred_contact[SURFACE],gt_hand_contact[SURFACE], reduction='mean')

        #     total_l1_surface[idx] += l1_loss.item()

        ###############################
            
        #geom_list = [hand_mesh_gt, obj_mesh_gt, hand_mesh_aug, obj_mesh_aug, hand_00_mesh, hand_00_mesh_o,hand_00_mesh_o2, hand_gt_contact, hand_mesh_aug_pred, obj_mesh_aug_pred]
        geom_list = [obj_mesh_gt, obj_mesh_aug, hand_00_mesh, hand_00_mesh_o,hand_00_mesh_o2, hand_gt_contact, obj_mesh_aug_pred]

        o3dv.draw_geometries(geom_list)

        continue
        # name = input("continue?")

        # if name == "N" :
        #     continue

        batch_size = sample['obj_sampled_idx'].shape[0]
        
        data_gpu = util.dict_to_device(sample, device)
        data = sample

        with torch.no_grad():
                
            out = model(data_gpu['hand_verts'], data_gpu['hand_feats'].to(torch.float32), data_gpu['obj_verts'], data_gpu['obj_feats'].to(torch.float32))
            out_o = model_copy(data_gpu['hand_verts'], data_gpu['hand_feats'].to(torch.float32), data_gpu['obj_verts'], data_gpu['obj_feats'].to(torch.float32))
            out_o2 = model_copy2(data_gpu['hand_verts'], data_gpu['hand_feats'].to(torch.float32), data_gpu['obj_verts'], data_gpu['obj_feats'].to(torch.float32))                
            
        out_lst = [out,out_o,out_o2] # out_o -> final , out -> baseline

        for out_temp in out_lst :

            hand_contact_target = util.class_to_val(out_temp['contact_hand']).unsqueeze(2)
            obj_contact_target  = util.class_to_val(out_temp['contact_obj']).unsqueeze(2)
        
            if contactopt_args.sharpen_thresh > 0: # If flag, sharpen contact
                print('Sharpening')
                obj_contact_target = util.sharpen_contact(obj_contact_target, slope=2, thresh=contactopt_args.sharpen_thresh)
                hand_contact_target = util.sharpen_contact(hand_contact_target, slope=2, thresh=contactopt_args.sharpen_thresh)

            if contactopt_args.rand_re > 1:    # If we desire random restarts
                
                print('Doing random optimization restarts')

                best_loss = torch.ones(batch_size) * 100000

                mtc_orig_rot = data_gpu['aug_rot'] # torch.Size([32, 1, 3])

                mtc_orig_trans = data_gpu['aug_trans'] # torch.Size([32, 1, 3])

                for re_it in range(contactopt_args.rand_re):

                    print(re_it)
                    # Add noise to hand translation and rotation

                    data_gpu['aug_rot']   = mtc_orig_rot.detach().clone()
                    data_gpu['aug_trans'] = mtc_orig_trans.detach().clone()

                    single_identity = np.eye(4)
                    hand_mTc_aug_matrix = np.repeat(single_identity[np.newaxis, :, :], batch_size, axis=0)

                    hand_mTc_aug_matrix[:, :3,  3] = data_gpu['aug_trans'][:,0,:].cpu()

                    data['hand_mTc_gt'] = torch.Tensor(hand_mTc_aug_matrix).detach().clone().to(device)

                    data_gpu['hand_mTc_aug'] = torch.Tensor(hand_mTc_aug_matrix).detach().clone().to(device)

                    random_rot_mat = pytorch3d.transforms.euler_angles_to_matrix(torch.randn((batch_size, 3), device=device) * contactopt_args.rand_re_rot / 180 * np.pi, 'ZYX')

                    data_gpu['hand_mTc_aug'][:, :3, :3] = torch.bmm(random_rot_mat, data_gpu['hand_mTc_aug'][:, :3, :3])
                    data_gpu['hand_mTc_aug'][:, :3, 3] += torch.randn((batch_size, 3), device=device) * contactopt_args.rand_re_trans

                    cur_result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=contactopt_args.n_iter, lr=contactopt_args.lr,
                                            w_cont_hand=contactopt_args.w_cont_hand, w_cont_obj=1, save_history=contactopt_args.vis, ncomps=contactopt_args.ncomps,
                                            w_cont_asym=contactopt_args.w_cont_asym, w_opt_trans=contactopt_args.w_opt_trans, w_opt_pose=contactopt_args.w_opt_pose,
                                            w_opt_rot=contactopt_args.w_opt_rot,
                                            caps_top=contactopt_args.caps_top, caps_bot=contactopt_args.caps_bot, caps_rad=contactopt_args.caps_rad,
                                            caps_on_hand=contactopt_args.caps_hand,
                                            contact_norm_method=contactopt_args.cont_method, w_pen_cost=contactopt_args.w_pen_cost,
                                            w_obj_rot=contactopt_args.w_obj_rot, pen_it=contactopt_args.pen_it)
                    
                    #import pdb; pdb.set_trace()

                    out_pose, out_tform, obj_rot_mat, opt_state = cur_result
                    #show_optimization(data, opt_state, hand_contact_target=hand_contact_target, obj_contact_target=obj_contact_target, is_video=True, vis_method=1)

                    if re_it == 0:
                        out_pose = torch.zeros_like(cur_result[0])
                        out_mTc = torch.zeros_like(cur_result[1])
                        obj_rot = torch.zeros_like(cur_result[2])
                        opt_state = cur_result[3]

                    loss_val = cur_result[3][-1]['loss']
                    for b in range(batch_size):
                        if loss_val[b] < best_loss[b]:
                            best_loss[b] = loss_val[b]
                            out_pose[b, :] = cur_result[0][b, :]
                            out_mTc[b, :, :] = cur_result[1][b, :, :]
                            obj_rot[b, :, :] = cur_result[2][b, :, :]
                            opt_state_best = opt_state        

            data_gpu['mesh_gt'] = Meshes(verts=data['object_full_verts'], faces=data['object_full_face']).to(device)

            obj_contact_upscale = util.upscale_contact(data_gpu['mesh_gt'], data_gpu['obj_sampled_idx'], obj_contact_target)

            all_data = list()
        
            for b in range(obj_contact_upscale.shape[0]):    # Loop over batch

                out_ho = HandObject()

                out_ho.load_from_batch(data['hand_beta'], out_pose.unsqueeze(1), out_mTc, data['hand_contact'], data['obj_contact'], data_gpu['mesh_gt'].cpu(), b, obj_rot=obj_rot)

                all_data.append({'out_ho': out_ho})

                out_ho.obj_verts = torch.Tensor(out_ho.obj_verts)
                out_ho.hand_verts = torch.Tensor(out_ho.hand_verts)
                out_ho.hand_contact = torch.Tensor(out_ho.hand_contact)
                out_ho.obj_contact = torch.Tensor(out_ho.obj_contact)
                out_ho.obj_faces = torch.Tensor(out_ho.obj_faces)

                vis_contactMap(out_ho)

                hand_mesh_gt, obj_mesh_gt = out_ho.get_o3d_meshes(hand_contact=True,obj_contact=True)

                ho_mesh = hand_mesh_gt + obj_mesh_gt

                import pdb; pdb.set_trace()

            #o3dio.write_triangle_mesh(f'figure/hand_obj_mesh_.obj',ho_mesh)
            

            # idx = sample['idx'].item()
            
            #taxonomy_str = str(TAX_DICT_R[sample['taxonomy'][batch_index].item() + 1]).zfill(2)

            #vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
            #vis.add_geometry(hand_gt_contact); vis.update_geometry(hand_gt_contact); vis.poll_events()
            #vis.update_renderer()
            #vis.capture_screen_image(f'export_testcase/hand_obj_mesh_{obj_item+1}_{taxonomy_str}_{idx}.png',do_render=True)
            #vis.destroy_window()

            #print(f'export_testcase/hand_obj_mesh_{taxonomy_str}_{idx}.obj')

            

            #import pdb; pdb.set_trace()
        
        with open('hand_contact_per_taxonomy_baseline.pkl', 'wb') as f: pickle.dump(hand_contact_per_taxonomy_baseline, f)
        with open('hand_contact_per_taxonomy_model1.pkl', 'wb') as f: pickle.dump(hand_contact_per_taxonomy_model1, f)
        with open('hand_contact_per_taxonomy_model2.pkl', 'wb') as f: pickle.dump(hand_contact_per_taxonomy_model2, f)
        with open('hand_contact_per_taxonomy_gt.pkl', 'wb') as f: pickle.dump(hand_contact_per_taxonomy_gt, f)

        with open('hand_contact_per_taxonomy_baseline.pkl', 'rb') as f: hand_contact_per_taxonomy_baseline = pickle.load(f)
        with open('hand_contact_per_taxonomy_model1.pkl', 'rb') as f: hand_contact_per_taxonomy_model1 = pickle.load(f)
        with open('hand_contact_per_taxonomy_model2.pkl', 'rb') as f: hand_contact_per_taxonomy_model2 = pickle.load(f)
        with open('hand_contact_per_taxonomy_gt.pkl', 'rb') as f: hand_contact_per_taxonomy_gt = pickle.load(f)

        hand_contact_per_taxonomy_baseline_tensor_stack = []
        hand_contact_per_taxonomy_model1_tensor_stack = []
        hand_contact_per_taxonomy_model2_tensor_stack = []
        hand_contact_per_taxonomy_gt_tesnor_stack = []

        hand_contact_per_taxonomy_baseline_type = []
        hand_contact_per_taxonomy_model1_type = []
        hand_contact_per_taxonomy_model2_type = []
        hand_contact_per_taxonomy_gt_type = []

        # 30 -> 28

        for obj_id, obj_contact_lst in enumerate(hand_contact_per_taxonomy_baseline) :

            for tax_id, taxonomy_contact_list in enumerate(obj_contact_lst) :
                
                if taxonomy_contact_list == [] :
                    continue

                stacked_tensor = torch.stack(taxonomy_contact_list, dim=0)
                hand_contact_per_taxonomy_baseline_tensor_stack.append(stacked_tensor)

                hand_contact_per_taxonomy_baseline_type.append( [obj_id,tax_id] )

        for obj_id, obj_contact_lst in enumerate(hand_contact_per_taxonomy_model1) :

            for tax_id, taxonomy_contact_list in enumerate(obj_contact_lst) :
                
                if taxonomy_contact_list == [] :
                    continue

                stacked_tensor = torch.stack(taxonomy_contact_list, dim=0)
                hand_contact_per_taxonomy_model1_tensor_stack.append(stacked_tensor)

                hand_contact_per_taxonomy_model1_type.append( [obj_id,tax_id] )

        for obj_id, obj_contact_lst in enumerate(hand_contact_per_taxonomy_model2) :

            for tax_id, taxonomy_contact_list in enumerate(obj_contact_lst) :
                
                if taxonomy_contact_list == [] :
                    continue

                stacked_tensor = torch.stack(taxonomy_contact_list, dim=0)
                hand_contact_per_taxonomy_model2_tensor_stack.append(stacked_tensor)

                hand_contact_per_taxonomy_model2_type.append( [obj_id,tax_id] )

        for obj_id, obj_contact_lst in enumerate(hand_contact_per_taxonomy_gt) :

            for tax_id, taxonomy_contact_list in enumerate(obj_contact_lst) :
                
                if taxonomy_contact_list == [] :
                    continue

                stacked_tensor = torch.stack(taxonomy_contact_list, dim=0)
                hand_contact_per_taxonomy_gt_tesnor_stack.append(stacked_tensor)

                hand_contact_per_taxonomy_gt_type.append( [obj_id,tax_id] )

        import pdb; pdb.set_trace()


        ho_flat = HandObject()
        mano_run = ManoLayer(mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=True, ncomps=45, side='right', flat_hand_mean=True, center_idx=0).to(device)
        pose_tensor = torch.Tensor(np.concatenate((np.array([[-2.2,0,0]]), np.zeros((1,48))), axis=1)).to(device)
        hand_verts, _ = mano_run(pose_tensor)
        # 
        ho_flat.hand_verts   = (hand_verts/1000)[0].to(device) # [778, 3]
        ho_flat.closed_faces =  mano_run.th_faces # torch.Size([1538, 3])

        ho_flat.obj_verts    = sample['object_full_verts'][0].to(device) # torch.Size([9892, 3]
        ho_flat.obj_faces    = sample['object_full_face'][0].to(device) # [16384, 3]

        # ho_flat.hand_contact =  torch.mean(hand_contact_per_taxonomy_pred_tensor_stack[0],0).cpu() * 2 # (778, 1)
        # vis_contactMap_hand(ho_flat)
        # ho_flat.hand_contact =  torch.mean(hand_contact_per_taxonomy_gt_tensor_stack[1],0).cpu() * 2 # (778, 1)
        # vis_contactMap_hand(ho_flat)
        # ho_flat.hand_contact =  torch.mean(hand_contact_per_taxonomy_aug_tensor_stack[0],0).cpu() * 2 # (778, 1)
        # vis_contactMap_hand(ho_flat)
        
        import pdb; pdb.set_trace()
        
        for i, contactMap in enumerate(hand_contact_per_taxonomy_baseline_tensor_stack) : 

            ho_flat.hand_contact =  torch.mean(contactMap,0).cpu() 
            hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
            vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
            vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/baseline_mean_{TAX_DICT_R[i+1]}.png',do_render=True)
            vis.destroy_window()

        for i, contactMap in enumerate(hand_contact_per_taxonomy_model1_tensor_stack) : 

            ho_flat.hand_contact =  torch.mean(contactMap,0).cpu() 
            hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
            vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
            vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/model1_mean_{TAX_DICT_R[i+1]}.png',do_render=True)
            vis.destroy_window()

        for i, contactMap in enumerate(hand_contact_per_taxonomy_model2_tensor_stack) : 

            ho_flat.hand_contact =  torch.mean(contactMap,0).cpu() 
            hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
            vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
            vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/model2_mean_{TAX_DICT_R[i+1]}.png',do_render=True)
            vis.destroy_window()

        for i, contactMap in enumerate(hand_contact_per_taxonomy_baseline_tensor_stack) : 

            ho_flat.hand_contact =  torch.var(contactMap,0).cpu() * 5
            hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
            vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
            vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/baseline_var_{TAX_DICT_R[i+1]}.png',do_render=True)
            vis.destroy_window()

        for i, contactMap in enumerate(hand_contact_per_taxonomy_model1_tensor_stack) : 

            ho_flat.hand_contact =  torch.var(contactMap,0).cpu() * 5
            hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
            vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
            vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/model1_var_{TAX_DICT_R[i+1]}.png',do_render=True)
            vis.destroy_window()

        for i, contactMap in enumerate(hand_contact_per_taxonomy_model2_tensor_stack) : 

            ho_flat.hand_contact =  torch.var(contactMap,0).cpu() * 5
            hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
            vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
            vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/model2_var_{TAX_DICT_R[i+1]}.png',do_render=True)
            vis.destroy_window()

        # torch.var(contactMap ,0).cpu()
        

    #final_run(model, test_dataloader, optimizer, scheduler, writer, exp_name, device)

    ##################### TODO: TRAINING CODE ########################

    
    ##################### TODO: LOAD CONTACT MAP ########################

    # incep_obj   = torch.load('100_incep_obj.pt')
    incep_hand  = torch.load('100_incep_hand.pt')
    # incep_obj_onehot_encoder   = torch.load('100_incep_onehot_encoder_obj.pt')
    incep_hand_onehot_encoder  = torch.load('100_incep_onehot_encoder_hand.pt')
    # incep_obj_onehot   = torch.load('100_incep_onehot_obj.pt')
    # incep_hand_onehot  = torch.load('100_incep_onehot_hand.pt')

    #hand_00_aug   = torch.load('nia_aug_00_mano_ce_hand.pt')
    #hand_30_aug   = torch.load('nia_aug_30_mano_ce_hand.pt')

    hand_00_aug   = torch.load('nia_00_ce_hand.pt')
    hand_30_aug   = torch.load('nia_aug_30_ce_hand.pt')

    hand_00   = torch.load('nia_30_ce_hand.pt')
    hand_30   = torch.load('nia_00_ce_hand.pt')
    
    ##################### TODO:  LOAD CONTACT MAP ########################
    

    ##################### TODO: COMAPRE CONTACT MAP ########################
    incep_hand_onehot_encoder_diff_incep_hand = incep_hand_onehot_encoder - incep_hand 
    # incep_hand_onehot_diff_incep_hand = incep_hand_onehot - incep_hand 
    # incep_hand_onehot_diff_incep_hand_onehot_encoder = incep_hand_onehot_encoder - incep_hand_onehot

    batch_absolute_1 = []
    # batch_absolute_2 = []
    # batch_absolute_3 = []

    for instance in incep_hand_onehot_encoder_diff_incep_hand : batch_absolute_1.append(torch.sum(torch.abs(instance)))

    # for instance in incep_hand_onehot_diff_incep_hand : batch_absolute_2.append(torch.sum(torch.abs(instance)))

    # for instance in incep_hand_onehot_diff_incep_hand_onehot_encoder : batch_absolute_3.append(torch.sum(torch.abs(instance)))
    
    batch_absolute_1 = torch.Tensor(batch_absolute_1)
    # batch_absolute_2 = torch.Tensor(batch_absolute_2)
    # batch_absolute_3 = torch.Tensor(batch_absolute_3)

    torch.topk(batch_absolute_1,20).indices

    # torch.topk(batch_absolute_2,20).indices
    # torch.topk(batch_absolute_3,20).indices

    # top_k = [ 619, 1079,  898, 1095,  910,  604, 1153, 1201,  453,  961,  632,  440,
    #      814,  326,  401, 1016, 1033, 1338, 1180, 1144,  623, 1066,   93, 1004,
    #      457, 1075,  578,  879,  907,  924,  763,  880, 1186,  951, 1030,  863,
    #     1185,  281,  761,  105, 1152,  531,  960,  839, 1085,  786,  338,   81,
    #      988,  120,  595,  816,  363, 1070, 1053, 1064,  424,  146,  208,  527,
    #      895, 1339, 1369,  434,  360,  575, 1176, 1388, 1154,  530,  268, 1268,
    #      501,   18, 1157,  941,    0,  455, 1058,  799, 1000,   57,  795,  914,
    #     1227,  357, 1228,  343, 1116, 1118,  136, 1224,  116, 1194,  617,  584,
    #     1087,  221,   41,  974]

    #top_k = [619, 1079,  898, 1095,  910,  604, 1153,  453, 1201,  961,  632,  440, 814,  326,  401, 1016, 1033, 1338, 1180, 1144]
    
    top_k = [ 268, 619]

    # for idx, data in enumerate(tqdm(test_50_dataloader)) : 
    #     if idx in top_k : 
    #         print(TAX_DICT_R[data['taxonomy'].item()+1])
    #         print(idx)

    ##################### TODO: COMAPRE CONTACT MAP ########################
    

    
    ##################### TODO: VISUALIZE CONTACT MAP USING .pt CONTACT MAP FIXME:CONDITION ######################## 


    # hand_00_aug   = torch.load('nia_aug_00_mano_ce_hand.pt')
    # hand_30_aug   = torch.load('nia_aug_30_mano_ce_hand.pt')

    # hand_00   = torch.load('nia_30_ce_hand.pt')
    # hand_30   = torch.load('nia_00_ce_hand.pt')
    
    '''
    for idx, data in enumerate(tqdm(test_50_dataloader)):

        #dict_keys(['taxonomy', 'aug_pose', 'aug_rot', 'aug_trans', 'hand_beta', 'obj_sampled_idx', 
        # 'hand_feats', 'obj_feats', 'hand_verts', 'obj_verts', 'hand_contact', 'obj_contact', 
        # 'hand_contact_aug', 'obj_normals_aug', 'object_full_face', 'object_full_verts'])
        if idx not in top_k :
            continue

        print(idx)

        data_gpu = util.dict_to_device(data, device)
        batch_size = data['obj_sampled_idx'].shape[0]

        batch_index = 0
        mano_run = ManoLayer(mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=True, ncomps=45, side='right', flat_hand_mean=True, center_idx=0).to(device)

        with torch.no_grad():
    
            # mesh = trimesh.load_mesh('/scratch/minjay/s2contact/hand.obj')

            # adjacency_matrix = mesh.vertex_adjacency_graph

            # sci_py = nx.to_scipy_sparse_array(adjacency_matrix).toarray()

            # laplacian = csgraph.laplacian(sci_py, normed=True)
            
            # sigma = 1

            # identity = np.eye(len(adjacency_matrix))

            # diffusion = np.linalg.inv(identity + sigma * laplacian)

            # incep_hand = diffusion @ np.array(incep_hand[idx][:,0].cpu()) @ diffusion.T
            # incep_hand_onehot_encoder = diffusion @ np.array(incep_hand_onehot_encoder[idx][:,0].cpu()) @ diffusion.T

            gt_ho = HandObject()

            gt_ho.hand_verts   = data['hand_verts'][batch_index].to(device) #* 1000
            gt_ho.closed_faces = mano_run.th_faces
            gt_ho.obj_verts    = data['object_full_verts'][0].to(device) # torch.Size([9892, 3]
            gt_ho.obj_faces    = data['object_full_face'][0].to(device) # [16384, 3]

            gt_ho.obj_contact  = data['obj_contact']
            gt_ho.hand_contact = data['hand_contact']
            
            hand_mesh_gt, obj_mesh_gt = gt_ho.get_o3d_meshes()

            ho_flat = HandObject()

            pose_tensor = torch.Tensor(np.concatenate((np.array([[-2.2,0,0]]), np.zeros((1,48))), axis=1)).to(device)
            hand_verts, _ = mano_run(pose_tensor)
            # 
            ho_flat.hand_verts   = (hand_verts/1000)[0].to(device) # [778, 3]
            ho_flat.closed_faces =  mano_run.th_faces # torch.Size([1538, 3])
            ho_flat.obj_verts    = data['object_full_verts'][0].to(device) # torch.Size([9892, 3]
            ho_flat.obj_faces    = data['object_full_face'][0].to(device) # [16384, 3]

            ho_flat.hand_contact =  data['hand_contact'][batch_index].cpu() # (778, 1)
            hand_gt = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

            ho_flat.hand_contact =  data['hand_contact_aug'][batch_index].cpu() # (778, 1)
            hand_aug = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
            
            #import pdb; pdb.set_trace()
            
            # if contactopt_args.sharpen_thresh > 0: # If flag, sharpen contact
            #     hand_00_map = util.sharpen_contact(hand_00[idx], slope=2, thresh=contactopt_args.sharpen_thresh)
            #     hand_30_map = util.sharpen_contact(hand_30[idx], slope=2, thresh=contactopt_args.sharpen_thresh)

            ho_flat.hand_contact =  incep_hand[idx].cpu() # (778, 1)
            incep_mesh = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

            ho_flat.hand_contact =  incep_hand_onehot_encoder[idx].cpu() #ncep_hand_onehot_encoder[idx].cpu() # (778, 1)
            incep_encoder_mesh = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

            # ho_flat.hand_contact =  hand_00_aug[idx].cpu() # (778, 1)
            # hand_00_aug_mesh = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

            # ho_flat.hand_contact =  hand_30_aug[idx].cpu() # (778, 1)
            # hand_30_aug_mesh = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

            # ho_flat.hand_contact =  incep_hand_onehot_encoder[batch_index].cpu() # (778, 1)
            # hand_incep_onehot_encode = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

            # hand_incep.translate((-0.2, 0, 0))
            # hand_incep_onehot.translate((-0.4, 0, 0))
            # hand_incep_onehot_encode.translate((-0.6, 0, 0))
            # hand_aug.translate((-0.8, 0, 0))
            # hand_gt.translate((-1.0, 0, 0))

            incep_mesh.translate((-0.2, 0, 0))
            incep_encoder_mesh.translate((-0.4, 0, 0))
            #hand_00_aug_mesh.translate((-0.6, 0, 0))
            #hand_30_aug_mesh.translate((-0.8, 0, 0))
            hand_aug.translate((-0.6, 0, 0))
            hand_gt.translate((-0.8, 0, 0))

            geom_list = [hand_mesh_gt,obj_mesh_gt,incep_mesh,incep_encoder_mesh,hand_gt,hand_aug]

            print(TAX_DICT_R[data['taxonomy'].item()+1])

            o3dv.draw_geometries(geom_list)


            ##################### TODO: COLLECTING CONTACT MAP PART 1 ######################## 
            # hand_contact_target = util.class_to_val(out['contact_hand']).unsqueeze(2)
            # obj_contact_target  = util.class_to_val(out['contact_obj']).unsqueeze(2)

            # hand_contact_pred.append(hand_contact_target)
            # obj_contact_pred.append(obj_contact_target)
            ##################### TODO: COLLECTING CONTACT MAP PART 1 ######################## 

    ##################### TODO: VISUALIZE CONTACT MAP USING .pt CONTACT MAP FIXME:GT ######################## 
    '''

    '''
    ##################### TODO: VISUALIZE CONTACT MAP USING .pt CONTACT MAP FIXME:GT ######################## 
    for idx, data in enumerate(tqdm(test_50_dataloader)):

        print(data['idx'])

        #if idx not in top_K : continue

        #dict_keys(['taxonomy', 'aug_pose', 'aug_rot', 'aug_trans', 'hand_beta', 'obj_sampled_idx', 
        # 'hand_feats', 'obj_feats', 'hand_verts', 'obj_verts', 'hand_contact', 'obj_contact', 
        # 'hand_contact_aug', 'obj_normals_aug', 'object_full_face', 'object_full_verts'])
    
        data_gpu = util.dict_to_device(data, device)
        batch_size = data['obj_sampled_idx'].shape[0]

        batch_index = 0
        mano_run = ManoLayer(mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=True, ncomps=45, side='right', flat_hand_mean=True, center_idx=0).to(device)

        with torch.no_grad():

            gt_ho = HandObject()

            gt_ho.hand_verts   = data['hand_verts'][batch_index].to(device) #* 1000
            gt_ho.closed_faces = mano_run.th_faces
            gt_ho.obj_verts    = data['object_full_verts'][0].to(device) # torch.Size([9892, 3]
            gt_ho.obj_faces    = data['object_full_face'][0].to(device) # [16384, 3]

            gt_ho.obj_contact  = data['obj_contact']
            gt_ho.hand_contact = data['hand_contact']
            
            hand_mesh_gt, obj_mesh_gt = gt_ho.get_o3d_meshes()

            ho_flat = HandObject()

            pose_tensor = torch.Tensor(np.concatenate((np.array([[-2.2,0,0]]), np.zeros((1,48))), axis=1)).to(device)
            hand_verts, _ = mano_run(pose_tensor)
            # 
            ho_flat.hand_verts   = (hand_verts/1000)[0].to(device) # [778, 3]
            ho_flat.closed_faces =  mano_run.th_faces # torch.Size([1538, 3])
            ho_flat.obj_verts    = data['object_full_verts'][0].to(device) # torch.Size([9892, 3]
            ho_flat.obj_faces    = data['object_full_face'][0].to(device) # [16384, 3]

            ho_flat.hand_contact =  data['hand_contact'][batch_index].cpu() # (778, 1)
            hand_gt = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

            # ho_flat.hand_contact =  data['hand_contact_aug'][batch_index].cpu() # (778, 1)
            # hand_aug = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
            
            #import pdb; pdb.set_trace()
            
            if contactopt_args.sharpen_thresh > 0: # If flag, sharpen contact
                hand_00_map = util.sharpen_contact(hand_00[idx], slope=2, thresh=contactopt_args.sharpen_thresh)
                hand_30_map = util.sharpen_contact(hand_30[idx], slope=2, thresh=contactopt_args.sharpen_thresh)

            ho_flat.hand_contact =  hand_00_map.cpu() # (778, 1)
            hand_00_mesh = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

            ho_flat.hand_contact =  hand_30_map.cpu() # (778, 1)
            hand_30_mesh = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

            # ho_flat.hand_contact =  incep_hand_onehot_encoder[batch_index].cpu() # (778, 1)
            # hand_incep_onehot_encode = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

            # hand_incep.translate((-0.2, 0, 0))
            # hand_incep_onehot.translate((-0.4, 0, 0))
            # hand_incep_onehot_encode.translate((-0.6, 0, 0))
            # hand_aug.translate((-0.8, 0, 0))
            # hand_gt.translate((-1.0, 0, 0))

            hand_00_mesh.translate((-0.2, 0, 0))
            hand_30_mesh.translate((-0.4, 0, 0))
            hand_gt.translate((-0.6, 0, 0))

            geom_list = [hand_mesh_gt,obj_mesh_gt,hand_00_mesh,hand_30_mesh,hand_gt]

            print(TAX_DICT_R[data['taxonomy'].item()+1])

            o3dv.draw_geometries(geom_list)


            ##################### TODO: COLLECTING CONTACT MAP PART 1 ######################## 
            # hand_contact_target = util.class_to_val(out['contact_hand']).unsqueeze(2)
            # obj_contact_target  = util.class_to_val(out['contact_obj']).unsqueeze(2)

            # hand_contact_pred.append(hand_contact_target)
            # obj_contact_pred.append(obj_contact_target)
            ##################### TODO: COLLECTING CONTACT MAP PART 1 ######################## 

    ##################### TODO: VISUALIZE CONTACT MAP USING .pt CONTACT MAP ######################## 
    '''
    
    '''
    ##################### TODO: COLLECTING CONTACT MAP PART 2 ######################## 
    hand_contact_pred_tensor = torch.Tensor().to(device)
    obj_contact_pred_tensor = torch.Tensor().to(device)
        
    for contact_list in hand_contact_pred :
        hand_contact_pred_tensor = torch.cat( (hand_contact_pred_tensor,contact_list) , dim=0)

    for contact_list in obj_contact_pred :
        obj_contact_pred_tensor = torch.cat(  (obj_contact_pred_tensor,contact_list) , dim=0)

    ## => save torch.pt
    ##################### TODO: COLLECTING CONTACT MAP PART 2 ######################## 
    '''
    
    '''
    #################### TODO: colecting hand_contact output #####################
    
    hand_contact_pred = []
    obj_contact_pred = []    

    for idx, data in enumerate(tqdm(test_50_dataloader)):
        
        data_gpu = util.dict_to_device(data, device)
        batch_size = data['obj_sampled_idx'].shape[0]

        with torch.no_grad():
            
            if LATENT_CONDITION :
                out = model.forward_onehot(data_gpu['hand_verts'], data_gpu['hand_feats'].to(torch.float32), data_gpu['obj_verts'], data_gpu['obj_feats'].to(torch.float32))
                #out = model.forward_onehot_w_mano(data_gpu['hand_verts'], data_gpu['hand_feats'].to(torch.float32), data_gpu['obj_verts'], data_gpu['obj_feats'].to(torch.float32))
            else :            
                out = model(data_gpu['hand_verts'], data_gpu['hand_feats'].to(torch.float32), data_gpu['obj_verts'], data_gpu['obj_feats'].to(torch.float32))

            hand_contact_target = util.class_to_val(out['contact_hand']).unsqueeze(2)
            obj_contact_target  = util.class_to_val(out['contact_obj']).unsqueeze(2)

            hand_contact_pred.append(hand_contact_target)
            obj_contact_pred.append(obj_contact_target)

           
            # for idx, class_idx in enumerate(data['taxonomy']) :
                
            #     hand_contact_per_taxonomy_pred[class_idx].append(hand_contact_target[idx])
            #     hand_contact_per_taxonomy_aug[class_idx].append(data_gpu['hand_contact_aug'][idx])
            #     hand_contact_per_taxonomy_gt[class_idx].append(data_gpu['hand_contact'][idx])


        hand_contact_pred_tensor = torch.Tensor().to(device)
        obj_contact_pred_tensor = torch.Tensor().to(device)
            
        for contact_list in hand_contact_pred :
            hand_contact_pred_tensor = torch.cat( (hand_contact_pred_tensor,contact_list) , dim=0)

        for contact_list in obj_contact_pred :
            obj_contact_pred_tensor = torch.cat(  (obj_contact_pred_tensor,contact_list) , dim=0)

    import pdb; pdb.set_trace()
    
    # torch.save(hand_contact_pred_tensor,"nia_00_ce_hand.pt")
    # torch.save(obj_contact_pred_tensor, "nia_00_ce_obj.pt")

    # torch.save(hand_contact_pred_tensor,"nia_aug_30_ce_hand.pt")
    # torch.save(obj_contact_pred_tensor, "nia_aug_30_ce_obj.pt")
    '''

    
    ## => save torch.pt

    #################### TODO: colecting hand_contact output #####################

    '''
    ### TODO: MAKING TEST LOADER STATIC DATASET ###
    
    test_dataset_data_pkl = {}
    testset_idx = 0

    hand_contact_per_taxonomy_pred = [ list() for i in range(28) ]
    hand_contact_per_taxonomy_gt   = [ list() for i in range(28) ]
    hand_contact_per_taxonomy_aug  = [ list() for i in range(28) ]
    hand_contact_per_taxonomy_check  = [ 0 for i in range(28) ]
    BP = False

    for idx, data in enumerate(tqdm(test_dataloader)):

        if BP :
            break
        
        data_gpu = util.dict_to_device(data, device)
        batch_size = data['obj_sampled_idx'].shape[0]

        with torch.no_grad():

            for idx, class_idx in enumerate(data['taxonomy']) :
                

                if len(hand_contact_per_taxonomy_pred[class_idx]) >= 50 :

                    if  hand_contact_per_taxonomy_check[class_idx] == 0 :

                        for i in hand_contact_per_taxonomy_pred : print(len(i),end =" ")

                    hand_contact_per_taxonomy_check[class_idx] = 1

                    if all(hand_contact_per_taxonomy_check) :
                        BP = True 
                        break


                    continue

                temp_dict = {}
                
                for dict_key in data.keys() :
                    temp_dict[dict_key] = data[dict_key][idx].cpu()

                test_dataset_data_pkl[testset_idx] = temp_dict
                testset_idx += 1
                hand_contact_per_taxonomy_pred[class_idx].append(0)

    with open('test_dataset_sampled_nia_50_trans_rot.pickle', 'wb') as handle:
        pickle.dump(test_dataset_data_pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
            #hand_contact_per_taxonomy_pred[class_idx].append(hand_contact_target[idx])
            #hand_contact_per_taxonomy_aug[class_idx].append(data['hand_contact_aug'][idx])
            #hand_contact_per_taxonomy_gt[class_idx].append(data['hand_contact'][idx])

    ### TODO:  MAKING TEST LOADER STATIC DATASET ###
    '''

    '''
    #### TODO: GET weight ######
   
    args_o.work_dir = '/scratch/minjay/coma_taxonomy_prediction'
    args_o.data_fp = osp.join(args_o.work_dir, 'data', args_o.dataset)
    
    transform_fp = osp.join(args_o.data_fp, 'transform.pkl')

    if not osp.exists(transform_fp):
        print('Generating transform matrices...')
        mesh = Mesh(filename=template_fp)
        ds_factors = [4, 2, 2, 2]
        _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
        tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}
        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))
    else:
        with open(transform_fp, 'rb') as f: tmp = pickle.load(f, encoding='latin1')

    edge_index_list = [to_edge_index(adj).to(device) for adj in tmp['adj']]
    down_transform_list = [
        to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]

    coma_model = AE(args_o.in_channels,
           args_o.out_channels,
           args_o.output_channels,
           args_o.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args_o.K).to(device)


    coma_check_point = '/scratch/minjay/coma_taxonomy_prediction/out/corp_learning_in_ch_4_latent_128_noise_30/checkpoints/checkpoint_120.pt'
    #coma_check_point = '/scratch/minjay/coma_taxonomy_prediction/out/corp_learning_in_ch_4_forget20/checkpoints/checkpoint_120.pt'
    #coma_check_point = '/scratch/minjay/coma_taxonomy_prediction/out/corp_learning_in_ch_4_forget00/checkpoints/checkpoint_120.pt'
    
    coma_checkpoint = torch.load(coma_check_point)

    print('############## pretrained_model_loaded #################')
    coma_model.load_state_dict(coma_checkpoint['model_state_dict_m1'])

    coma_model.eval()

    coma_dataset = DictDataset_PLUG_COMA('test_dataset_sampled_nia_50.pickle')
    coma_dataloader = DataLoader(coma_dataset, batch_size=128, shuffle=False, collate_fn=DictDataset_PLUG_COMA.collate_fn)

    num_samples = len(coma_dataloader.dataset)
    losses = torch.zeros(num_samples)

    n = 0
    paths = []
    '''

    '''
    num_samples = len(train_dataloader.dataset)
    losses = torch.zeros(num_samples)

    n = 0
    paths = []

    with torch.no_grad():

        for batch_idx, data in enumerate(tqdm(train_dataloader)):

            data_gpu = util.dict_to_device(data, device)
            batch_size = data['obj_sampled_idx'].shape[0]

            x_mesh = data_gpu['mano_verts'].to(device)
            x_contact_gt = data_gpu['hand_contact'].to(device)

            x = torch.cat((x_mesh, x_contact_gt),dim=2)

            out, logit = coma_model(x)

            gt_taxonomy = [ int(tax) for tax in data['taxonomy'] ] 
            labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

            CE = nn.CrossEntropyLoss(reduction='none')

            loss_ce = CE(logit, labels)             

            for b in range(x.size(0)): 
                losses[n] = loss_ce[b]
                paths.append(data['idx'][b])
                n += 1

    losses = (losses - losses.min()) / (losses.max() - losses.min())

    losses = losses.reshape(-1, 1)

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)

    gmm.fit(losses)

    prob = gmm.predict_proba(losses)

    prob = prob[:, gmm.means_.argmin()]

    prob_dict = {}

    for count, dict_idx in enumerate(paths) : 
        prob_dict[dict_idx.item()] = prob[count]

    with open('train_00_error_weight.pickle', 'wb') as handle: pickle.dump(prob_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    
    #### TODO: GET weight ######

    '''
    #### TODO: Visualize coma weight ######

    with torch.no_grad():

        for batch_idx, data in enumerate(tqdm(coma_dataloader)):

            data_gpu = util.dict_to_device(data, device)
            batch_size = data['obj_sampled_idx'].shape[0]

            x_mesh = data_gpu['mano_verts'].to(device)
            x_contact_gt = data_gpu['hand_contact'].to(device)

            x = torch.cat((x_mesh, x_contact_gt),dim=2)

            out, logit = coma_model(x)

            gt_taxonomy = [ int(tax) for tax in data['taxonomy'] ] 

            labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

            CE = nn.CrossEntropyLoss(reduction='none')

            loss_ce = CE(logit, labels)             

            for b in range(x.size(0)): 
                losses[n] = loss_ce[b]
                paths.append(data['index'][b])
                n += 1

    losses = (losses - losses.min()) / (losses.max() - losses.min())

    losses = losses.reshape(-1, 1)

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)

    gmm.fit(losses)

    prob = gmm.predict_proba(losses)

    prob = prob[:, gmm.means_.argmin()]

    hand_contact_per_taxonomy_gt   = [ list() for i in range(28) ]

    weight_sum = [ 0 for i in range(28) ]
    weight_count = [ 0 for i in range(28) ]
    weight_mean = [ 0 for i in range(28) ]

    with torch.no_grad():

        for batch_idx, data in enumerate(tqdm(coma_dataloader)):

            data_gpu = util.dict_to_device(data, device)
            batch_size = data['obj_sampled_idx'].shape[0]

            for idx, class_idx in enumerate(data['taxonomy']) :
                
                weight = prob[data_gpu['index'][idx]]

                if weight < 0.5 :
                     continue

                hand_contact_per_taxonomy_gt[class_idx].append(data_gpu['hand_contact'][idx] * weight)
                weight_sum[class_idx]   += weight
                weight_count[class_idx] += 1

                ###### PROB
    
    for i in range(28) :
        if weight_count[i]  == 0 :
            weight_mean[i] = 0
            continue
        weight_mean[i] = weight_sum[i] / weight_count[i] 

    for tmp in hand_contact_per_taxonomy_gt :
        print(len(tmp), end = " ")

    hand_contact_per_taxonomy_gt_tensor_stack = []

    for contact_list in hand_contact_per_taxonomy_gt :

        if contact_list == [] :
            hand_contact_per_taxonomy_gt_tensor_stack.append(torch.Tensor([0]))
            continue

        stacked_tensor = torch.stack(contact_list, dim=0)
        hand_contact_per_taxonomy_gt_tensor_stack.append(stacked_tensor)

    ho_flat = HandObject()
    mano_run = ManoLayer(mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=True, ncomps=45, side='right', flat_hand_mean=True, center_idx=0).to(device)
    pose_tensor = torch.Tensor(np.concatenate((np.array([[-2.2,0,0]]), np.zeros((1,48))), axis=1)).to(device)
    hand_verts, _ = mano_run(pose_tensor)
    # 
    ho_flat.hand_verts   = (hand_verts/1000)[0].to(device) # [778, 3]
    ho_flat.closed_faces =  mano_run.th_faces # torch.Size([1538, 3])

    ho_flat.obj_verts    = data['object_full_verts'][0].to(device) # torch.Size([9892, 3]
    ho_flat.obj_faces    = data['object_full_face'][0].to(device) # [16384, 3]

    # for i, contactMap in enumerate(hand_contact_per_taxonomy_gt_tensor_stack) : 

    #     ho_flat.hand_contact =  torch.mean(contactMap / weight_mean[i] ,0).cpu() * 2 
    #     hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
    #     vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
    #     vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
    #     vis.update_renderer()
    #     vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/gt_mean_{i}.png',do_render=True)
    #     vis.destroy_window()

    for i, contactMap in enumerate(hand_contact_per_taxonomy_gt_tensor_stack) : 

        if contactMap.shape[0] == 1 :
            continue

        ho_flat.hand_contact =  torch.var(contactMap ,0).cpu() * 3
        hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
        vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
        vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/gt_var_{i}.png',do_render=True)
        vis.destroy_window()


    import pdb; pdb.set_trace()

    
    #### TODO: Visualize coma weight ######
    '''

    '''

    # for i in hand_contact_per_taxonomy_pred : print(len(i))

    hand_contact_per_taxonomy_pred_tensor_stack = []
    hand_contact_per_taxonomy_aug_tensor_stack = []
    hand_contact_per_taxonomy_gt_tensor_stack = []

    for contact_list in hand_contact_per_taxonomy_pred :
        stacked_tensor = torch.stack(contact_list, dim=0)
        hand_contact_per_taxonomy_pred_tensor_stack.append(stacked_tensor)

    for contact_list in hand_contact_per_taxonomy_aug :
        stacked_tensor = torch.stack(contact_list, dim=0)
        hand_contact_per_taxonomy_aug_tensor_stack.append(stacked_tensor)

    for contact_list in hand_contact_per_taxonomy_gt :
        stacked_tensor = torch.stack(contact_list, dim=0)
        hand_contact_per_taxonomy_gt_tensor_stack.append(stacked_tensor)

    ho_flat = HandObject()
    mano_run = ManoLayer(mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=True, ncomps=45, side='right', flat_hand_mean=True, center_idx=0).to(device)
    pose_tensor = torch.Tensor(np.concatenate((np.array([[-2.2,0,0]]), np.zeros((1,48))), axis=1)).to(device)
    hand_verts, _ = mano_run(pose_tensor)
    # 
    ho_flat.hand_verts   = (hand_verts/1000)[0].to(device) # [778, 3]
    ho_flat.closed_faces =  mano_run.th_faces # torch.Size([1538, 3])
    ho_flat.hand_contact =  hand_contact_target[0].cpu() # (778, 1)

    ho_flat.obj_verts    = data['object_full_verts'][0].to(device) # torch.Size([9892, 3]
    ho_flat.obj_faces    = data['object_full_face'][0].to(device) # [16384, 3]

    # ho_flat.hand_contact =  torch.mean(hand_contact_per_taxonomy_pred_tensor_stack[0],0).cpu() * 2 # (778, 1)
    # vis_contactMap_hand(ho_flat)
    # ho_flat.hand_contact =  torch.mean(hand_contact_per_taxonomy_gt_tensor_stack[1],0).cpu() * 2 # (778, 1)
    # vis_contactMap_hand(ho_flat)
    # ho_flat.hand_contact =  torch.mean(hand_contact_per_taxonomy_aug_tensor_stack[0],0).cpu() * 2 # (778, 1)
    # vis_contactMap_hand(ho_flat)

    for i, contactMap in enumerate(hand_contact_per_taxonomy_aug_tensor_stack) : 

        ho_flat.hand_contact =  torch.mean(contactMap,0).cpu() * 2
        hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
        vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
        vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/aug_mean_{i}.png',do_render=True)
        vis.destroy_window()

    for i, contactMap in enumerate(hand_contact_per_taxonomy_gt_tensor_stack) : 

        ho_flat.hand_contact =  torch.mean(contactMap,0).cpu() * 2
        hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
        vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
        vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/gt_mean_{i}.png',do_render=True)
        vis.destroy_window()

    for i, contactMap in enumerate(hand_contact_per_taxonomy_pred_tensor_stack) : 

        ho_flat.hand_contact =  torch.mean(contactMap,0).cpu() * 2
        hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
        vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
        vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/pred_mean_{i}.png',do_render=True)
        vis.destroy_window()

    for i, contactMap in enumerate(hand_contact_per_taxonomy_aug_tensor_stack) : 

        ho_flat.hand_contact =  torch.var(contactMap,0).cpu() * 3
        hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
        vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
        vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/aug_var_{i}.png',do_render=True)
        vis.destroy_window()

    for i, contactMap in enumerate(hand_contact_per_taxonomy_gt_tensor_stack) : 

        ho_flat.hand_contact =  torch.var(contactMap,0).cpu() * 3
        hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
        vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
        vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/gt_var_{i}.png',do_render=True)
        vis.destroy_window()

    for i, contactMap in enumerate(hand_contact_per_taxonomy_pred_tensor_stack) : 

        ho_flat.hand_contact =  torch.var(contactMap,0).cpu() * 3
        hand = ho_flat.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)
        vis = o3d.visualization.Visualizer(); vis.create_window(visible=False) 
        vis.add_geometry(hand); vis.update_geometry(hand); vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'{IMAGE_SAVE_PATH}/pred_var_{i}.png',do_render=True)
        vis.destroy_window()

    ### TODO: USIING CLASSIFIER ###
    '''

    
    ### TODO: Optimization Process ###
    
    all_data = list()

    for idx, data in enumerate(tqdm(test_50_dataloader)):
        
        data_gpu = util.dict_to_device(data, device)
        batch_size = data['obj_sampled_idx'].shape[0]

        with torch.no_grad():
            
            if LATENT_CONDITION :
                out = model.forward_onehot(data_gpu['hand_verts'], data_gpu['hand_feats'].to(torch.float32), data_gpu['obj_verts'], data_gpu['obj_feats'].to(torch.float32))
            else :            
                out = model(data_gpu['hand_verts'], data_gpu['hand_feats'].to(torch.float32), data_gpu['obj_verts'], data_gpu['obj_feats'].to(torch.float32))

            hand_contact_target = util.class_to_val(out['contact_hand']).unsqueeze(2)
            obj_contact_target  = util.class_to_val(out['contact_obj']).unsqueeze(2)

        # dict_keys(['taxonomy', 'aug_pose', 'aug_rot', 'aug_trans', 'hand_beta', 
        # 'obj_sampled_idx', 'hand_feats', 'obj_feats', 'hand_verts', 'obj_verts', 
        # 'hand_contact', 'obj_contact', 'hand_contact_aug', 'obj_normals_aug', 'object_full_face', 'object_full_verts'])
    
        if contactopt_args.sharpen_thresh > 0: # If flag, sharpen contact
            print('Sharpening')
            obj_contact_target = util.sharpen_contact(obj_contact_target, slope=2, thresh=contactopt_args.sharpen_thresh)
            hand_contact_target = util.sharpen_contact(hand_contact_target, slope=2, thresh=contactopt_args.sharpen_thresh)

        if contactopt_args.rand_re > 1:    # If we desire random restarts
            
            # mtc_orig = data_gpu['hand_mTc_aug'].detach().clone()

            # self.hand_pose, self.hand_beta, self.hand_rot, self.hand_trans

            print('Doing random optimization restarts')

            best_loss = torch.ones(batch_size) * 100000

            mtc_orig_rot = data_gpu['aug_rot'] # torch.Size([32, 1, 3])

            mtc_orig_trans = data_gpu['aug_trans'] # torch.Size([32, 1, 3])

            for re_it in range(contactopt_args.rand_re):

                print(re_it)
                # Add noise to hand translation and rotation

                #data_gpu['hand_mTc_aug'] = mtc_orig.detach().clone()
                data_gpu['aug_rot']   = mtc_orig_rot.detach().clone()
                data_gpu['aug_trans'] = mtc_orig_trans.detach().clone()

                single_identity = np.eye(4)
                hand_mTc_aug_matrix = np.repeat(single_identity[np.newaxis, :, :], batch_size, axis=0)

                #rotation_matrix = pytorch3d.transforms.euler_angles_to_matrix(data_gpu['aug_rot'][:,0,:].cpu() * contactopt_args.rand_re_rot / 180 * np.pi, 'ZYX')

                #hand_mTc_aug_matrix[:, :3, :3] = rotation_matrix
                hand_mTc_aug_matrix[:, :3,  3] = data_gpu['aug_trans'][:,0,:].cpu()

                data['hand_mTc_gt'] = torch.Tensor(hand_mTc_aug_matrix).detach().clone().to(device)

                data_gpu['hand_mTc_aug'] = torch.Tensor(hand_mTc_aug_matrix).detach().clone().to(device)

                random_rot_mat = pytorch3d.transforms.euler_angles_to_matrix(torch.randn((batch_size, 3), device=device) * contactopt_args.rand_re_rot / 180 * np.pi, 'ZYX')

                data_gpu['hand_mTc_aug'][:, :3, :3] = torch.bmm(random_rot_mat, data_gpu['hand_mTc_aug'][:, :3, :3])
                data_gpu['hand_mTc_aug'][:, :3, 3] += torch.randn((batch_size, 3), device=device) * contactopt_args.rand_re_trans

                cur_result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=contactopt_args.n_iter, lr=contactopt_args.lr,
                                           w_cont_hand=contactopt_args.w_cont_hand, w_cont_obj=1, save_history=contactopt_args.vis, ncomps=contactopt_args.ncomps,
                                           w_cont_asym=contactopt_args.w_cont_asym, w_opt_trans=contactopt_args.w_opt_trans, w_opt_pose=contactopt_args.w_opt_pose,
                                           w_opt_rot=contactopt_args.w_opt_rot,
                                           caps_top=contactopt_args.caps_top, caps_bot=contactopt_args.caps_bot, caps_rad=contactopt_args.caps_rad,
                                           caps_on_hand=contactopt_args.caps_hand,
                                           contact_norm_method=contactopt_args.cont_method, w_pen_cost=contactopt_args.w_pen_cost,
                                           w_obj_rot=contactopt_args.w_obj_rot, pen_it=contactopt_args.pen_it)
                
                #import pdb; pdb.set_trace()

                out_pose, out_tform, obj_rot_mat, opt_state = cur_result
                #show_optimization(data, opt_state, hand_contact_target=hand_contact_target, obj_contact_target=obj_contact_target, is_video=True, vis_method=1)

                if re_it == 0:
                    out_pose = torch.zeros_like(cur_result[0])
                    out_mTc = torch.zeros_like(cur_result[1])
                    obj_rot = torch.zeros_like(cur_result[2])
                    opt_state = cur_result[3]

                loss_val = cur_result[3][-1]['loss']
                for b in range(batch_size):
                    if loss_val[b] < best_loss[b]:
                        best_loss[b] = loss_val[b]
                        out_pose[b, :] = cur_result[0][b, :]
                        out_mTc[b, :, :] = cur_result[1][b, :, :]
                        obj_rot[b, :, :] = cur_result[2][b, :, :]
                        opt_state_best = opt_state

        else:
            result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=contactopt_args.n_iter, lr=contactopt_args.lr,
                                   w_cont_hand=contactopt_args.w_cont_hand, w_cont_obj=1, save_history=contactopt_args.vis, ncomps=contactopt_args.ncomps,
                                   w_cont_asym=contactopt_args.w_cont_asym, w_opt_trans=contactopt_args.w_opt_trans, w_opt_pose=contactopt_args.w_opt_pose,
                                   w_opt_rot=contactopt_args.w_opt_rot,
                                   caps_top=contactopt_args.caps_top, caps_bot=contactopt_args.caps_bot, caps_rad=contactopt_args.caps_rad,
                                   caps_on_hand=contactopt_args.caps_hand,
                                   contact_norm_method=contactopt_args.cont_method, w_pen_cost=contactopt_args.w_pen_cost,
                                   w_obj_rot=contactopt_args.w_obj_rot, pen_it=contactopt_args.pen_it)
            out_pose, out_mTc, obj_rot, opt_state = result

        

        data_gpu['mesh_gt'] = Meshes(verts=data['object_full_verts'], faces=data['object_full_face']).to(device)

        obj_contact_upscale = util.upscale_contact(data_gpu['mesh_gt'], data_gpu['obj_sampled_idx'], obj_contact_target)
    
        for b in range(obj_contact_upscale.shape[0]):    # Loop over batch

            
            #gt_ho = HandObject()
            #in_ho = HandObject()
            out_ho = HandObject()
            #gt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_gt'], b)
            #in_ho.load_from_batch(data['hand_beta_aug'], data['hand_pose_aug'], data['hand_mTc_aug'], hand_contact_target, obj_contact_upscale, data['mesh_aug'], b)
            
            out_ho.load_from_batch(data['hand_beta'], out_pose.unsqueeze(1), out_mTc, data['hand_contact'], data['obj_contact'], data_gpu['mesh_gt'].cpu(), b, obj_rot=obj_rot)
            # out_ho.calc_dist_contact(hand=True, obj=True)

            #all_data.append({'gt_ho': gt_ho, 'in_ho': in_ho, 'out_ho': out_ho})
            all_data.append({'out_ho': out_ho})

            out_ho.obj_verts = torch.Tensor(out_ho.obj_verts)
            out_ho.hand_verts = torch.Tensor(out_ho.hand_verts)
            out_ho.hand_contact = torch.Tensor(out_ho.hand_contact)
            out_ho.obj_contact = torch.Tensor(out_ho.obj_contact)
            out_ho.obj_faces = torch.Tensor(out_ho.obj_faces)

            vis_contactMap(out_ho)

            out_ho.get_o3d_meshes()

            # hand_mesh_gt, obj_mesh_gt = gt_ho.get_o3d_meshes()
            # hand_gt.translate((-1.0, 0, 0))

            # geom_list = [hand_mesh_gt,obj_mesh_gt,hand_incep,hand_incep_onehot,hand_incep_onehot_encode,hand_gt,hand_aug]

            # o3dv.draw_geometries(geom_list)


        if contactopt_args.vis:
            show_optimization(data, opt_state_best,  hand_contact_target.detach().cpu().numpy(), obj_contact_upscale.detach().cpu().numpy(), is_video=True, vis_method=1)
            # show_optimization(data, opt_state, hand_contact_target.detach().cpu().numpy(), obj_contact_upscale.detach().cpu().numpy(),
            #                   is_video=contactopt_args.video, vis_method=contactopt_args.vis_method)
            

        if idx >= contactopt_args.partial > 0:   # Speed up for eval
            break
    

    #out_file = 'data/optimized_{}.pkl'.format(contactopt_args.split+'_'+contactopt_args.model)
    #print('Saving to {}. Len {}'.format(out_file, len(all_data)))
    #pickle.dump(all_data, open(out_file, 'wb'))
    

if __name__ == '__main__':
    util.hack_filedesciptor()
    args = arguments.run_contactopt_parse_args()

    if args.split == 'aug':     # Settings defaults for Perturbed ContactPose
        defaults = {'lr': 0.01,
                    'n_iter': 250,
                    'w_cont_hand': 2.0,
                    'sharpen_thresh': -1,
                    'ncomps': 15,
                    'w_cont_asym': 2,
                    'w_opt_trans': 0.3,
                    'w_opt_rot': 1.0,
                    'w_opt_pose': 1.0,
                    'caps_rad': 0.001,
                    'cont_method': 0,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 600,
                    'pen_it': 0,
                    'rand_re': 10,
                    'rand_re_trans': 0.01,
                    'rand_re_rot': 5,
                    'w_obj_rot': 0,
                    'vis_method': 1}
    elif args.split == 'im' or args.split == 'demo':    # Settings defaults for image-based pose estimates
        defaults = {'lr': 0.01,
                    'n_iter': 250,
                    'w_cont_hand': 2.5,
                    'sharpen_thresh': -1,
                    'ncomps': 15,
                    'w_cont_asym': 2,
                    'w_opt_trans': 0.3,
                    'w_opt_rot': 1,
                    'w_opt_pose': 1.0,
                    'caps_rad': 0.001,
                    'cont_method': 0,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 320,
                    'pen_it': 0,
                    # 'rand_re': 8,
                    'rand_re': 1,
                    'rand_re_trans': 0.02,
                    'rand_re_rot': 5,
                    'w_obj_rot': 0,
                    'vis_method': 1}
    elif args.split == 'fine':  # Settings defaults for small-scale refinement
        defaults = {'lr': 0.003,
                    'n_iter': 250,
                    'w_cont_hand': 0,
                    'sharpen_thresh': 0.3,
                    'ncomps': 15,
                    'w_cont_asym': 4,
                    'w_opt_trans': 0.03,
                    'w_opt_rot': 1.0,
                    'w_opt_pose': 1.0,
                    'caps_rad': 0.001,
                    'cont_method': 5,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 600,
                    'pen_it': 0,
                    'rand_re': 1,
                    'rand_re_trans': 0.00,
                    'rand_re_rot': 0,
                    'w_obj_rot': 0,
                    'vis_method': 5}

    for k in defaults.keys():   # Override arguments that have not been manually set with defaults
        if vars(args)[k] is None:
            vars(args)[k] = defaults[k]

    print(args)

    torch.multiprocessing.set_start_method('spawn')

    start_time = time.time()
    run_contactopt(args)
    print('Elapsed time:', time.time() - start_time)

