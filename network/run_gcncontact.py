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

TAXONOMY_CONDITION = True
LATENT_CONDITION = True
GAN_HAND = True
#EXP_NAME = 'NIA_AUG_TRAINING_taxonomy_condition_encoder_joint_fix'
EXP_NAME = 'GANHAND_AUG_TRANS_0.05_ROT_0.1_incep_onehot_econder'
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

IMAGE_SAVE_PATH = 'contactmap_sync_weighted_30_th_50/'

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

COMMON_KEY = [1, 2, 3, 4, 7, 9, 11, 12, 14, 17, 18, 19, 20, 22, 26, 27, 28, 29, 31, 33]

if GAN_HAND :
    TAX_DICT =   {1:1,2:2,3:3,4:4,6:5,7:6,8:7,9:8,11:9,12:10,14:11,15:12,17:13,18:14,19:15,20:16,22:17,26:18,27:19,28:20,29:21,31:22,33:23}

else :
    TAX_DICT   = {1:1,2:2,3:3,4:4,5:5,7:6,9:7,10:8,11:9,12:10,13:11,14:12,16:13,17:14,18:15,19:16,20:17,22:18,23:19,24:20,25:21,26:22,27:23,28:24,29:25,30:26,31:27,33:28}
    TAX_DICT_R = {1:1,2:2,3:3,4:4,5:5,6:7,7:9,8:10,9:11,10:12,11:13,12:14,13:16,14:17,15:18,16:19,17:20,18:22,19:23,20:24,21:25,22:26,23:27,24:28,25:29,26:30,27:31,28:33}

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


def vis_contactMap(gt_ho):

    hand, obj = gt_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)

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

        ho_gt.hand_verts   = ho_info['hand_verts'].to(self.device) #* 1000
        ho_gt.closed_faces = self.mano_faces
        ho_gt.obj_verts    = ho_info['obj_verts'].to(self.device)
        ho_gt.obj_faces    = obj_faces.to(self.device)
        ho_gt.hand_joints  = ho_info['hand_joints'].to(self.device)

        sample['object_full_face'] = ho_gt.obj_faces 
        sample['object_full_verts'] = ho_gt.obj_verts

        gt_contact_hand = ho_info['hand_contact']

        ho_gt.obj_contact  = ho_info['obj_contact']
        ho_gt.hand_contact  = gt_contact_hand

        ho_gt.hand_beta = hand_dict['hand_beta']

        ho_gt.hand_pose = (hand_dict['hand_pose']).mm(self.pca_inv)[:,:self.ncomps]

        ho_gt.hand_rot  = hand_dict['hand_mTc'] # Rotation

        sample['taxonomy']     = TAX_DICT[int(self.taxonomyfixed[int(file_name.split('.')[0])])] - 1

        sample['hand_joints_gt'] = ho_gt.hand_joints
        sample['hand_pose_gt'] = hand_dict['hand_pose'][0]

        aug_trans=0.05
        aug_rot=0.1 #.1
        aug_pca=0.3 #.2 # 3

        aug_t = np.random.randn(3) * aug_trans
        aug_p = np.concatenate((np.random.randn(3) * aug_rot, np.random.randn(self.ncomps) * aug_pca)).astype(np.float32)

        ho_gt.calc_dist_contact(hand=True, obj=True)

        ho_aug = HandObject()

        ho_aug.load_from_ho_new(ho_gt, self.mano_run, self.device, aug_p, aug_t)

        ######## ---- ########
        sample['aug_pose']  = ho_aug.hand_pose
        sample['aug_rot']   = ho_aug.hand_rot
        sample['aug_trans'] = ho_aug.hand_trans.float()
        sample['hand_beta'] = ho_aug.hand_beta

        sample['idx'] = idx
        sample['weight'] = self.weight_dict[idx]

        ######## ---- ########

        ho_aug.closed_faces = self.mano_faces

        #ho_aug.calc_dist_contact(hand=True, obj=True)
        #vis_contactMap_two(ho_gt,ho_aug)

        #vis_contactMap_surface(ho_gt,SURFACE)

        obj_sampled_idx = np.random.randint(0, len(ho_gt.obj_verts), 2048)

        ######## ---- ########
        sample['obj_sampled_idx'] = obj_sampled_idx
        ######## ---- ########

        hand_feats_aug, obj_feats_aug = ho_aug.generate_pointnet_features(obj_sampled_idx,self.device)

        ######## ---- ########
        sample['obj_normals_aug'] = ho_aug.obj_normals
        sample['hand_verts_gt']   = ho_gt.hand_verts
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

class CustomDataset_NonAug(Dataset):
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

        ho_gt.hand_verts   = ho_info['hand_verts'].to(self.device) #* 1000
        ho_gt.closed_faces = self.mano_faces
        ho_gt.obj_verts    = ho_info['obj_verts'].to(self.device)
        ho_gt.obj_faces    = obj_faces.to(self.device)
        ho_gt.hand_joints  = ho_info['hand_joints'].to(self.device)

        sample['object_full_face'] = ho_gt.obj_faces 
        sample['object_full_verts'] = ho_gt.obj_verts

        gt_contact_hand = ho_info['hand_contact']
        gt_contact_hand[SURFACE] = 0.05

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
        aug_pca=0.3 #.2 # 3

        aug_t = np.random.randn(3) * aug_trans
        aug_p = np.concatenate((np.random.randn(3) * aug_rot, np.random.randn(self.ncomps) * aug_pca)).astype(np.float32)

        ho_gt.calc_dist_contact(hand=True, obj=True)

        ho_aug = HandObject()

        ho_aug.load_from_ho_new(ho_gt, self.mano_run, self.device, aug_p, aug_t)

        ######## ---- ########
        sample['aug_pose']  = ho_aug.hand_pose
        sample['aug_rot']   = ho_aug.hand_rot
        sample['aug_trans'] = ho_aug.hand_trans.float()
        sample['hand_beta'] = ho_aug.hand_beta

        sample['idx'] = idx
        sample['weight'] = self.weight_dict[idx]

        ######## ---- ########

        ho_aug.closed_faces = self.mano_faces

        obj_sampled_idx = np.random.randint(0, len(ho_gt.obj_verts), 2048)

        ######## ---- ########
        sample['obj_sampled_idx'] = obj_sampled_idx
        ######## ---- ########

        hand_feats_aug, obj_feats_aug = ho_aug.generate_pointnet_features(obj_sampled_idx,self.device)

        ######## ---- ########
        sample['obj_normals_aug'] = ho_aug.obj_normals
        sample['hand_verts_gt']   = ho_gt.hand_verts
        ######## ---- ########
        
        sample['hand_feats']   = hand_feats_aug
        sample['obj_feats']    = obj_feats_aug


        obj_verts = ho_aug.obj_verts[obj_sampled_idx, :] 
        obj_contact = ho_gt.obj_contact[obj_sampled_idx, :] 

        sample['hand_verts']   = ho_aug.hand_verts
        sample['obj_verts']    = obj_verts
        
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

    '''
    
    def __getitem__(self, idx):

        ## ORIGINAL GT TRAINING

        sample = {}

        ho_gt = HandObject()
        
        file_name = self.file_names[idx]

        #import pdb; pdb.set_trace()

        ho_info    = self.dict[int(file_name.split('.')[0])]
        obj_faces  = self.obj_faces_dict[int(ho_info['obj_ids'])]

        ho_gt.hand_verts   = ho_info['hand_verts'].to(self.device)
        ho_gt.closed_faces = self.mano_faces
        ho_gt.obj_verts    = ho_info['obj_verts'].to(self.device)
        ho_gt.obj_faces    = obj_faces.to(self.device)
        ho_gt.hand_joints  = ho_info['hand_joints'].to(self.device)

        ho_gt.obj_contact  = ho_info['obj_contact']
        ho_gt.hand_contact  = ho_info['hand_contact']

        #ho_gt.calc_dist_contact(hand=True, obj=True)

        sample['taxonomy']     = TAX_DICT[int(self.taxonomyfixed[int(file_name.split('.')[0])])] - 1

        print(sample['taxonomy'])

        vis_contactMap(ho_gt)

        #import pdb; pdb.set_trace()

        # NEW CONTACT of Predicited Mesh

        #ho_gt.calc_dist_contact(hand=True, obj=True)

        #vis_contactMap(ho_gt)
                
        obj_sampled_idx = np.random.randint(0, len(ho_gt.obj_verts), 2048)

        hand_feats_aug, obj_feats_aug = ho_gt.generate_pointnet_features(obj_sampled_idx,self.device)

        obj_verts = ho_gt.obj_verts[obj_sampled_idx, :] 
        obj_contact = ho_gt.obj_contact[obj_sampled_idx, :] 

        # has_nan = np.isnan(ho_gt.obj_contact)

        # print('---------')

        # if has_nan.any():
        #     print('nan obj')

        #     has_nan_obj = np.isnan(ho_gt.obj_contact)
        #     ## 뽑아보기
        #     indices = np.where(has_nan_obj)[0]

        #     print(indices)
        #     print(ho_info['obj_ids'])

        # has_nan = np.isnan(ho_gt.hand_contact)
        # if has_nan.any():
        #     print('nan hand')

        #     has_nan_hand = np.isnan(ho_gt.hand_contact)
        #     ## 뽑아보기
        #     indices = np.where(has_nan_hand)[0]

        #     print(indices)
        #     print('hand')

            #import open3d as o3d

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(ho_gt.hand_verts)
            # colors = np.expand_dims(np.ones(778), -1)
            # colors = np.concatenate([colors, colors, colors], axis=1)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud('hand.ply', pcd)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(ho_gt.obj_verts)
            # colors = np.expand_dims(np.ones(ho_gt.obj_verts.shape[0]), -1)
            # colors[indices] = 0
            # colors = np.concatenate([colors, colors, colors], axis=1)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud('obj.ply', pcd)

            #import pdb; pdb.set_trace()

        # has_nan = np.isnan(ho_gt.hand_verts)
        # if has_nan.any():
        #     print('nan hand')
        #     import pdb; pdb.set_trace()

        sample['hand_verts']   = ho_gt.hand_verts
        sample['obj_verts']    = obj_verts
        sample['hand_feats']   = hand_feats_aug
        sample['obj_feats']    = obj_feats_aug


        sample['taxonomy']     = ho_info['taxonomy']
        sample['hand_contact'] = ho_gt.hand_contact
        sample['obj_contact']  = obj_contact

        # hand_verts
        # obj_verts
        #
        #
        #
        #



        return sample  # Dummy label, replace as needed
    '''

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

    '''
    def __getitem__(self, idx):

        sample = {}

        ho_gt = HandObject()

        ho_gt = self.dict[idx]['ho_gt']

        #ho_aug = self.dict[idx]['ho_aug']

        # Testing input as a real GT #
        ##############################
        # Testing input as a real GT #

        #ho_gt.calc_dist_contact(hand=True, obj=True)

        #vis_contactMap(ho_gt)

        #obj_sampled_idx = self.dict[idx]['obj_sampled_idx']
                
        obj_sampled_idx = np.random.randint(0, len(ho_gt.obj_verts), 2048)

        #hand_feats_aug, obj_feats_aug = ho_gt.generate_pointnet_features(obj_sampled_idx,self.device)

        ho_gt.hand_verts   = torch.Tensor(ho_gt.hand_verts).to(self.device)
        ho_gt.obj_verts    = torch.Tensor(ho_gt.obj_verts).to(self.device)
        ho_gt.obj_faces    = torch.Tensor(ho_gt.obj_faces).to(self.device)
        ho_gt.hand_joints  = torch.Tensor(ho_gt.hand_joints).to(self.device)

        ## Testing ##
        


        hand_feats_aug, obj_feats_aug = ho_gt.generate_pointnet_features(obj_sampled_idx,self.device)

        obj_verts = ho_gt.obj_verts[obj_sampled_idx, :] 
        obj_contact = ho_gt.obj_contact[obj_sampled_idx, :] 


        sample['hand_verts']   = ho_gt.hand_verts.to(self.device)
        sample['obj_verts']    = obj_verts.to(self.device)
        sample['hand_feats']   = hand_feats_aug.to(self.device)
        sample['obj_feats']    = obj_feats_aug.to(self.device)
        sample['taxonomy']     = 0
        sample['hand_contact'] = ho_gt.hand_contact
        sample['obj_contact']  = obj_contact

        return sample
    '''
    
    
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

def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer, exp_name, device, args_o):

    train_losses, test_losses = [], []

    s_writer = SummaryWriter(f'runs/{exp_name}')

    epoch_division_idx = 0

    f1_score_hand = 0
    f1_score_obj = 0


    ## LOAD COMA MODEL ##

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
    
    # coma_checkpoint = torch.load(COMA_PATH)

    # print('############## pretrained_model_loaded #################')
    # coma_model.load_state_dict(coma_checkpoint['model_state_dict_m1'])

    # coma_model.eval()

    # for para in coma_model.parameters(): 
    #     para.requires_grad = False

    # model.add_coma_model(coma_model)
    ## LOAD COMA MODEL ##

    for epoch in range(1, epochs + 1):

        t = time.time()

        train(model, optimizer, train_loader, device, writer, s_writer, epoch, epochs)

        #train_w_coma(model, coma_model, optimizer, train_loader, device, writer, s_writer, epoch, epochs)
                
        t_duration = time.time() - t

        test_hand_contact_loss, test_obj_contact_loss , f1_score_hand, f1_score_obj, precision_hand, recall_hand, precision_obj, recall_obj = test(model, test_loader, epoch, 0, 20, device)

        #total_precision_hand / count_taken, total_recall_hand / count_taken, total_precision_obj / count_taken, total_recall_obj / count_taken

        scheduler.step()

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
            writer.save_checkpoint(model, optimizer, epoch, exp_name)
        # if epoch % 10 == 0 :
        #     writer.save_checkpoint(model, optimizer, scheduler, epoch)
    
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


def train(model, optimizer, train_loader, device, writer, s_writer, epoch, epochs):

    model.train()

    total_contact_loss_hand = 0
    total_contact_loss_obj = 0

    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)

    criterion_hand = torch.nn.NLLLoss(weight=bin_weights)
    criterion_obj = torch.nn.NLLLoss(weight=bin_weights)

    # a = iter(train_loader)

    #count_taken = 0
    ## 아래에 Count_taken 수정해줘야 함

    # for i in tqdm(range(start_idx,end_idx)) :

    #     sample = next(a)
    iteration = 200

    len_train_loader = len(train_loader)

    mok, left = divmod(len_train_loader, iteration)

    for idx, sample in enumerate(tqdm(train_loader)) :

        optimizer.zero_grad()

        if LATENT_CONDITION :
            out = model.forward_onehot(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

        else :            
            out = model(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))

        #import pdb; pdb.set_trace()

        contact_hand = out['contact_hand'].to(device)
        contact_obj  = out['contact_obj'].to(device)

        gt_contact_map_hand = val_to_class_0_9(sample['hand_contact']).squeeze(2).long().to(device)
        gt_contact_map_obj  = val_to_class_0_9(sample['obj_contact']).squeeze(2).long().to(device)

        contact_classify_loss_hand = criterion_hand(contact_hand.permute(0, 2, 1), gt_contact_map_hand)
        contact_classify_loss_obj = criterion_obj(contact_obj.permute(0, 2, 1), gt_contact_map_obj)

        #gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 
        #gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

        # taxonomy_loss = criterion_taxonomy(pred_taxonomy,(gt_taxonomy - 1).to(device))

        #total_taxonomy_loss += taxonomy_loss

        # pred 랑 x 랑 compare 해보기 

        # l1_loss = F.l1_loss(out[:,:,:3], sample['mano_verts'], reduction='mean')
        # total_l1_loss += l1_loss

        loss =  contact_classify_loss_hand + contact_classify_loss_obj

        loss.backward()

        total_contact_loss_hand += contact_classify_loss_hand.item()
        total_contact_loss_obj  += contact_classify_loss_obj.item()

        loss.item()

        optimizer.step()

        if idx % iteration == 0 and idx > 0 :
            
            info = {
                'epochs' : epochs,
                'current_epoch': (epoch-1) * mok + idx // iteration ,
                'train_contact_loss_obj' : total_contact_loss_obj / iteration,
                'train_contact_loss_hand' : total_contact_loss_hand / iteration,
                'train_taxonomy_loss' : 0,
                'joint_loss' : 0,
                'maon_pose_loss' : 0,
                'mok': mok 
            }
            
            writer.print_info_train(info)
            writer.s_writer_train(info,s_writer,info['current_epoch'])

            total_contact_loss_hand = 0
            total_contact_loss_obj = 0
        
    #count_taken = len(train_loader)
        
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

            #gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

            #gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

            #acc_taxonomy = (pred_taxonomy.cpu().data.numpy().argmax(1) == (gt_taxonomy-1).numpy()).mean()

            #total_taxonomy_acr += acc_taxonomy.item()

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
    
    with torch.no_grad():
        
        for i in tqdm(range(start_idx,end_idx)) :

            sample = next(a)
            count_taken += 1
            
            if LATENT_CONDITION :
                out = model.forward_onehot(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))
                #out = model.forward_onehot_w_mano(sample['hand_verts'], sample['hand_feats'].to(torch.float32), sample['obj_verts'], sample['obj_feats'].to(torch.float32))
            else :            
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

            #gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

            #gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

            #acc_taxonomy = (pred_taxonomy.cpu().data.numpy().argmax(1) == (gt_taxonomy-1).numpy()).mean()

            #total_taxonomy_acr += acc_taxonomy.item()
                        
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

    # print('Running split', args.split)

    # print('Start loading train_dataset')
    train_dataset = GANHAND('train')
    #train_dataset = CustomDataset('/scratch/minjay/NIA_EXTRACT_FOR_TRAINING','train')
    # train_dataset = DexYCB('train')
    # print('train_dataset loaded')
    # # train_dataset = ContactPose('/scratch/minjay/ContactOpt/data','train')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=GANHAND.collate_fn)
    #train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=CustomDataset.collate_fn)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=DexYCB.collate_fn)

    # print('Start loading test_dataset')
    test_dataset = GANHAND('test')
    #test_dataset = CustomDataset('/scratch/minjay/NIA_EXTRACT_FOR_TRAINING','test')
    # test_dataset = DexYCB('test')
    # print('test_dataset loaded')
    # #test_dataset = ContactPose('/scratch/minjay/ContactOpt/data','test')

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=GANHAND.collate_fn)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True,collate_fn=CustomDataset.collate_fn)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=DexYCB.collate_fn)

    #shuffle = args.vis or args.partial > 0
    #print('Shuffle:', shuffle)
    #test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=6, collate_fn=ContactDBDataset.collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeepContactNet(model=args.model).to(device)

    coma_checkpoint = torch.load(COMA_PATH)

    print('############## pretrained_model_loaded #################')

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

    # coma_model.eval()

    # for para in coma_model.parameters(): para.requires_grad = False

    # coma_model.load_state_dict(coma_checkpoint['model_state_dict_m1'])

    # model.add_coma_model(coma_model)

    #model.load_state_dict(torch.load(MODEL_LOAD_PATH)['model_state_dict'])    
    
    ############################ args ############################

    #epochs = 20
    epochs = 100
    lr_decay = 0.99
    decay_step = 1
    optimizer = 'Adam'
    weight_decay = 0.0005
    lr = 1e-3

    #exp_name = 'training_contactPose_GT_Testing'
    #exp_name = 'training_contactPose_AUG_Testing'
    #exp_name = 'training_NIA_AUG_Testing'
    #exp_name = 'training_NIA_GT_Testing'
    #exp_name = 'DeepContact_Testing'
    exp_name = EXP_NAME
    #exp_name = 'NIA_AUG_TRAINING_onehot_global_feature'

    ############################ args ############################

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # optimizer = torch.optim.Adam(model.parameters(),
        #                          lr=lr,
        #                          weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay,momentum=0.9)
    else:
        raise RuntimeError('Use optimizers of SGD or Adam')

    from utils import utils, writer

    args = None

    writer = writer.Writer(exp_name,args)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             decay_step,
    #                                             gamma=lr_decay)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) 
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,25], gamma=0.1) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,75], gamma=0.1) 

    #optimizer.load_state_dict(torch.load(MODEL_LOAD_PATH)['optimizer_state_dict'])    
    #scheduler.load_state_dict(torch.load(MODEL_LOAD_PATH)['scheduler_state_dict'])    
    ##################### TODO: TRAINING CODE ########################

    run(model, train_dataloader, test_dataloader, epochs, optimizer, scheduler, writer, exp_name, device, args_o)
    #run(model, test_dataloader, test_dataloader, epochs, optimizer, scheduler, writer, exp_name, device, args_o)
    #final_run(model, test_dataloader, optimizer, scheduler, writer, exp_name, device)

    ##################### TODO: TRAINING CODE ########################


    ##################### TODO: LOAD DICT DATASET ########################
    test_50_dataset = DictDataset('test_dataset_sampled_nia_50.pickle')
    #test_50_dataset = DictDataset('test_dataset_sampled_nia_50_no_aug.pickle')
    
    test_50_dataloader = DataLoader(test_50_dataset, batch_size=1, shuffle=False, collate_fn=DictDataset.collate_fn)
    ##################### TODO: LOAD DICT DATASET ########################

    
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

            pose_tensor = torch.Tensor(np.concatenate((np.array([[-2,0,0]]), np.zeros((1,48))), axis=1)).to(device)
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

            pose_tensor = torch.Tensor(np.concatenate((np.array([[-2,0,0]]), np.zeros((1,48))), axis=1)).to(device)
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

    '''
    ## => save torch.pt

    #################### TODO: colecting hand_contact output #####################

    
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

    with open('test_dataset_sampled_nia_50_aug_hard.pickle', 'wb') as handle:
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
    pose_tensor = torch.Tensor(np.concatenate((np.array([[-2,0,0]]), np.zeros((1,48))), axis=1)).to(device)
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
    pose_tensor = torch.Tensor(np.concatenate((np.array([[-2,0,0]]), np.zeros((1,48))), axis=1)).to(device)
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

    '''
    ### TODO: Optimization Process ###
    
    all_data = list()

    for idx, data in enumerate(tqdm(test_50_dataloader)):

        # if idx not in [632,401,326,604,453] :
        #     continue
        
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
    '''

if __name__ == '__main__':
    util.hack_filedesciptor()
    args = arguments.run_contactopt_parse_args()

    if args.split == 'aug':     # Settings defaults for Perturbed ContactPose
        defaults = {'lr': 0.01,
                    'n_iter': 250,
                    'w_cont_hand': 2.0,
                    'sharpen_thresh': -1,
                    'ncomps': 45, #15,
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
                    'rand_re': 20,
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

