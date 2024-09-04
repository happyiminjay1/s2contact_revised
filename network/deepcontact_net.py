# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import network.pointnet as pointnet
import torch.nn.functional as F
from pytorch3d import ops, transforms
import network.util as util
import network.dgcnn as dgcnn
import network.incep_edgeconv as incep
import network.incep_edgeconv_onehot as incep_onehot
import network.incep_edgeconv_encoder_onehot as incep_onehot_encoder
import network.incep_edgeconv_encoder_onehot_mano as incep_onehot_encoder_mano
import network.incep_classifier as incep_classifier

class DeepContactNet(nn.Module):
    def __init__(self, model='pointnet',normalize_pts=True):
        super(DeepContactNet, self).__init__()

        if model == 'pointnet':
            self.pointnet = pointnet.Net()
        elif model =='dgcnn':
            self.pointnet = dgcnn.Net()
        elif model =='incep':
            self.pointnet = incep.Net()
        elif model =='incep_onehot':
            self.pointnet = incep_onehot.Net()
        elif model == 'incep_onehot_encoder':
            self.pointnet = incep_onehot_encoder.Net()
        elif model == 'incep_onehot_encoder_mano':
            self.pointnet = incep_onehot_encoder_mano.Net()
        elif model == 'incep_classifier':
            self.pointnet = incep_classifier.Net()
            
        self.model_type_name = model
        self.normalize_pts = normalize_pts

        pointnet_total_params = sum(p.numel() for p in self.pointnet.parameters() if p.requires_grad)
        print('Backbone params: {}'.format(pointnet_total_params))

    def add_coma_model(self, C1, C2) :
        self.C1 = C1
        self.C2 = C2

    def forward(self, hand_verts, hand_feats, obj_verts, obj_feats):
        device = hand_verts.device
        batch_size = hand_verts.shape[0]
        out = dict()

        if self.normalize_pts:
            tform = self.get_normalizing_tform(hand_verts, obj_verts)
            hand_verts = util.apply_tform(tform, hand_verts)
            obj_verts = util.apply_tform(tform, obj_verts)
            # util.vis_pointcloud(obj_verts, hand_verts)  # View pointnet input

        x, pos, batch = self.verts_to_pointcloud(hand_verts, hand_feats, obj_verts, obj_feats)

        contact_batched = self.pointnet(x, pos, batch)

        contact = contact_batched.view(batch_size, hand_verts.shape[1] + obj_verts.shape[1], 10)

        out['contact_hand'] = contact[:, :hand_verts.shape[1], :]
        out['contact_obj'] = contact[:, hand_verts.shape[1]:, :]

        return out

    def forward_mano(self, hand_verts, hand_feats, obj_verts, obj_feats):
        device = hand_verts.device
        batch_size = hand_verts.shape[0]
        out = dict()

        if self.normalize_pts:
            tform = self.get_normalizing_tform(hand_verts, obj_verts)
            hand_verts = util.apply_tform(tform, hand_verts)
            obj_verts = util.apply_tform(tform, obj_verts)
            # util.vis_pointcloud(obj_verts, hand_verts)  # View pointnet input

        x, pos, batch = self.verts_to_pointcloud(hand_verts, hand_feats, obj_verts, obj_feats)

        contact_batched, mano_param = self.pointnet(x, pos, batch)

        contact = contact_batched.view(batch_size, hand_verts.shape[1] + obj_verts.shape[1], 10)

        out['contact_hand'] = contact[:, :hand_verts.shape[1], :]
        out['contact_obj'] = contact[:, hand_verts.shape[1]:, :]
        out['mano_param'] = mano_param

        return out
    
    def forward_onehot(self, hand_verts, hand_feats, obj_verts, obj_feats):

        device = hand_verts.device
        batch_size = hand_verts.shape[0]
        out = dict()

        if self.normalize_pts:
            tform = self.get_normalizing_tform(hand_verts, obj_verts)
            hand_verts = util.apply_tform(tform, hand_verts)
            obj_verts = util.apply_tform(tform, obj_verts)
            # util.vis_pointcloud(obj_verts, hand_verts)  # View pointnet input

        x, pos, onehot, batch = self.verts_to_pointcloud_onehot(hand_verts, hand_feats, obj_verts, obj_feats)

        contact_batched = self.pointnet(x, pos, onehot, batch)

        contact = contact_batched.view(batch_size, hand_verts.shape[1] + obj_verts.shape[1], 10)

        out['contact_hand'] = contact[:, :hand_verts.shape[1], :]
        out['contact_obj'] = contact[:, hand_verts.shape[1]:, :]
        

        return out

    def forward_onehot_w_mano(self, hand_verts, hand_feats, obj_verts, obj_feats):

        device = hand_verts.device
        batch_size = hand_verts.shape[0]
        out = dict()

        if self.normalize_pts:
            tform = self.get_normalizing_tform(hand_verts, obj_verts)
            hand_verts = util.apply_tform(tform, hand_verts)
            obj_verts = util.apply_tform(tform, obj_verts)
            # util.vis_pointcloud(obj_verts, hand_verts)  # View pointnet input

        x, pos, onehot, batch = self.verts_to_pointcloud_onehot(hand_verts, hand_feats, obj_verts, obj_feats)

        contact_batched, mano_param = self.pointnet(x, pos, onehot, batch)

        contact = contact_batched.view(batch_size, hand_verts.shape[1] + obj_verts.shape[1], 10)

        out['contact_hand'] = contact[:, :hand_verts.shape[1], :]
        out['contact_obj'] = contact[:, hand_verts.shape[1]:, :]

        return out

    def softargmax1d(self,input_, beta=10):
        _, _, n = input_.shape
        device = input_.device
        input_ = nn.functional.softmax(beta * input_, dim=2)
        indices = torch.linspace(0, 1, n).to(device)
        result = torch.sum((n - 1) * input_ * indices, dim=-1)
        return result

    def forward_onehot_w_coma(self, hand_verts, hand_feats, obj_verts, obj_feats):

        device = hand_verts.device
        batch_size = hand_verts.shape[0]
        out = dict()

        if self.normalize_pts:
            tform = self.get_normalizing_tform(hand_verts, obj_verts)
            hand_verts = util.apply_tform(tform, hand_verts)
            obj_verts = util.apply_tform(tform, obj_verts)
            # util.vis_pointcloud(obj_verts, hand_verts)  # View pointnet input

        x, pos, onehot, batch = self.verts_to_pointcloud_onehot(hand_verts, hand_feats, obj_verts, obj_feats)

        contact_batched, mano_param = self.pointnet(x, pos, onehot, batch)

        contact = contact_batched.view(batch_size, hand_verts.shape[1] + obj_verts.shape[1], 10)

        out['contact_hand'] = contact[:, :hand_verts.shape[1], :]
        out['contact_obj'] = contact[:, hand_verts.shape[1]:, :]
        out['mano_param'] = mano_param

        return out
        

    def forward_coma(self, hand_contact, x_mesh):

        out = dict()

        x_contact_gt = hand_contact

        cls_softargmax = self.softargmax1d(x_contact_gt,beta=100) / 10

        cls_softargmax = cls_softargmax.unsqueeze(2)

        coma_input = torch.cat((x_mesh, cls_softargmax),dim=2)

        _ , logit = self.coma(coma_input)

        out['logit'] = logit

        return out

    def forward_C1(self, x_mesh, hand_contact):

        cls_softargmax = self.softargmax1d(hand_contact,beta=100) / 10

        cls_softargmax = cls_softargmax.unsqueeze(2)

        coma_input = torch.cat((x_mesh, cls_softargmax),dim=2)

        out , logit = self.C1(coma_input)

        return out, logit, coma_input

    def forward_C2(self, x_mesh, hand_contact):

        cls_softargmax = self.softargmax1d(hand_contact,beta=100) / 10

        cls_softargmax = cls_softargmax.unsqueeze(2)

        coma_input = torch.cat((x_mesh, cls_softargmax),dim=2)

        out , logit = self.C2(coma_input)

        return out, logit, coma_input

    @staticmethod
    def get_normalizing_tform(hand_verts, obj_verts, random_rot=True):
        """
        Find a 4x4 rigid transform to normalize the pointcloud. We choose the object center of mass to be the origin,
        the hand center of mass to be along the +X direction, and the rotation around this axis to be random.
        :param hand_verts: (batch, 778, 3)
        :param obj_verts: (batch, 2048, 3)
        :return: tform: (batch, 4, 4)
        """
        with torch.no_grad():
            obj_centroid = torch.mean(obj_verts, dim=1)  # (batch, 3)
            hand_centroid = torch.mean(hand_verts, dim=1)

            x_vec = F.normalize(hand_centroid - obj_centroid, dim=1)  # From object to hand
            if random_rot:
                rand_vec = transforms.random_rotations(hand_verts.shape[0], device=hand_verts.device)   # Generate random rot matrix
                y_vec = F.normalize(torch.cross(x_vec, rand_vec[:, :3, 0]), dim=1)  # Make orthogonal
            else:
                ref_pt = hand_verts[:, 80, :]
                y_vec = F.normalize(torch.cross(x_vec, ref_pt - obj_centroid), dim=1)  # From object to hand ref point

            z_vec = F.normalize(torch.cross(x_vec, y_vec), dim=1)  # Z axis

            tform = ops.eyes(4, hand_verts.shape[0], device=hand_verts.device)
            tform[:, :3, 0] = x_vec
            tform[:, :3, 1] = y_vec
            tform[:, :3, 2] = z_vec
            tform[:, :3, 3] = obj_centroid

            return torch.inverse(tform)

    @staticmethod
    def verts_to_pointcloud(hand_verts, hand_feats, obj_verts, obj_feats):
        """
        Convert hand and object vertices and features from Pytorch3D padded format (batch, vertices, N)
        to Pytorch-Geometric packed format (all_vertices, N)
        """
        batch_size = hand_verts.shape[0]
        device = hand_verts.device

        ptcloud_pos = torch.cat((hand_verts, obj_verts), dim=1)
        ptcloud_x = torch.cat((hand_feats, obj_feats), dim=1)

        _, N, _ = ptcloud_pos.shape  # (batch_size, num_points, 3)
        pos = ptcloud_pos.view(batch_size * N, -1)
        batch = torch.zeros((batch_size, N), device=device, dtype=torch.long)
        for i in range(batch_size):
            batch[i, :] = i
        batch = batch.view(-1)
        x = ptcloud_x.view(-1, hand_feats.shape[2])

        # print('x', x.shape, pos.shape, batch.shape)
        return x, pos, batch

    @staticmethod
    def verts_to_pointcloud_onehot(hand_verts, hand_feats, obj_verts, obj_feats):
        """
        Convert hand and object vertices and features from Pytorch3D padded format (batch, vertices, N)
        to Pytorch-Geometric packed format (all_vertices, N)
        """

        batch_size = hand_verts.shape[0]
        device = hand_verts.device
        
        ptcloud_pos = torch.cat((hand_verts, obj_verts), dim=1)
        ptcloud_x = torch.cat((hand_feats[:,:,:25], obj_feats[:,:,:25]), dim=1)

        _, N, _ = ptcloud_pos.shape  # (batch_size, num_points, 3)
        pos = ptcloud_pos.view(batch_size * N, -1)
        batch = torch.zeros((batch_size, N), device=device, dtype=torch.long)
        for i in range(batch_size):
            batch[i, :] = i
        batch = batch.view(-1)
        x = ptcloud_x.view(-1, hand_feats[:,:,:25].shape[2])
        
        onehot_hand = hand_feats[:,:,25:]
        onehot_obj  = obj_feats[:,:,25:]
        ptcloud_onehot = torch.cat((onehot_hand,onehot_obj), dim=1)
        onehot = ptcloud_onehot.view(-1, onehot_hand.shape[2])

        # print('x', x.shape, pos.shape, batch.shape)
        return x, pos, onehot, batch
