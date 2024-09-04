# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from network.loader import *
import network.util as util
from network.hand_object import HandObject
import time
from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import visualization as o3dv


def vis_contactMap(gt_ho):

    hand, obj = gt_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)

    geom_list = [hand,obj]

    o3dv.draw_geometries(geom_list,window_name=str('contactmap'))

def show_optimization(data, opt_state, hand_contact_target=None, obj_contact_target=None, is_video=False, label=None, vis_method=1, delay=0.001):
    """Displays video/still frame of optimization process
    Contact visualization options:
    0 GT contact on opt
    1 Predicted contact on opt
    2 Live contact on opt hand
    3 Live contact on both
    4 No contact on any
    5 No hand contact, predicted obj contact
    """
    

    batch_size = data['aug_pose'].shape[0]
    device = data['aug_pose'].device

    gt_ho = HandObject()
    opt_ho = HandObject()

    data['mesh_gt'] = Meshes(verts=data['object_full_verts'], faces=data['object_full_face'])

    #gt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_gt'])

    data['hand_pose_gt'] = torch.cat([data['aug_rot'], data['aug_pose']], dim=2).to(device)

    gt_ho.load_from_batch(data['hand_beta'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact'], data['obj_contact'], data['mesh_gt'])

    #opt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_aug'], obj_rot=opt_state[-1]['obj_rot'])
    opt_ho.load_from_batch(data['hand_beta'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact'], data['obj_contact'], data['mesh_gt'], obj_rot=opt_state[-1]['obj_rot'])

    hand_mesh_gt, obj_mesh_gt = gt_ho.get_o3d_meshes()
    hand_mesh_opt, obj_mesh_opt = opt_ho.get_o3d_meshes()
    geom_list = [hand_mesh_gt, obj_mesh_gt, obj_mesh_opt, hand_mesh_opt]

    if vis_method == 1 or vis_method == 5:

        util.mesh_set_color(hand_contact_target, hand_mesh_opt)
        util.mesh_set_color(obj_contact_target, obj_mesh_opt)

        if obj_contact_target.shape[1] == util.SAMPLE_VERTS_NUM: obj_contact_target = upscale_contact(data['mesh_gt'], data['obj_sampled_idx'], obj_contact_target.to(device))

        util.mesh_set_color(obj_contact_target, obj_mesh_opt)
    if vis_method == 2 or vis_method == 3:
        util.mesh_set_color(opt_state[-1]['contact_hand'].squeeze(), hand_mesh_opt)
        if opt_state[-1]['contact_obj'].shape[1] == util.SAMPLE_VERTS_NUM:
            c = upscale_contact(data['mesh_gt'], data['obj_sampled_idx'], opt_state[-1]['contact_obj'])
            util.mesh_set_color(c, obj_mesh_opt)
        else:
            util.mesh_set_color(opt_state[-1]['contact_obj'].squeeze(), obj_mesh_opt)
    if vis_method == 4 or vis_method == 5:
        hand_mesh_gt.paint_uniform_color(np.asarray([150.0, 250.0, 150.0]) / 255)   # Green
        hand_mesh_opt.paint_uniform_color(np.asarray([250.0, 150.0, 150.0]) / 255)  # Red

    if vis_method == 4:
        obj_mesh_gt.paint_uniform_color(np.asarray([100.0, 100.0, 100.0]) / 255)   # Gray
        obj_mesh_opt.paint_uniform_color(np.asarray([100.0, 100.0, 100.0]) / 255)  # Gray

    if label is not None:
        lbl_verts = util.text_3d(label, pos=[0, 0.1, 0], font_size=20, density=2)
        geom_list.append(lbl_verts)

    #hand_mesh_opt.vertices = o3du.Vector3dVector(opt_state[-1]['hand_verts'].squeeze())
    #hand_mesh_opt.compute_vertex_normals()

    hand_mesh_gt.translate((0, 0.2, 0))
    obj_mesh_gt.translate((0, 0.2, 0))

    if not is_video:
        o3dv.draw_geometries(geom_list)
    else:
        vis = o3dv.VisualizerWithKeyCallback()
        vis.create_window()
        for g in geom_list:
            vis.add_geometry(g)

        for i in range(len(opt_state)):
            out_dict = opt_state[i % len(opt_state)]

            if out_dict['obj_rot'][0, 0, 0] < 1:
                obj_verts = util.apply_rot(out_dict['obj_rot'], data['mesh_aug'].verts_padded(), around_centroid=True).squeeze()
                obj_mesh_opt.vertices = o3du.Vector3dVector(obj_verts)
            
            hand_mesh_opt.vertices = o3du.Vector3dVector(np.array(out_dict['hand_verts'][0,:,:].squeeze()))

            if vis_method == 2 or vis_method == 3:
                util.mesh_set_color(out_dict['contact_hand'].squeeze(), hand_mesh_opt)
            if vis_method == 3:
                if out_dict['contact_obj'].shape[1] == util.SAMPLE_VERTS_NUM:
                    c = util.upscale_contact(data['mesh_gt'], data['obj_sampled_idx'], out_dict['contact_obj'])
                    util.mesh_set_color(c, obj_mesh_opt)
                else:
                    util.mesh_set_color(out_dict['contact_obj'].squeeze(), obj_mesh_opt)

            vis.update_geometry(hand_mesh_opt)
            vis.update_geometry(obj_mesh_opt)
            vis.poll_events()
            vis.update_renderer()

            #time.sleep(0.1)

            if i % len(opt_state) == 0:
                time.sleep(1)
            # time.sleep(delay)

        time.sleep(10)
        #vis.destroy_window()



