import os
import time
import torch
import json
from glob import glob

from torch.utils.tensorboard import SummaryWriter

class Writer:
    def __init__(self, exp_name, args=None):
        self.args = args

        path = f'./{exp_name}'
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Folder '{path}' created.")

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
        else:
            self.log_file = os.path.join(
                    f'./{exp_name}', 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train Hand Contact Loss: {:.4f}, Train Obj Contact Loss: {:.4f}, Train taxonomy Loss: {:.4f}, f1_score_hand: {:.4f}, tax_acc: {:.4f}, f1_score_obj: {:.4f},  Test Hand Contact Loss: {:.4f} , Test Obj Contact Loss: {:.4f} , precision_hand: {:.4f} , recall_hand: {:.4f} , precision_obj: {:.4f} , recall_obj: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_contact_loss_hand'], info['train_contact_loss_obj'], info['train_taxonomy_loss'], info['f1_score_hand'], info['tax_acc'], info['f1_score_obj'], info['test_hand_contact_loss'], info['test_obj_contact_loss'],info['precision_hand'],info['recall_hand'],info['precision_obj'],info['recall_obj'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def print_info_train(self, info):
        message = 'Epoch: {}/{}, Train Hand Contact Loss: {:.4f}, Train Obj Contact Loss: {:.4f}, Train taxonomy Loss: {:.4f}  Train joint Loss: {:.4f}  Train mano_pose Loss: {:.4f}' \
                .format(info['current_epoch'], info['epochs'] * info['mok'], \
                info['train_contact_loss_hand'], info['train_contact_loss_obj'], info['train_taxonomy_loss'], info['joint_loss'], info['mano_pose_loss'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def print_info_train_stepB(self, info):
        message = 'Epoch: {}/{}, Train Hand Contact Loss: {:.4f}, Train Obj Contact Loss: {:.4f}, Train taxonomy Loss: {:.4f}' \
                .format(info['current_epoch'], info['epochs'] * info['mok'], \
                info['train_contact_loss_hand'], info['train_contact_loss_obj'], info['train_taxonomy_loss'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def print_info_train_stepB_V2(self, info):
        message = 'Epoch: {}/{}, Train Hand Contact Loss: {:.4f}, Train Obj Contact Loss: {:.4f}, Train taxonomy Loss: {:.4f}, Source Contact Loss {:.4f}' \
                .format(info['current_epoch'], info['epochs'] * info['mok'], \
                info['train_contact_loss_hand'], info['train_contact_loss_obj'], info['train_taxonomy_loss'], info['total_contact_loss_gt'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def print_info_train_stepA(self, info):
        message = 'Epoch: {}/{}, train_acc1 : {:.4f}, train_acc2 : {:.4f}, loss_s : {:.4f}  recon : {:.4f}' \
                .format(info['current_epoch'], info['epochs'] * info['mok'], \
                info['train_acc1'], info['train_acc2'], info['total_loss_s'], info['total_recon'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def print_info_train_stepAB(self, info):
        message = 'Epoch: {}/{}, train_acc1 : {:.4f}, train_acc2 : {:.4f}, loss_s : {:.4f}  recon : {:.4f}  contact_obj : {:.4f}   contact_hand : {:.4f} ' \
                .format(info['current_epoch'], info['epochs'] * info['mok'], \
                info['train_acc1'], info['train_acc2'], info['total_loss_s'], info['total_recon'], info['train_contact_loss_obj'], info['train_contact_loss_hand'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def print_info_test(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, f1_score_hand: {:.4f}, tax_acc: {:.4f}, f1_score_obj: {:.4f},  Test Hand Contact Loss: {:.4f} , Test Obj Contact Loss: {:.4f} , precision_hand: {:.4f} , recall_hand: {:.4f} , precision_obj: {:.4f} , recall_obj: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['f1_score_hand'], info['tax_acc'], info['f1_score_obj'], info['test_hand_contact_loss'], info['test_obj_contact_loss'],info['precision_hand'],info['recall_hand'],info['precision_obj'],info['recall_obj'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)


    def s_writer(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/train_contact_hand", info['train_contact_loss_hand'], epoch)
        s_writer.add_scalar("Loss/train_contact_obj", info['train_contact_loss_obj'], epoch)
        s_writer.add_scalar("Loss/train_taxonomy_loss", info['train_taxonomy_loss'], epoch)

        s_writer.add_scalar("Loss/f1_score_hand", info['f1_score_hand'], epoch)
        s_writer.add_scalar("Loss/test_taxonomy_acc", info['tax_acc'], epoch)
        s_writer.add_scalar("Loss/f1_score_obj", info['f1_score_obj'], epoch)

        s_writer.add_scalar("Loss/test_hand_contact_loss", info['test_hand_contact_loss'], epoch)
        s_writer.add_scalar("Loss/test_obj_contact_loss", info['test_obj_contact_loss'], epoch)

        s_writer.add_scalar("Loss/precision_hand", info['precision_hand'], epoch)
        s_writer.add_scalar("Loss/recall_hand", info['recall_hand'], epoch)
        s_writer.add_scalar("Loss/precision_obj", info['precision_obj'], epoch)
        s_writer.add_scalar("Loss/recall_obj", info['recall_obj'], epoch)


        s_writer.flush()

    def s_writer_train(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/train_contact_hand", info['train_contact_loss_hand'], epoch)
        s_writer.add_scalar("Loss/train_contact_obj", info['train_contact_loss_obj'], epoch)
        s_writer.add_scalar("Loss/train_taxonomy_loss", info['train_taxonomy_loss'], epoch)

        s_writer.add_scalar("Loss/train_joint_loss", info['joint_loss'], epoch)
        s_writer.add_scalar("Loss/train_mano_pose_loss", info['mano_pose_loss'], epoch)

        s_writer.flush()

    def s_writer_train_stepB(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/stepB/train_contact_hand", info['train_contact_loss_hand'], epoch)
        s_writer.add_scalar("Loss/stepB/train_contact_obj", info['train_contact_loss_obj'], epoch)
        s_writer.add_scalar("Loss/stepB/train_taxonomy_loss", info['train_taxonomy_loss'], epoch)

        s_writer.flush()

    def s_writer_train_stepB_V2(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/stepB/train_contact_hand", info['train_contact_loss_hand'], epoch)
        s_writer.add_scalar("Loss/stepB/train_contact_obj", info['train_contact_loss_obj'], epoch)
        s_writer.add_scalar("Loss/stepB/train_taxonomy_loss", info['train_taxonomy_loss'], epoch)
        s_writer.add_scalar("Loss/stepB/total_contact_loss_gt", info['total_contact_loss_gt'], epoch)

        s_writer.flush()

    def s_writer_train_stepB_iteration(self, info, s_writer, iteration):

        s_writer.add_scalar("Loss/iterB/train_contact_hand", info['train_contact_loss_hand'], iteration)
        s_writer.add_scalar("Loss/iterB/train_contact_obj", info['train_contact_loss_obj'], iteration)
        s_writer.add_scalar("Loss/iterB/train_taxonomy_loss", info['train_taxonomy_loss'], iteration)

        s_writer.flush()

    def s_writer_train_stepB_iteration_V2(self, info, s_writer, iteration):

        s_writer.add_scalar("Loss/iterB/train_contact_hand", info['train_contact_loss_hand'], iteration)
        s_writer.add_scalar("Loss/iterB/train_contact_obj", info['train_contact_loss_obj'], iteration)
        s_writer.add_scalar("Loss/iterB/train_taxonomy_loss", info['train_taxonomy_loss'], iteration)
        s_writer.add_scalar("Loss/iterB/loss_gt", info['loss_gt'], iteration)

        s_writer.flush()

    def s_writer_train_stepA_iteration(self, info, s_writer, iteration):

        s_writer.add_scalar("Loss/iterA/train_acc1", info['train_acc1'], iteration)
        s_writer.add_scalar("Loss/iterA/train_acc2", info['train_acc2'], iteration)
        s_writer.add_scalar("Loss/iterA/total_loss_s", info['loss_s'], iteration)
        s_writer.add_scalar("Loss/iterA/total_recon", info['total_recon'], iteration)

        s_writer.flush()

    def s_writer_train_stepAB_iteration(self, info, s_writer, iteration):

        s_writer.add_scalar("Loss/iter/train_acc1",   info['train_acc1'], iteration)
        s_writer.add_scalar("Loss/iter/train_acc2",   info['train_acc2'], iteration)
        s_writer.add_scalar("Loss/iter/total_loss_s", info['total_loss_s'], iteration)
        s_writer.add_scalar("Loss/iter/total_recon",  info['total_recon'], iteration)
        s_writer.add_scalar("Loss/iter/total_obj_contact",   info['train_contact_loss_obj'], iteration)
        s_writer.add_scalar("Loss/iter/total_hand_contact",  info['train_contact_loss_hand'], iteration)

        s_writer.flush()

    def s_writer_train_stepA(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/stepA/ain_acc1", info['train_acc1'], epoch)
        s_writer.add_scalar("Loss/stepA/train_acc2", info['train_acc2'], epoch)
        s_writer.add_scalar("Loss/stepA/total_loss_s", info['total_loss_s'], epoch)
        s_writer.add_scalar("Loss/stepA/total_recon", info['total_recon'], epoch)

        s_writer.flush()

    def s_writer_train_stepAB(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/train_acc1",   info['train_acc1'], epoch)
        s_writer.add_scalar("Loss/train_acc2",   info['train_acc2'], epoch)
        s_writer.add_scalar("Loss/total_loss_s", info['total_loss_s'], epoch)
        s_writer.add_scalar("Loss/total_recon",  info['total_recon'], epoch)
        s_writer.add_scalar("Loss/total_obj_contact",   info['train_contact_loss_obj'], epoch)
        s_writer.add_scalar("Loss/total_hand_contact",  info['train_contact_loss_hand'], epoch)

        s_writer.flush()

    def s_writer_test(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/f1_score_hand", info['f1_score_hand'], epoch)
        s_writer.add_scalar("Loss/test_taxonomy_acc", info['tax_acc'], epoch)
        s_writer.add_scalar("Loss/f1_score_obj", info['f1_score_obj'], epoch)

        s_writer.add_scalar("Loss/test_hand_contact_loss", info['test_hand_contact_loss'], epoch)
        s_writer.add_scalar("Loss/test_obj_contact_loss", info['test_obj_contact_loss'], epoch)

        s_writer.add_scalar("Loss/precision_hand", info['precision_hand'], epoch)
        s_writer.add_scalar("Loss/recall_hand", info['recall_hand'], epoch)
        s_writer.add_scalar("Loss/precision_obj", info['precision_obj'], epoch)
        s_writer.add_scalar("Loss/recall_obj", info['recall_obj'], epoch)


        s_writer.flush()


    def save_checkpoint(self, model, optimizer, epoch, exp_name):

        path = f'./{exp_name}'
            
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            os.path.join(path,
                         'checkpoint_{:03d}.pt'.format(epoch)))
