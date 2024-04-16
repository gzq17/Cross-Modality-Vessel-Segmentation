from curses import echo
import os
import random
import argparse
import time
import numpy as np
import torch, sys, copy
import torch.nn.functional as F
torch.cuda.manual_seed(1000)
np.random.seed(1000)
# import voxelmorph with pytorch backend
from model.reg_model import VxmDense
from model.loss_f import NCC, MSE, Grad, Dice
from data_process.reg_load_pairs import load_data_pairs, get_batch_patches, read_txt
from data_process.de_compose import compute_dice
import SimpleITK as sitk
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
#python add_label/add_label_train.py --image-loss 'mse' --weight 0.01 --model_dir 'model5/checkpoint/'
# data organization parameters
#model5 --image-loss 'mse' --weight 0.01  dice-weight 0.01
#model4 ncc weight 1.0 dice-weight 0.1
parser.add_argument('--root_path', type=str, default='/disk1/guozhanqiang/Cerebral/transfer_data2/')
parser.add_argument('--model_dir', type=str, default='model/')
parser.add_argument('--task', type=str, default='MRA2CTA', help='MRA2CTA, CTA2MRA')

parser.add_argument('--move_path', type=str, default='')
parser.add_argument('--move_label_path', type=str, default='')
parser.add_argument('--fix_path', type=str, default='')
parser.add_argument('--fix_label_path', type=str, default='')
parser.add_argument('--fix_stage_path', type=str, default='data2_model/supervised/model4/result_train_mra/')

parser.add_argument('--train_txt_path', type=str, default='data2/data_txt/train.txt')
parser.add_argument('--test_txt_path', type=str, default='data2/data_txt/test_mra.txt')
parser.add_argument('--train_mode', type=str, default='reg', help='seg, reg, or all')
parser.add_argument('--train', action='store_true', default=False)
# training parameters
parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--steps-per-epoch', type=int, default=50)
parser.add_argument('--load-model', type=bool, default=False, help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=100)
parser.add_argument('--reg_lr', type=float, default=1e-4)
parser.add_argument('--patch', type=list, default=[256, 256, 96])

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=1,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', type=bool, default=True, help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--dice_weight', type=float, default=0.01)
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--smooth_weight', type=float, default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

source_mode, target_mode = args.task[:3], args.task[4:7]
print(f'source mode:{source_mode}, target mode:{target_mode}')
out_path_parent = args.root_path + args.model_dir + args.task + '/'
os.makedirs(out_path_parent, exist_ok=True)
model_dir = out_path_parent + 'checkpoint-reg/'
os.makedirs(model_dir, exist_ok=True)
device = 'cuda'
# args.move_path = 'data2/' + source_mode + '_data/'
# args.move_label_path = 'data2/' + source_mode + '_label/'
args.move_path = 'registration_data/' + source_mode + '_data_reg/'
args.move_label_path = 'registration_data/' + source_mode + '_label_reg/'
args.fix_path = 'data2/' + target_mode + '_data/'
args.fix_label_path = 'data2/' + target_mode + '_label/'

patch = args.patch
bidir = args.bidir


def get_registraion_model(load_name=None):
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
    if load_name is None:
        reg_model = VxmDense(inshape=(patch[0], patch[1], patch[2]), nb_unet_features=[enc_nf, dec_nf],
         bidir=bidir, src_feats=2, trg_feats=2, int_steps=args.int_steps,int_downsize=args.int_downsize)
    else:
        reg_model = VxmDense.load(load_name, device)
    return reg_model.to(device)

def decompose_vol2cube(img, xx, yy, zz, patch):
    img_list = []
    x_b_list = [0, xx - patch[0]]
    y_b_list = [0, yy - patch[1]]
    z_b_list = [0, zz - patch[2]]
    for x_b in x_b_list:
        for y_b in y_b_list:
            for z_b in z_b_list:
                img_patch = copy.deepcopy(img[x_b:x_b+patch[0], y_b:y_b+patch[1], z_b:z_b+patch[2]])
                img_list.append(img_patch)
    return img_list

def compose_cube2vol(img_list, xx, yy, zz, patch):
    result = np.zeros((xx, yy, zz))
    xx_, yy_, zz_ = xx - xx // 2, yy - yy // 2, zz - zz // 2
    result[0:xx // 2, 0:yy//2, 0:zz//2] = img_list[0][0:xx // 2, 0:yy//2, 0:zz//2]
    result[0:xx // 2, 0:yy//2, zz//2:] = img_list[1][0:xx // 2, 0:yy//2, patch[2] - zz_:]
    result[0:xx // 2, yy//2:, 0:zz//2] = img_list[2][0:xx // 2, patch[1] - yy_:, 0:zz//2]
    result[0:xx // 2, yy//2:, zz//2:] = img_list[3][0:xx // 2, patch[1] - yy_:, patch[2] - zz_:]
    result[xx // 2:, 0:yy//2, 0:zz//2] = img_list[4][patch[0] - xx_:, 0:yy//2, 0:zz//2]
    result[xx // 2:, 0:yy//2, zz//2:] = img_list[5][patch[0] - xx_:, 0:yy//2, patch[2] - zz_:]
    result[xx // 2:, yy//2:, 0:zz//2] = img_list[6][patch[0] - xx_:, patch[1] - yy_:, 0:zz//2]
    result[xx // 2:, yy//2:, zz//2:] = img_list[7][patch[0] - xx_:, patch[1] - yy_:, patch[2] - zz_:]
    return result

def registration_validation(model, img_clec, batch_size):
    dice_before, dice_after = [], []
    for i in range(0, 5):
        move_img, fix_img, move_lbl, fix_lbl = img_clec[0][i], img_clec[2][i], img_clec[4][i], img_clec[5][i]
        move_ori, fix_ori = img_clec[1][i], img_clec[3][i]
        fix_stage = img_clec[6][i]
        xx, yy, zz = move_img.shape[0], move_img.shape[1], move_img.shape[2]
        move_img_list = decompose_vol2cube(move_img, xx, yy, zz, patch)
        fix_img_list = decompose_vol2cube(fix_img, xx, yy, zz, patch)
        move_lbl_list = decompose_vol2cube(move_lbl, xx, yy, zz, patch)
        fix_lbl_list = decompose_vol2cube(fix_lbl, xx, yy, zz, patch)
        move_ori_list = decompose_vol2cube(move_ori, xx, yy, zz, patch)
        fix_ori_list = decompose_vol2cube(fix_ori, xx, yy, zz, patch)
        fix_stage_list = decompose_vol2cube(fix_stage, xx, yy, zz, patch)
        for i in range(0, len(move_img_list)):
            move_img_patch = move_img_list[i][np.newaxis, np.newaxis, :, :, :]
            fix_img_patch = fix_img_list[i][np.newaxis, np.newaxis, :, :, :]
            move_lbl_patch = move_lbl_list[i][np.newaxis, np.newaxis, :, :, :]
            fix_lbl_patch = fix_lbl_list[i][np.newaxis, np.newaxis, :, :, :]
            move_ori_patch = move_ori_list[i][np.newaxis, np.newaxis, :, :, :]
            fix_ori_patch = fix_ori_list[i][np.newaxis, np.newaxis, :, :, :]
            fix_stage_patch = fix_stage_list[i][np.newaxis, np.newaxis, :, :, :]

            batch_move_img_torch = torch.from_numpy(move_img_patch).to(device).float()
            batch_move_ori_torch = torch.from_numpy(copy.deepcopy(move_ori_patch)).to(device).float()
            batch_fix_img_torch = torch.from_numpy(fix_img_patch).to(device).float()
            batch_fix_ori_torch = torch.from_numpy(copy.deepcopy(fix_ori_patch)).to(device).float()
            batch_move_lbl_torch = torch.from_numpy(move_lbl_patch).to(device).float()

            move_cat = torch.cat([batch_move_img_torch, torch.from_numpy(copy.deepcopy(move_lbl_patch)).to(device).float()], dim=1)
            fix_cat = torch.cat([batch_fix_img_torch, torch.from_numpy(copy.deepcopy(fix_stage_patch)).to(device).float()], dim=1)
            with torch.no_grad():
                y_move, y_move_label = model.test_forward(move_cat, fix_cat, batch_move_lbl_torch)
                # y_move, y_move_label = model.test_forward(batch_move_ori_torch, batch_fix_ori_torch, batch_move_lbl_torch)
                # y_move, y_move_label = model.test_forward(batch_move_img_torch, batch_fix_img_torch, batch_move_lbl_torch)
            batch_fix_lbl_c = fix_lbl_list[i].astype("int")
            batch_move_lbl_c = move_lbl_list[i].astype("int")
            y_batch_warp = y_move_label.detach().cpu().numpy().squeeze()
            y_batch_warp[y_batch_warp > 0.5] = 1
            y_batch_warp[y_batch_warp <= 0.5] = 0
            dice1 = compute_dice(copy.deepcopy(batch_move_lbl_c), copy.deepcopy(batch_fix_lbl_c))
            dice2 = compute_dice(copy.deepcopy(y_batch_warp), copy.deepcopy(batch_fix_lbl_c))
            dice_before.append(dice1)
            dice_after.append(dice2)
    dice_before = np.array(dice_before).mean()
    dice_after = np.array(dice_after).mean()
    return dice_before, dice_after

def registration_test():
    load_name = model_dir + 'best.pt'
    out_dir = model_dir + 'reg_result/'
    os.makedirs(out_dir, exist_ok=True)
    reg_model = get_registraion_model(load_name)
    reg_model.eval()
    img_list = read_txt(args.root_path + args.train_txt_path)
    dice_patch_before, dice_patch_after = [], []
    dice_com_before, dice_com_after = [], []
    for k in range(0, len(img_list)):
        img_clec = load_data_pairs(root_path=args.root_path, img_list=img_list[k:k+1], move_path=args.move_path,
         fix_path=args.fix_path, move_label_path=args.move_label_path, fix_label_path=args.fix_label_path, 
         fix_stage_path=args.fix_stage_path, move=source_mode.lower(), fix=target_mode.lower())
        ref_img_name = args.root_path + args.fix_label_path + img_list[k] + '_' + target_mode.lower() + '_label.nii.gz'
        if not os.path.exists(ref_img_name):
            ref_img_name = args.root_path + args.fix_label_path + img_list[k] + '_label.nii.gz'
        img_ = sitk.ReadImage(ref_img_name)
        out_name = out_dir + img_list[k] + '_transfer_label.nii.gz'
        move_img, fix_img, move_lbl, fix_lbl = img_clec[0][0], img_clec[2][0], img_clec[4][0], img_clec[5][0]
        move_ori, fix_ori = img_clec[1][0], img_clec[3][0]
        fix_stage = img_clec[6][0]
        xx, yy, zz = move_img.shape[0], move_img.shape[1], move_img.shape[2]
        move_img_list = decompose_vol2cube(move_img, xx, yy, zz, patch)
        fix_img_list = decompose_vol2cube(fix_img, xx, yy, zz, patch)
        move_lbl_list = decompose_vol2cube(move_lbl, xx, yy, zz, patch)
        fix_lbl_list = decompose_vol2cube(fix_lbl, xx, yy, zz, patch)
        move_ori_list = decompose_vol2cube(move_ori, xx, yy, zz, patch)
        fix_ori_list = decompose_vol2cube(fix_ori, xx, yy, zz, patch)
        fix_stage_list = decompose_vol2cube(fix_stage, xx, yy, zz, patch)
        result_list = []
        dice_before, dice_after = [], []
        for i in range(0, len(move_img_list)):
            move_img_patch = move_img_list[i][np.newaxis, np.newaxis, :, :, :]
            fix_img_patch = fix_img_list[i][np.newaxis, np.newaxis, :, :, :]
            move_lbl_patch = move_lbl_list[i][np.newaxis, np.newaxis, :, :, :]
            fix_lbl_patch = fix_lbl_list[i][np.newaxis, np.newaxis, :, :, :]
            move_ori_patch = move_ori_list[i][np.newaxis, np.newaxis, :, :, :]
            fix_ori_patch = fix_ori_list[i][np.newaxis, np.newaxis, :, :, :]
            fix_stage_patch = fix_stage_list[i][np.newaxis, np.newaxis, :, :, :]
            batch_move_img_torch = torch.from_numpy(move_img_patch).to(device).float()
            batch_fix_img_torch = torch.from_numpy(fix_img_patch).to(device).float()
            batch_move_ori_torch = torch.from_numpy(move_ori_patch).to(device).float()
            batch_fix_ori_torch = torch.from_numpy(fix_ori_patch).to(device).float()
            batch_move_lbl_torch = torch.from_numpy(move_lbl_patch).to(device).float()
            batch_fix_lbl_torch = torch.from_numpy(fix_lbl_patch).to(device).float()
            batch_fix_stage_torch = torch.from_numpy(fix_stage_patch).to(device).float()
            move_cat = torch.cat([batch_move_img_torch, batch_move_lbl_torch], dim=1)
            fix_cat = torch.cat([batch_fix_img_torch, batch_fix_stage_torch], dim=1)
            # y_move, y_move_label = reg_model.test_forward(batch_move_img_torch, batch_fix_img_torch, batch_move_lbl_torch)
            y_move, y_move_label = reg_model.test_forward(move_cat, fix_cat, batch_move_lbl_torch)
            # y_move, y_move_label = reg_model.test_forward(batch_move_ori_torch, batch_fix_ori_torch, batch_move_lbl_torch)
            batch_fix_lbl_c = fix_lbl_list[i].astype("int")
            # result_list.append(batch_fix_lbl_c)
            batch_move_lbl_c = move_lbl_list[i].astype("int")
            y_batch_warp = y_move_label.detach().cpu().numpy().squeeze()
            y_batch_warp[y_batch_warp > 0.5] = 1
            y_batch_warp[y_batch_warp <= 0.5] = 0
            result_list.append(y_batch_warp)
            dice1 = compute_dice(copy.deepcopy(batch_move_lbl_c), copy.deepcopy(batch_fix_lbl_c))
            dice2 = compute_dice(copy.deepcopy(y_batch_warp), copy.deepcopy(batch_fix_lbl_c))
            dice_before.append(dice1)
            dice_after.append(dice2)
        dice_before = np.array(dice_before).mean()
        dice_after = np.array(dice_after).mean()
        print(str(k+1) + '/' + str(len(img_list)), img_list[k], dice_before, dice_after)
        dice_patch_before.append(dice_before)
        dice_patch_after.append(dice_after)
        result = compose_cube2vol(result_list, xx, yy, zz, patch)
        dice1 = compute_dice(copy.deepcopy(move_lbl), copy.deepcopy(fix_lbl))
        dice2 = compute_dice(copy.deepcopy(result), copy.deepcopy(fix_lbl))
        result = result.swapaxes(0, 2)
        result = sitk.GetImageFromArray(result)
        result.CopyInformation(img_)
        sitk.WriteImage(result, out_name)
        dice_com_before.append(dice1)
        dice_com_after.append(dice2)
        print(str(k+1) + '/' + str(len(img_list)), img_list[k], dice1, dice2)
    print(np.array(dice_patch_before).mean(), np.array(dice_patch_after).mean())
    print(np.array(dice_com_before).mean(), np.array(dice_com_after).mean())

def adjust_lr(optimizer, epoch, lr):
    lr_c = lr * ((1 - epoch/(args.epochs + 1)) ** 0.9)
    for p in optimizer.param_groups:
        p['lr'] = lr_c

def split_seg(seg):
    prob_seg = np.zeros((seg.shape[0], 2, *seg.shape[2:]))
    prob_seg[:, 0:1, ...][seg == 0] = 1
    prob_seg[:, 1:2, ...][seg == 1] = 1
    return prob_seg

def train_registration():
    print(("dice weight:{}, " + args.image_loss + ",  smooth:{}").format(args.dice_weight, args.smooth_weight))
    print(args.batch_size, np.array(patch))
    model_dir_registration = model_dir
    os.makedirs(model_dir_registration, exist_ok=True)
    loss_file_name = model_dir_registration + 'loss.txt'
    loss_file = open(loss_file_name, 'w')
    reg_model = get_registraion_model()
    reg_model.train()
    reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=args.reg_lr)

    if args.image_loss == 'ncc':
        image_loss_func = NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = MSE().loss
    image_dice_func = Dice().loss
    if bidir:
        reg_losses = [image_loss_func, image_loss_func, image_dice_func, image_dice_func]
        reg_weights = [0.5, 0.5, args.dice_weight, args.dice_weight]
    else:
        reg_losses = [image_loss_func, image_dice_func]
        reg_weights = [1, args.dice_weight]
    reg_losses += [Grad('l2', loss_mult=args.int_downsize).loss]
    reg_weights += [args.smooth_weight]
    
    img_list = read_txt(args.root_path + args.train_txt_path)
    print("load done, num:{}".format(len(img_list)))
    info = ('dice weight:{}, ' + args.image_loss + ',  smooth:{}').format(args.dice_weight, args.smooth_weight)
    info = info + 'move_path:' + args.move_path + '\n'
    info = info + 'fix_path:' + args.fix_path + '\n'
    info = info + 'move_label_path:' + args.move_label_path + '\n'
    info = info + 'fix_label_path:' + args.fix_label_path + '\n'
    info = info + 'fix_stage_path:' + args.fix_stage_path + '\n'
    loss_file.write(info + '\n\n')
    print(info)
    
    img_clec = load_data_pairs(root_path=args.root_path, img_list=img_list, move_path=args.move_path,
     fix_path=args.fix_path, move_label_path=args.move_label_path, fix_label_path=args.fix_label_path, 
     fix_stage_path=args.fix_stage_path, move=source_mode.lower(), fix=target_mode.lower())

    print("load data done")
    dice_max = 0.0
    max_index = 0
    for epoch in range(0, args.epochs):
        if epoch % 100 == 0:
            reg_model.save(os.path.join(model_dir_registration, '%04d.pt' % epoch))
        adjust_lr(reg_optimizer, epoch, args.reg_lr)
        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        for step in range(args.steps_per_epoch):
            step_start_time = time.time()
            batch_move_img, batch_move_ori, batch_fix_img, batch_fix_ori, batch_move_lbl, batch_fix_lbl, batch_fix_stage = get_batch_patches(img_clec, args.batch_size, np.array(patch))
            batch_move_lbl_ = split_seg(batch_move_lbl)
            batch_fix_stage_ = split_seg(batch_fix_stage)
            batch_move_img_t = torch.from_numpy(copy.deepcopy(batch_move_img)).to(device).float()
            batch_move_ori_t = torch.from_numpy(copy.deepcopy(batch_move_ori)).to(device).float()
            batch_fix_img_t = torch.from_numpy(copy.deepcopy(batch_fix_img)).to(device).float()
            batch_fix_ori_t = torch.from_numpy(copy.deepcopy(batch_fix_ori)).to(device).float()
            batch_move_lbl_t = torch.from_numpy(copy.deepcopy(batch_move_lbl_)).to(device).float()
            batch_fix_stage_t = torch.from_numpy(copy.deepcopy(batch_fix_stage_)).to(device).float()
            
            y_true = [copy.deepcopy(np.concatenate([batch_fix_img, batch_fix_stage], axis=1)), 
                      copy.deepcopy(np.concatenate([batch_move_img, batch_move_lbl], axis=1)), 
                      copy.deepcopy(batch_fix_stage_), copy.deepcopy(batch_move_lbl_)]
            # y_true = [copy.deepcopy(batch_fix_img), copy.deepcopy(batch_move_img), copy.deepcopy(batch_fix_stage_), copy.deepcopy(batch_move_lbl_)]
            # y_true = [copy.deepcopy(batch_fix_ori), copy.deepcopy(batch_move_ori), copy.deepcopy(batch_fix_stage_), copy.deepcopy(batch_move_lbl_)]
            y_true = [torch.from_numpy(d).to(device).float() for d in y_true]
            # y_pred = reg_model(batch_move_ori_t, batch_fix_ori_t, batch_move_lbl_t, batch_fix_stage_t)
            # y_pred = reg_model(batch_move_img_t, batch_fix_img_t, batch_move_lbl_t, batch_fix_stage_t)
            y_pred = reg_model(torch.cat([batch_move_img_t, torch.from_numpy(copy.deepcopy(batch_move_lbl)).to(device).float()], dim=1), 
                               torch.cat([batch_fix_img_t, torch.from_numpy(copy.deepcopy(batch_fix_stage)).to(device).float()], dim=1), 
                               batch_move_lbl_t, batch_fix_stage_t)

            loss = 0 
            loss_list = []
            for n, loss_function in enumerate(reg_losses):
                # if n == 1 or n == 3:
                #     continue
                # if n == 2 or n == 3:
                #     continue
                if n == len(y_pred) - 1:
                    curr_loss = loss_function(y_pred[n], y_pred[n]) * reg_weights[n]
                else:
                    if y_pred[n].shape[1] >= 1:
                        curr_loss = loss_function(y_true[n][:, :1], y_pred[n][:, :1]) * reg_weights[n]
                    else:
                        curr_loss = loss_function(y_true[n], y_pred[n]) * reg_weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss
            print("epoch:{}/{},step:{}/{},loss:{}".format(epoch+1, args.epochs, step, args.steps_per_epoch, loss.item()))
            if step % 10 == 0:
                loss_file.write("epoch:{}/{},step:{}/{},loss:{}\n".format(epoch+1, args.epochs, step, args.steps_per_epoch, loss.item()))
                loss_file.flush()
            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            reg_optimizer.zero_grad()
            loss.backward()
            reg_optimizer.step()
            epoch_step_time.append(time.time() - step_start_time)
        # print epoch info
        dice_before, dice_after = registration_validation(reg_model, img_clec, 1)
        if dice_after > dice_max:
            dice_max = dice_after
            max_index = epoch
            reg_model.save(os.path.join(model_dir_registration, 'best.pt'))
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        max_dice_info = 'max dice:{:.4f}, max index:{}'.format(dice_max, max_index)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        print("dice before:{:.4f}, dice after:{:.4f}".format(dice_before, dice_after))
        print(max_dice_info)
        loss_file.write("\n")
        loss_file.write(' - '.join((epoch_info, time_info, loss_info)))
        loss_file.write("\ndice before:{:.4f}, dice after:{:.4f}".format(dice_before, dice_after))
        loss_file.write('\n' + max_dice_info)
        loss_file.write("\n\n")
        loss_file.flush()
    reg_model.save(os.path.join(model_dir_registration, '%04d.pt' % args.epochs))

def main():
    if args.train:
        train_registration()
    else:
        registration_test()

if __name__ == '__main__':
    main()

