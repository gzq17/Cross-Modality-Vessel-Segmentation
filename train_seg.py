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
from model.seg_model import SegmentationModel
from model.loss_f import dice_loss
from data_process.load_patch import load_data_pairs, get_batch_patches, read_txt
# from other_fun.evaluate import post_result
from data_process.de_compose import decompose_vol2cube, compose_label_cube2vol, compute_dice
import SimpleITK as sitk
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/disk1/guozhanqiang/Cerebral/transfer_data2/')
parser.add_argument('--model_dir', type=str, default='model/')
parser.add_argument('--task', type=str, default='MRA2CTA-source', help='MRA2CTA-source, MRA2CTA-target')
parser.add_argument('--label_path', type=str, default='')
parser.add_argument('--input_mode', type=str, default='all', help='ori, ehc or all')
parser.add_argument('--train_txt_path', type=str, default='data2/data_txt/train.txt')
parser.add_argument('--test_txt_path', type=str, default='data2/data_txt/test_cta.txt')
parser.add_argument('--load_name', type=str, default='')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--test_over', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=8, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--steps-per-epoch', type=int, default=50)
parser.add_argument('--seg_lr', type=float, default=1e-3)
parser.add_argument('--patch', type=list, default=[96, 96, 80])

args = parser.parse_args()
print(args.train)
source_mode, target_mode, train_now = args.task[:3], args.task[4:7], args.task.split('-')[-1]
print(f'source mode:{source_mode}, target mode:{target_mode}, train:{train_now}')
out_path_parent = args.root_path + args.model_dir + args.task[:7] + '/'
os.makedirs(out_path_parent, exist_ok=True)
model_dir = out_path_parent + 'checkpoint-' + train_now + '/'
os.makedirs(model_dir, exist_ok=True)
device = 'cuda'
if train_now == 'source':
    args.label_path = 'data2/' + source_mode + '_label/'
    train_mode = source_mode
else:
    train_mode = target_mode
    if args.label_path == '':
        args.label_path = args.model_dir + args.task[:7] + '/reg_generate_label/'
if 'icp' in args.label_path:
    model_dir = out_path_parent + 'checkpoint-' + train_now + '-icp/'
    os.makedirs(model_dir, exist_ok=True)
print(args.label_path)

def adjust_lr(optimizer, epoch, lr):
    lr_c = lr * ((1 - epoch/(args.epochs + 1)) ** 0.9)
    for p in optimizer.param_groups:
        p['lr'] = lr_c

def get_segmentation_model(inchannel, load_name=None):
    if load_name is None:
        seg_model = SegmentationModel(inchannel, 2, activate='relu', norm='instance')
    else:
        seg_model = SegmentationModel(inchannel, 2, activate='relu', norm='instance')
        seg_model.load_state_dict(torch.load(load_name))
    return seg_model.to(device)

def split_seg(seg):
    prob_seg = np.zeros((*seg.shape, 2))
    prob_seg[:, :, :, :, 0][seg == 0] = 1
    prob_seg[:, :, :, :, 1][seg == 1] = 1
    return prob_seg

def train():
    loss_file_name = model_dir + 'loss.txt'
    loss_file = open(loss_file_name, 'w')
    if args.input_mode == 'ori' or args.input_mode == 'ehc':
        seg_model = get_segmentation_model(inchannel=1)
    elif args.input_mode == 'all':
        seg_model = get_segmentation_model(inchannel=2)
    else:
        return
    seg_model.train()
    seg_optimizer = torch.optim.Adam(seg_model.parameters(), lr=args.seg_lr)
    img_list = read_txt(args.root_path + args.train_txt_path)
    print(img_list)
    print("load done, num:{}".format(len(img_list))) 
    all_data_clec = load_data_pairs(root_path=args.root_path, img_list=img_list, img_path='data2/' + train_mode + '_data/',
     label_path=args.label_path, mode=train_mode)
    print("load data done")

    info = "mode = " + train_mode + "\n"
    info = info + "train num = {}".format(len(img_list)) + "\n"
    info = info + args.root_path + "\n"
    info = info + "batch size = {}".format(args.batch_size) + "\n\n"
    print(info)
    loss_file.write(info)
    loss_file.flush()
    val_dice_max = 0.0
    epoch_max_index = 0
    for epoch in range(0, args.epochs):
        adjust_lr(seg_optimizer, epoch, args.seg_lr)
        epoch_total_loss = []
        for step in range(args.steps_per_epoch):
            batch_img_c, batch_ehc, batch_lbl = get_batch_patches(all_data_clec, args.batch_size, np.array(args.patch))
            if args.input_mode == 'ori':
                batch_img = batch_img_c
            elif args.input_mode == 'ehc':
                batch_img = batch_ehc
            else:
                batch_img = np.concatenate([batch_img_c, batch_ehc], axis=1)
            batch_img = torch.from_numpy(batch_img).to(device).float()
            batch_lbl = batch_lbl.astype(np.int64).squeeze(1)

            gt_label = torch.from_numpy(split_seg(batch_lbl).astype(np.int64)).cuda()
            pred_soft = seg_model(batch_img)
            pred_loss = dice_loss(pred_soft, gt_label)
            epoch_total_loss.append(pred_loss.item())
            print("epoch:{}/{}, step:{}/{}, pred_loss:{}".format(epoch + 1, args.epochs, step + 1, args.steps_per_epoch, pred_loss.item()))
            if step % 10 == 0:
                loss_file.write("epoch:{}/{}, step:{}/{}, pred_loss:{}\n".format(epoch + 1, args.epochs, step + 1, args.steps_per_epoch, pred_loss.item()))
                loss_file.flush()
            seg_optimizer.zero_grad()
            pred_loss.backward()
            seg_optimizer.step()
        epoch_dice = segmentation_validation(seg_model, all_data_clec, 1)
        epoch_info = "{}/{}, average loss:{}, dice:{}".format(epoch, args.epochs, np.mean(epoch_total_loss), epoch_dice)
        if epoch_dice > val_dice_max:
            val_dice_max = epoch_dice
            epoch_max_index = epoch
            torch.save(seg_model.state_dict(), model_dir + 'bestest.pth')
        print(epoch_max_index, val_dice_max)
        epoch_info += "\n max epoch:{}, max dice:{}".format(epoch_max_index, val_dice_max)
        print(epoch_info)
        loss_file.write("\n")
        loss_file.write(epoch_info)
        loss_file.write("\n\n")
        loss_file.flush()
        if epoch == args.epochs - 1 or ((epoch + 1) % 200 == 0):
            torch.save(seg_model.state_dict(), model_dir +'lastest.pth')

def segmentation_validation(model, all_data_clec, batch_size):
    dice_list = []
    for i in range(0, 10):
        batch_img_c, batch_ehc, batch_lbl = get_batch_patches(all_data_clec, batch_size, np.array(args.patch))
        if batch_lbl[0, 0].sum() == 0:
            continue
        if args.input_mode == 'ori':
            batch_img = batch_img_c
        elif args.input_mode == 'ehc':
            batch_img = batch_ehc
        else:
            batch_img = np.concatenate([batch_img_c, batch_ehc], axis=1)
        batch_img = torch.from_numpy(batch_img).to(device).float()
        with torch.no_grad():
            y_pred = model(batch_img)
        pred_soft = y_pred.cpu().detach().numpy()
        vol_result = np.argmax(pred_soft, axis=1)
        batch_fix_lbl_c = batch_lbl[0, 0].astype("int")
        vol_result = vol_result[0].astype("int")
        dice = compute_dice(copy.deepcopy(vol_result), copy.deepcopy(batch_fix_lbl_c))
        dice_list.append(dice)
    dice_mean = np.array(dice_list).mean()
    return dice_mean

def segmentation_test():
    load_name = model_dir + 'lastest.pth'
    out_dir = model_dir + 'result/'
    if 'train' in args.test_txt_path: 
        if args.test_over:
            out_dir = model_dir + 'result_train_over/'
        else:
            out_dir = model_dir + 'result_train/'
    os.makedirs(out_dir, exist_ok=True)
    if args.input_mode == 'ori' or args.input_mode == 'ehc':
        seg_model = get_segmentation_model(inchannel=1, load_name=load_name)
    elif args.input_mode == 'all':
        seg_model = get_segmentation_model(inchannel=2, load_name=load_name)
    else:
        return
    seg_model.eval()
    img_list = sorted(read_txt(args.root_path + args.test_txt_path))
    dice_list = []
    time_list, time_list2 = [], []
    for k in range(0, len(img_list)):
        print("{}/{}".format(k + 1, len(img_list)))
        all_data_clec = load_data_pairs(root_path=args.root_path, img_list=img_list[k:k+1], img_path='data2/' + target_mode + '_data/',
         label_path='data2/' + target_mode + '_label/', mode=target_mode)
        lbl_name = args.root_path + 'data2/' + target_mode + '_label/' + img_list[k] + '_' + target_mode.lower() + '_label.nii.gz'
        if not os.path.exists(lbl_name):
            lbl_name = args.root_path + 'data2/' + target_mode + '_label/' + img_list[k] + '_label.nii.gz'
        img_ = sitk.ReadImage(lbl_name)
        dim = np.array([img_.GetSize()[0], img_.GetSize()[1], img_.GetSize()[2]]).astype("int64")
        out_name = out_dir + img_list[k] + '_' + target_mode.lower() + '_label.nii.gz'
        img, ehc, lbl = all_data_clec[0][0], all_data_clec[1][0], all_data_clec[2][0]
        time_bb = time.time()
        ori_list = decompose_vol2cube(img, args.batch_size, np.array(args.patch))
        ehc_list = decompose_vol2cube(ehc, args.batch_size, np.array(args.patch))
        result_list = []
        time_b = time.time()
        for i in range(0, len(ori_list)):
            img_patch = ori_list[i]
            ehc_patch = ehc_list[i]
            if args.input_mode == 'ori':
                batch_img = img_patch
            elif args.input_mode == 'ehc':
                batch_img = ehc_patch
            else:
                batch_img = np.concatenate([img_patch, ehc_patch], axis=1)
            batch_img = torch.from_numpy(batch_img).to(device).float()
            with torch.no_grad():
                y_pred = seg_model(batch_img)
            pred_soft = y_pred.cpu().detach().numpy()
            result_list.append(pred_soft)
        time_e = time.time()
        time_list.append(time_e - time_b)
        result = compose_label_cube2vol(result_list, dim, args.batch_size, np.array(args.patch), class_n=2)
        time_ee = time.time()
        time_list2.append(time_ee - time_bb)
        if args.test_over:
            result[0] = result[0] - 0.47
            result[1] = result[1] + 0.47
        result = np.argmax(result, axis=0)
        dice1 = compute_dice(copy.deepcopy(result), copy.deepcopy(lbl))
        dice_list.append(dice1)
        result = result.swapaxes(0, 2)
        result = sitk.GetImageFromArray(result)
        result.CopyInformation(img_)
        sitk.WriteImage(result, out_name)
        print(img_list[k], dice1)
    print(np.array(time_list).mean(), np.array(time_list2).mean())
    print(np.array(dice_list).mean())

def get_param_num(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def main():
    if args.train:
        train()
    else:
        segmentation_test()

if __name__ == '__main__':
    main()
