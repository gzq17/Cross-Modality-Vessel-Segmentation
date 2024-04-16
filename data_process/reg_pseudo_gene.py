import os, sys
import numpy as np
import SimpleITK as sitk
from skimage import transform
import skimage
import math
import copy
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/disk1/guozhanqiang/Cerebral/transfer_data2/')
parser.add_argument('--model_dir', type=str, default='model/')
parser.add_argument('--task', type=str, default='MRA2CTA', help='MRA2CTA, MRA2CTA')
parser.add_argument('--reg_result_path', type=str, default='')
parser.add_argument('--stage1_path', type=str, default='')
args = parser.parse_args()
source_mode, target_mode = args.task[:3], args.task[4:7]
print(f'source mode:{source_mode}, target mode:{target_mode}')
out_path_parent = args.root_path + args.model_dir + args.task[:7] + '/'
os.makedirs(out_path_parent, exist_ok=True)
model_dir = out_path_parent + 'reg_generate_label/'
os.makedirs(model_dir, exist_ok=True)
if args.reg_result_path == '':
    args.reg_result_path = args.root_path + args.model_dir + args.task[:7] + '/checkpoint-reg/reg_result/'
if args.stage1_path == '':
    args.stage1_path = args.root_path + args.model_dir + args.task[:7] + '/checkpoint-source/result_train/'


def generate_label():
    reg_result_path = args.reg_result_path
    label_path = args.root_path + 'data2/' + target_mode + '_label/'
    out_path = model_dir
    os.makedirs(out_path, exist_ok=True)
    stage1_path = args.stage1_path
    if os.path.exists(reg_result_path) and not os.path.exists(out_path):
        os.mkdir(out_path)
    name_list = sorted(os.listdir(reg_result_path))
    dice_before_list, dice_after_list = [], []
    for name in name_list:
        if name [-1] != 'z':
            continue
        reg_result_name = reg_result_path + name
        stage1_name = stage1_path + name[:12] + '_' + target_mode.lower() + '_label.nii.gz'
        lbl_name = label_path + name[:12] + '_' + target_mode.lower() + '_label.nii.gz'
        reg_result = sitk.GetArrayFromImage(sitk.ReadImage(reg_result_name))
        stage1_result = sitk.GetArrayFromImage(sitk.ReadImage(stage1_name))
        img_ = sitk.ReadImage(lbl_name)
        lbl = sitk.GetArrayFromImage(img_)
        [label, num] = skimage.measure.label(stage1_result, return_num=True)
        new_result = np.zeros(stage1_result.shape)
        new_result[reg_result == 1] = 1
        for i in tqdm(range(1, num + 1)):
            temp = (label == i).sum()
            if temp < 100:
                continue
            temp = np.zeros(stage1_result.shape)
            temp[label == i] = 1
            A = ((temp == 1) & (reg_result == 1)).sum()
            B = (temp == 1).sum()
            pp = A / B
            if pp > 0.6:
                new_result[temp == 1] = 1
        dice_before = compute_dice(copy.deepcopy(reg_result), copy.deepcopy(lbl))
        dice_after = compute_dice(copy.deepcopy(new_result), copy.deepcopy(lbl))
        print(dice_before, dice_after)
        dice_before_list.append(dice_before)
        dice_after_list.append(dice_after)
        out_name = out_path + name[:12] + '_' + target_mode.lower() + '_label.nii.gz'
        new_result_ = sitk.GetImageFromArray(new_result)
        new_result_.CopyInformation(img_)
        sitk.WriteImage(new_result_, out_name)
    print(np.mean(dice_before_list), np.mean(dice_after_list))

def compute_dice(y_pred, y_lbl):
    dice_down = y_lbl.sum() + y_pred.sum()
    y_pred[y_pred == 0] = 2
    dice_up = (y_pred == y_lbl).sum()
    dice = 2 * dice_up / dice_down
    return dice

if __name__ == '__main__':
    generate_label()
