from tkinter import E
import numpy as np
import SimpleITK as sitk
import os, sys
import math
from scipy.ndimage import rotate, zoom
import copy

def read_txt(file_name=None):
    if file_name is None:
        return None
    name_list = []
    f = open(file_name, 'r')
    a = f.readlines()
    for name in a:
        name_list.append(name[:-1])
    return name_list

def norm_img(fix_ori):
    high_low = fix_ori.max() - fix_ori.min()
    intensity_low = fix_ori.min() + high_low * 0.025
    intensity_high = fix_ori.max() - high_low * 0.025
    fix_ori = (fix_ori - intensity_low) / (intensity_high - intensity_low)
    fix_ori = np.clip(fix_ori, 0.0, 1.0)
    return fix_ori

def load_data_pairs(root_path, img_list, move_path='', fix_path='', move_label_path='',
 fix_label_path='', fix_stage_path='', move='mra', fix='cta'):
    move_img_clec, move_ori_clec, fix_img_clec, fix_ori_clec, move_lbl_clec, fix_lbl_clec, fix_stage_clec = [], [], [], [], [], [], []
    other = 0
    np.random.shuffle(img_list)
    for name in img_list:
        move_img_name = root_path + move_path + name + '_' + move + '_frangi.nii.gz'
        move_ori_name = root_path + move_path + name + '_' + move + '.nii.gz'
        fix_img_name = root_path + fix_path + name + '_' + fix + '_frangi.nii.gz'
        fix_ori_name = root_path + fix_path + name + '_' + fix + '.nii.gz'

        move_img = sitk.GetArrayFromImage(sitk.ReadImage(move_img_name, sitk.sitkFloat32))
        move_ori = sitk.GetArrayFromImage(sitk.ReadImage(move_ori_name, sitk.sitkFloat32))
        fix_img = sitk.GetArrayFromImage(sitk.ReadImage(fix_img_name, sitk.sitkFloat32))
        fix_ori = sitk.GetArrayFromImage(sitk.ReadImage(fix_ori_name, sitk.sitkFloat32))

        move_img = move_img.swapaxes(0, 2)
        move_ori = move_ori.swapaxes(0, 2)
        fix_img = fix_img.swapaxes(0, 2)
        fix_ori = fix_ori.swapaxes(0, 2)

        fix_ori = norm_img(fix_ori)
        move_ori = norm_img(move_ori)

        move_img_clec.append(move_img)
        move_ori_clec.append(move_ori)
        fix_img_clec.append(fix_img)
        fix_ori_clec.append(fix_ori)

        move_lbl_name = root_path + move_label_path + name + '_' + move + '_label.nii.gz'
        fix_lbl_name = root_path + fix_label_path + name + '_' + fix + '_label.nii.gz'
        fix_stage1_name = root_path + fix_stage_path + name + '_' + fix + '_label.nii.gz'
        if not os.path.exists(fix_stage1_name):
            other += 1
            fix_stage1_name = root_path + fix_label_path + name + '_' + fix + '_label.nii.gz'
        print(fix_stage1_name)
        move_lbl = sitk.GetArrayFromImage(sitk.ReadImage(move_lbl_name, sitk.sitkInt32))
        fix_lbl = sitk.GetArrayFromImage(sitk.ReadImage(fix_lbl_name, sitk.sitkInt32))
        fix_stage1 = sitk.GetArrayFromImage(sitk.ReadImage(fix_stage1_name, sitk.sitkInt32))

        move_lbl = move_lbl.swapaxes(0, 2)
        fix_lbl = fix_lbl.swapaxes(0, 2)
        fix_stage1 = fix_stage1.swapaxes(0, 2)

        move_lbl_clec.append(move_lbl)
        fix_lbl_clec.append(fix_lbl)
        fix_stage_clec.append(fix_stage1)
    print("fix_ stage1 label:{}, real label:{}".format(len(img_list) - other, other))
    return (move_img_clec, move_ori_clec, fix_img_clec, fix_ori_clec, move_lbl_clec, fix_lbl_clec, fix_stage_clec)

def get_one_img_patch(img, actual_dim, local_l, local_r, bbox_l, bbox_r, rand_angle, rot_flag=False, order=0):
    img_patch = np.zeros([actual_dim[0], actual_dim[1], actual_dim[2]]).astype("float32")
    img_patch[local_l[0]:local_r[0], local_l[1]:local_r[1], local_l[2]:local_r[2]] = img[bbox_l[0]:bbox_r[0],
                                                                                            bbox_l[1]:bbox_r[1],
                                                                                            bbox_l[2]:bbox_r[2]]
    if rot_flag and np.random.random() > 0.65:
        img_patch = rotate(img_patch, angle=rand_angle, axes=(1, 0), reshape=False, order=order)
    return img_patch

def get_batch_patches(img_clec, batch_size, patch_size, rot_flag=False):
    move_img_clec, move_ori_clec, fix_img_clec, fix_ori_clec = img_clec[0], img_clec[1], img_clec[2], img_clec[3]
    move_lbl_clec, fix_lbl_clec, fix_stage_clec = img_clec[4], img_clec[5], img_clec[6]
    batch_move_img = np.zeros([batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float)
    batch_move_ori = np.zeros([batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float)
    batch_fix_img = np.zeros([batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float)
    batch_fix_ori = np.zeros([batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float)
    batch_move_lbl = np.zeros([batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float)
    batch_fix_lbl = np.zeros([batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float)
    batch_fix_stage = np.zeros([batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float)

    for k in range(0, batch_size):
        rand_idx = np.floor(np.random.random() * len(move_img_clec)).astype("int")
        rand_move = move_img_clec[rand_idx]
        zoom_ratio = 1.0
        actual_dim = np.round(patch_size * zoom_ratio).astype("int")
        local_l, local_r, bbox_l, bbox_r = generateLocation(rand_move, patch_size, actual_dim, rand_move.shape)
        rand_angle = np.random.random() * 50 - 25
        batch_move_img[k, 0, :, :, :] = get_one_img_patch(move_img_clec[rand_idx], actual_dim, local_l, local_r,
         bbox_l, bbox_r, rand_angle, rot_flag=rot_flag, order=1)
        batch_move_ori[k, 0, :, :, :] = get_one_img_patch(move_ori_clec[rand_idx], actual_dim, local_l, local_r,
         bbox_l, bbox_r, rand_angle, rot_flag=rot_flag, order=1)
        batch_fix_img[k, 0, :, :, :] = get_one_img_patch(fix_img_clec[rand_idx], actual_dim, local_l, local_r,
         bbox_l, bbox_r, rand_angle, rot_flag=rot_flag, order=1)
        batch_fix_ori[k, 0, :, :, :] = get_one_img_patch(fix_ori_clec[rand_idx], actual_dim, local_l, local_r,
         bbox_l, bbox_r, rand_angle, rot_flag=rot_flag, order=1)
        batch_move_lbl[k, 0, :, :, :] = get_one_img_patch(move_lbl_clec[rand_idx], actual_dim, local_l, local_r,
         bbox_l, bbox_r, rand_angle, rot_flag=rot_flag, order=0)
        batch_fix_lbl[k, 0, :, :, :] = get_one_img_patch(fix_lbl_clec[rand_idx], actual_dim, local_l, local_r,
         bbox_l, bbox_r, rand_angle, rot_flag=rot_flag, order=0)
        batch_fix_stage[k, 0, :, :, :] = get_one_img_patch(fix_stage_clec[rand_idx], actual_dim, local_l, local_r,
         bbox_l, bbox_r, rand_angle, rot_flag=rot_flag, order=0)

    return batch_move_img, batch_move_ori, batch_fix_img, batch_fix_ori, batch_move_lbl, batch_fix_lbl, batch_fix_stage

def generateLocation(rand_img, patch_size, actual_dim, img_shape):
    start = np.zeros((3,)).astype(np.int)
    for i in range(0, 3):
        start[i] = patch_size[i] // 2
    location = np.array(
        [np.floor(np.random.random() * (img_shape[i] - patch_size[i])).astype("int") for i in range(0, 3)])
    location += start
    half_size_l = np.array([actual_dim[0], actual_dim[1], actual_dim[2]], dtype="int") // 2
    half_size_r = np.array([actual_dim[0], actual_dim[1], actual_dim[2]], dtype="int") - half_size_l

    bbox_l_tmp = location - half_size_l
    bbox_r_tmp = location + half_size_r

    bbox_l = np.max((bbox_l_tmp, np.array([0, 0, 0])), axis=0)
    bbox_r = np.min((bbox_r_tmp, np.array(img_shape) - 1), axis=0)

    local_l = bbox_l - bbox_l_tmp
    local_r = actual_dim + bbox_r - bbox_r_tmp
    return local_l, local_r, bbox_l, bbox_r
