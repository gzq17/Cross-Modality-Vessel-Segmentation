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

def get_one_img_patch(img, actual_dim, local_l, local_r, bbox_l, bbox_r, rand_angle, rot_flag=False, order=0):
    img_patch = np.zeros([actual_dim[0], actual_dim[1], actual_dim[2]]).astype("float32")
    img_patch[local_l[0]:local_r[0], local_l[1]:local_r[1], local_l[2]:local_r[2]] = img[bbox_l[0]:bbox_r[0],
                                                                                            bbox_l[1]:bbox_r[1],
                                                                                            bbox_l[2]:bbox_r[2]]
    if rot_flag and np.random.random() > 0.65:
        img_patch = rotate(img_patch, angle=rand_angle, axes=(1, 0), reshape=False, order=order)
    return img_patch

def load_data_pairs(root_path, img_list, img_path='', label_path='', mode=''):
    img_clec, enhance_clec, label_clec = [], [], []
    np.random.shuffle(img_list)
    k = 0
    for name in img_list:
        # img_name = root_path + img_path + name + '_' + mode.lower() + '.nii.gz'
        # ehc_name = root_path + img_path + name + '_' + mode.lower() + '_frangi.nii.gz'
        # lbl_name = root_path + label_path + name + '_transfer_label.nii.gz'
        img_name = root_path + img_path + name + '_' + mode.lower() + '.nii.gz'
        ehc_name = root_path + img_path + name + '_' + mode.lower() + '_frangi.nii.gz'
        lbl_name = root_path + label_path + name + '_' + mode.lower() + '_label.nii.gz'
        if not os.path.exists(lbl_name):
            k += 1
            lbl_name = root_path + 'data2/' + mode.upper() + '_label/' + name + '_' + mode.lower() + '_label.nii.gz'
        if not os.path.exists(img_name):
            img_name = root_path + 'data2/MRA_add/' + name + '.nii.gz'
            ehc_name = root_path + 'data2/MRA_add/' + name + '_frangi.nii.gz'
            lbl_name = root_path + label_path + name + '_label.nii.gz'
        print(lbl_name)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_name, sitk.sitkFloat32))
        ehc = sitk.GetArrayFromImage(sitk.ReadImage(ehc_name, sitk.sitkFloat32))
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(lbl_name, sitk.sitkInt32))
        print(lbl.max(), lbl.min())

        img = img.swapaxes(0, 2)
        ehc = ehc.swapaxes(0, 2)
        lbl = lbl.swapaxes(0, 2)
        if ehc.max() != 1.0 or ehc.min() != 0.0:
            print(name)
        high_low = img.max() - img.min()
        intensity_low = img.min() + high_low * 0.025
        intensity_high = img.max() - high_low * 0.025
        img = (img - intensity_low) / (intensity_high - intensity_low)
        img = np.clip(img, 0.0, 1.0)

        img_clec.append(img)
        enhance_clec.append(ehc)
        label_clec.append(lbl)
    print(k, len(img_list) - k)

    return (img_clec, enhance_clec, label_clec)


def get_batch_patches(all_data_clec, batch_size, patch_size, rot_flag=False):
    img_clec, enhance_clec, label_clec = all_data_clec[0], all_data_clec[1], all_data_clec[2]
    batch_img = np.zeros([batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float)
    batch_ehc = np.zeros([batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float)
    batch_lbl = np.zeros([batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float)

    for k in range(0, batch_size):
        rand_idx = np.floor(np.random.random() * len(img_clec)).astype("int")
        rand_img = img_clec[rand_idx]
        zoom_ratio = 1.0
        actual_dim = np.round(patch_size * zoom_ratio).astype("int")
        local_l, local_r, bbox_l, bbox_r = generateLocation(rand_img, patch_size, actual_dim, rand_img.shape)
        rand_angle = np.random.random() * 50 - 25
        batch_img[k, 0, :, :, :] = get_one_img_patch(img_clec[rand_idx], actual_dim, local_l, local_r,
         bbox_l, bbox_r, rand_angle, rot_flag=rot_flag, order=1)
        batch_ehc[k, 0, :, :, :] = get_one_img_patch(enhance_clec[rand_idx], actual_dim, local_l, local_r,
         bbox_l, bbox_r, rand_angle, rot_flag=rot_flag, order=1)
        batch_lbl[k, 0, :, :, :] = get_one_img_patch(label_clec[rand_idx], actual_dim, local_l, local_r,
         bbox_l, bbox_r, rand_angle, rot_flag=rot_flag, order=0)

    return batch_img, batch_ehc, batch_lbl

