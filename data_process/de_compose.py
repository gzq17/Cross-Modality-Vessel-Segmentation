import copy
import numpy as np
import math

def start_end(dim, cube_size, overlap):
    fold = np.zeros((3,)).astype('int32')
    for i in range(0, 3):
        fold[i] = math.ceil(1 + (dim[i] - cube_size[i]) / (cube_size[i] - overlap[i]))
    x_s_list, x_e_list, y_s_list, y_e_list, z_s_list, z_e_list = [], [], [], [], [], []
    for i in range(fold[0]):
        x_s = i * (cube_size[0] - overlap[0])
        x_e = x_s + cube_size[0]
        if x_e > dim[0]:
            x_e = dim[0]
            x_s = dim[0] - cube_size[0]
        for j in range(fold[1]):
            y_s = j * (cube_size[1] - overlap[1])
            y_e = y_s + cube_size[1]
            if y_e > dim[1]:
                y_e = dim[1]
                y_s = dim[1] - cube_size[1]
            for k in range(fold[2]):
                z_s = k * (cube_size[2] - overlap[2])
                z_e = z_s + cube_size[2]
                if z_e > dim[2]:
                    z_e = dim[2]
                    z_s = dim[2] - cube_size[2]
                x_s_list.append(x_s)
                x_e_list.append(x_e)
                y_s_list.append(y_s)
                y_e_list.append(y_e)
                z_s_list.append(z_s)
                z_e_list.append(z_e)
    return x_s_list, x_e_list, y_s_list, y_e_list, z_s_list, z_e_list

def decompose_vol2cube(vol_data, batch_size, cube_size, overlap=None):
    if vol_data.shape[2] < cube_size[2]:
        cube_size[2] = 64
    cube_img_list = []
    if overlap is None:
        overlap = cube_size // 2
    dim = np.array([vol_data.shape[0], vol_data.shape[1], vol_data.shape[2]])
    x_s_list, x_e_list, y_s_list, y_e_list, z_s_list, z_e_list = start_end(dim, cube_size, overlap)
    for ind in range(0, len(x_e_list), batch_size):
        cube_img = np.zeros([batch_size, 1, cube_size[0], cube_size[1], cube_size[2]]).astype('float32')
        for cnt in range(0, batch_size):
            if ind + cnt == len(x_e_list):
                break
            x_s, x_e = x_s_list[ind + cnt], x_e_list[ind + cnt]
            y_s, y_e = y_s_list[ind + cnt], y_e_list[ind + cnt]
            z_s, z_e = z_s_list[ind + cnt], z_e_list[ind + cnt]
            cube_img[cnt, 0, :, :, :] = copy.deepcopy(vol_data[x_s:x_e, y_s:y_e, z_s:z_e])
        cube_img_list.append(cube_img)
    return cube_img_list

def compose_label_cube2vol(cube_list, dim, batch_size, cube_size, overlap=None, class_n=2):
    if dim[2] < cube_size[2]:
        cube_size[2] = 64
    prob_classes_mat = (np.zeros([class_n, dim[0], dim[1], dim[2]])).astype('float')
    idx_classes_mat = (np.zeros([class_n, dim[0], dim[1], dim[2]])).astype('int32')
    if overlap is None:
        overlap = cube_size // 2
    x_s_list, x_e_list, y_s_list, y_e_list, z_s_list, z_e_list = start_end(dim, cube_size, overlap)
    p_count = 0
    for ind in range(0, len(x_s_list), batch_size):
        cube_batch = cube_list[p_count]
        p_count += 1
        for cnt in range(0, batch_size):
            if ind + cnt == len(x_s_list):
                break
            x_s, x_e = x_s_list[ind + cnt], x_e_list[ind + cnt]
            y_s, y_e = y_s_list[ind + cnt], y_e_list[ind + cnt]
            z_s, z_e = z_s_list[ind + cnt], z_e_list[ind + cnt]
            for k in range(class_n):
                prob_classes_mat[k, x_s:x_e, y_s:y_e, z_s:z_e] = prob_classes_mat[k, x_s:x_e, y_s:y_e,
                                                                 z_s:z_e] + cube_batch[cnt, k, :, :, :]
                idx_classes_mat[k, x_s:x_e, y_s:y_e, z_s:z_e] = idx_classes_mat[k, x_s:x_e, y_s:y_e, z_s:z_e] + np.ones(
                    [cube_size[0], cube_size[1], cube_size[2]])
    img = prob_classes_mat / idx_classes_mat
    return img

def compute_dice(y_pred, y_lbl):
    dice_down = y_lbl.sum() + y_pred.sum()
    y_pred[y_pred == 0] = 2
    dice_up = (y_pred == y_lbl).sum()
    dice = 2 * dice_up / dice_down
    return dice
