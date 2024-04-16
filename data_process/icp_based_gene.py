import os, sys
import numpy as np
import SimpleITK as sitk
from skimage import transform
import skimage
import math
import open3d as o3d
import copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, skeletonize_3d
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/disk1/guozhanqiang/Cerebral/transfer_data2/')
parser.add_argument('--model_dir', type=str, default='model/')
parser.add_argument('--task', type=str, default='MRA2CTA', help='MRA2CTA, MRA2CTA')
parser.add_argument('--result_path', type=str, default='')
parser.add_argument('--result_over_path', type=str, default='')
args = parser.parse_args()
source_mode, target_mode = args.task[:3], args.task[4:7]
print(f'source mode:{source_mode}, target mode:{target_mode}')
out_path_parent = args.root_path + args.model_dir + args.task[:7] + '/'
os.makedirs(out_path_parent, exist_ok=True)
model_dir = out_path_parent + 'icp_generate_label/'
os.makedirs(model_dir, exist_ok=True)
if args.result_over_path == '':
    args.result_over_path = args.root_path + args.model_dir + args.task[:7] + '/checkpoint-source/result_train_over/'
if args.result_path == '':
    args.result_path = args.root_path + args.model_dir + args.task[:7] + '/checkpoint-source/result_train/'

def icp_reg():
    result_path = args.result_path
    result_over_path = args.result_over_path
    source_label_path = args.root_path + 'data2/' + source_mode + '_label/'
    lbl_path = args.root_path + 'data2/' + target_mode + '_label/'
    out_path = model_dir
    img_list = sorted(os.listdir(result_path))
    dice1_list, dice2_list = [], []
    for name in img_list:
        res_name = result_path + name
        res_over_name = result_over_path + name
        lbl_name = lbl_path + name
        print(lbl_name)
        source_name = source_label_path + name.replace(target_mode.lower(), source_mode.lower())
        img_ = sitk.ReadImage(res_name)
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(lbl_name))
        res_img = sitk.GetArrayFromImage(sitk.ReadImage(res_name))
        res_over_img = sitk.GetArrayFromImage(sitk.ReadImage(res_over_name))
        source_img = sitk.GetArrayFromImage(sitk.ReadImage(source_name))
        res_point = img2point(res_img)
        source_point = img2point(source_img)
        transfer_img = registration_fun(res_point, source_point, res_img)
        print('***** point icp done *****')
        # final_label = generate_label_arrocding_over2(transfer_img, res_img, res_over_img)
        final_label = generate_label_arrocding_over(transfer_img, res_over_img)
        dice1 = 2 * (lbl * res_img).sum() / (lbl.sum() + res_img.sum())
        dice2 = 2 * (lbl * final_label).sum() / (lbl.sum() + final_label.sum())
        print(name, dice1, dice2)
        dice1_list.append(dice1)
        dice2_list.append(dice2)
        out_name = out_path + name
        final_label_ = sitk.GetImageFromArray(final_label)
        final_label_.CopyInformation(img_)
        sitk.WriteImage(final_label_, out_name)
    print(np.array(dice1_list).mean(), np.array(dice2_list).mean())

def generate_label_arrocding_over(transfer_img, over_img):
    center_img = skeletonize_3d(over_img)
    add_points = according_transfer(transfer_img, center_img)
    label_points = get_final_result(over_img, add_points)
    center_transfer = np.zeros(transfer_img.shape)
    for i in range(0, add_points.shape[0]):
        x, y, z = int(add_points[i][0]), int(add_points[i][1]), int(add_points[i][2])
        center_transfer[x, y, z] = 1
    final_label = np.zeros(transfer_img.shape)
    for i in range(0, label_points.shape[0]):
        x, y, z = int(label_points[i][0]), int(label_points[i][1]), int(label_points[i][2])
        final_label[x, y, z] = 1
    return final_label

def generate_label_arrocding_over2(transfer_img, result, over_img):
    center_img = skeletonize_3d(over_img)
    result_center = skeletonize_3d(result)
    add_points = according_transfer(transfer_img, result_center)
    label_points = get_final_result(result, add_points)
    final_label = np.zeros(transfer_img.shape)
    for i in range(0, label_points.shape[0]):
        x, y, z = int(label_points[i][0]), int(label_points[i][1]), int(label_points[i][2])
        final_label[x, y, z] = 1
    add_points2 = get_points_from_center(transfer_img, center_img, result_center)
    label_points2 = get_final_result(over_img, add_points2)
    for i in range(0, label_points2.shape[0]):
        x, y, z = int(label_points2[i][0]), int(label_points2[i][1]), int(label_points2[i][2])
        final_label[x, y, z] = 1
    return final_label

def according_transfer(transfer_img, center_img):
    transfer_index = np.where(transfer_img == 1)
    transfer_arr = np.concatenate([np.array(transfer_index[0])[:, np.newaxis],
                                np.array(transfer_index[1])[:, np.newaxis],
                                np.array(transfer_index[2])[:, np.newaxis]], axis=1)
    center_index = np.where(center_img == 1)
    center_arr = np.concatenate([np.array(center_index[0])[:, np.newaxis],
                                np.array(center_index[1])[:, np.newaxis],
                                np.array(center_index[2])[:, np.newaxis]], axis=1)
    dist = cdist(transfer_arr, center_arr, metric='euclidean')
    print(dist.shape, center_arr.shape, transfer_arr.shape)
    add_points = np.array([[0, 0, 0]])
    for i in range(center_arr.shape[0]):
        dist_i = dist[:, i]
        min_dist = dist_i.min()
        if min_dist <= 4:
            one_point = np.array([[center_arr[i][0], center_arr[i][1], center_arr[i][2]]])
            add_points = np.concatenate([add_points, one_point], axis=0)
    add_points = add_points[1:]
    print(add_points.shape)
    return add_points

def get_final_result(frangi_img, add_points):
    frangi_index = np.where(frangi_img == 1)
    frangi_arr = np.concatenate([np.array(frangi_index[0])[:, np.newaxis],
                                 np.array(frangi_index[1])[:, np.newaxis],
                                 np.array(frangi_index[2])[:, np.newaxis]], axis=1)
    dist = cdist(frangi_arr, add_points, metric='euclidean')
    print(dist.shape, frangi_arr.shape, add_points.shape)
    label_points = np.array([[0, 0, 0]])
    for i in range(frangi_arr.shape[0]):
        dist_i = dist[i]
        min_dist = dist_i.min()
        if min_dist <= 4:
            one_point = np.array([[frangi_arr[i][0], frangi_arr[i][1], frangi_arr[i][2]]])
            label_points = np.concatenate([label_points, one_point], axis=0)
    label_points = label_points[1:]
    print(label_points.shape)
    return label_points

def get_points_from_center(transfer_img, center_img, result_center):
    transfer_index = np.where(transfer_img == 1)
    transfer_arr = np.concatenate([np.array(transfer_index[0])[:, np.newaxis],
                                np.array(transfer_index[1])[:, np.newaxis],
                                np.array(transfer_index[2])[:, np.newaxis]], axis=1)
    center_index = np.where(center_img == 1)
    center_arr = np.concatenate([np.array(center_index[0])[:, np.newaxis],
                                np.array(center_index[1])[:, np.newaxis],
                                np.array(center_index[2])[:, np.newaxis]], axis=1)
    result_center_index = np.where(result_center == 1)
    result_center_arr = np.concatenate([np.array(result_center_index[0])[:, np.newaxis],
                                np.array(result_center_index[1])[:, np.newaxis],
                                np.array(result_center_index[2])[:, np.newaxis]], axis=1)
    center_dist = cdist(center_arr, result_center_arr, metric='euclidean')
    dist = cdist(transfer_arr, center_arr, metric='euclidean')
    add_points = np.array([[0, 0, 0]])
    for i in range(0, center_arr.shape[0]):
        center_dist_i = center_dist[i]
        if center_dist_i.min() < 1.5:
            continue
        dist_i = dist[:, i]
        if dist_i.min() <= 4:
            one_point = np.array([[center_arr[i][0], center_arr[i][1], center_arr[i][2]]])
            add_points = np.concatenate([add_points, one_point], axis=0)
    add_points = add_points[1:]
    print(add_points.shape)
    return add_points

def img2point(img):
    index = np.where(img == 1)
    index_arr = np.concatenate([np.array(index[0])[:, np.newaxis],
                                      np.array(index[1])[:, np.newaxis],
                                      np.array(index[2])[:, np.newaxis]], axis=1)
    return index_arr

def registration_fun(target_points, source_points, img, post=False):
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)

    n = np.asarray(target.points).shape[0]
    acc_res, mat = register_target_with_source(target, source)
    mat_inv = np.linalg.inv(mat)
    target.paint_uniform_color([0, 0.651, 0.929])
    source.paint_uniform_color([1, 0.706, 0])
    source_point_ori = copy.deepcopy(np.asarray(source.points))
    mix_pcd = target + source.transform(mat)
    o3d.io.write_point_cloud('point_cloud_2.ply', mix_pcd)

    pc = np.asarray(mix_pcd.points)
    print(pc.shape)
    points1 = pc[:n, :]
    points2 = pc[n:, :]
    print(points1[-1], points2[0])
    result = np.zeros(img.shape)
    for i in range(points2.shape[0]):
        x, y, z = int(points2[i][0] + 0.5), int(points2[i][1] + 0.5), int(points2[i][2] + 0.5)
        if x < 0 or y < 0 or z < 0:
            continue
        if x >= result.shape[0] or y >= result.shape[1] or z >= result.shape[2]:
            continue
        result[x, y, z] = 1
    if post:
        post_result = copy.deepcopy(result)
        post_result[post_result == 0] = 2
        post_result[post_result == 1] = 0
        post_result[post_result == 2] = 1
        dis = distance_transform_edt(post_result)
        dis[dis <= 2] = 1
        dis[dis > 2] = 0
        index = np.where(dis == 1)
        index_arr = np.concatenate([np.array(index[0])[:, np.newaxis],
                                    np.array(index[1])[:, np.newaxis],
                                    np.array(index[2])[:, np.newaxis]], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(index_arr)
        tar_arr = copy.deepcopy(index_arr)
        ori_arr = pcd.transform(mat_inv)
        ori_arr = np.asarray(ori_arr.points)
        new_result = copy.deepcopy(result)
        for i in tqdm(range(0, tar_arr.shape[0])):
            x, y, z = round(tar_arr[i][0]), round(tar_arr[i][1]), round(tar_arr[i][2])
            patch = result[x-1:x+2, y-1:y+2, z-1:z+2]
            if result[x, y, z] == 1:
                continue
            ori_point = ori_arr[i][np.newaxis, :]
            dist = cdist(source_point_ori, ori_point).min()
            if dist < 1 and patch.mean() >= 0.3:
                new_result[x, y, z] = 1
        return new_result
    return result
   
def register_target_with_source(target_pcd, source_pcd, voxel_size=2):

    target_down = target_pcd.voxel_down_sample(voxel_size)
    source_down = source_pcd.voxel_down_sample(voxel_size)
    
    target_down_small = target_pcd.voxel_down_sample(voxel_size)
    source_down_small = source_pcd.voxel_down_sample(voxel_size)
    
    
    target_down = estimate_normals(target_down, voxel_size * 10)
    source_down = estimate_normals(source_down, voxel_size * 10)
    
    target_down_small = estimate_normals(target_down_small, voxel_size * 2)
    source_down_small = estimate_normals(source_down_small, voxel_size * 2)
    
    target_down_fpfh = extract_fpfh(target_down, voxel_size * 5)
    source_down_fpfh = extract_fpfh(source_down, voxel_size * 5)
    
    ransac_res = execute_global_registration(source_down, 
                                                   target_down, 
                                                   source_down_fpfh, 
                                                   target_down_fpfh, 
                                                   voxel_size * 5)

    acc_res = refine_registration(source_down_small, 
                                          target_down_small, 
                                          ransac_res.transformation, 
                                          voxel_size)
    print(acc_res)
    print(acc_res.transformation)

    mat = acc_res.transformation
    return acc_res, mat

def estimate_normals(pcd, radius):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return pcd

def extract_fpfh(pcd, radius):
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    return pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, 
                                target_fpfh, distance_threshold):
   result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
       source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
       o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
           o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
           o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
               distance_threshold)], o3d.pipelines.registration.RANSACConvergenceCriteria(5000000, 500))
   return result

def refine_registration(source, target, init_trans, distance_threshold):
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

if __name__ == '__main__':
    icp_reg()
