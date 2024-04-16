import open3d as o3d

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