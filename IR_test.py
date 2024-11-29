import numpy as np
import os

import torch

from metric import compute_pose_error_new

path_query = "public_marepo/datasets/pgt_7scenes_chess/test/poses/"
path_reference_database = "public_marepo/datasets/pgt_7scenes_chess/train/poses/"

def find_nearest_camera(query_pose, ref_poses, alpha=1.0, beta=1.0):
    # Extract rotation and translation from query pose
    R_query = query_pose[0, :3, :3]  # Shape: 3x3
    t_query = query_pose[0, :3, 3]   # Shape: 3

    # Extract rotation and translation from reference poses
    R_ref = ref_poses[:, :3, :3]  # Shape: 4000x3x3
    t_ref = ref_poses[:, :3, 3]   # Shape: 4000x3

    # Compute rotation distance
    R_rel = torch.matmul(R_ref, R_query.T)  # Relative rotation, Shape: 4000x3x3
    trace_R_rel = torch.einsum('bii->b', R_rel)  # Trace of each relative rotation
    d_rot = torch.acos((trace_R_rel - 1) / 2)  # Geodesic distance, Shape: 4000

    # Compute translation distance
    d_trans = torch.norm(t_ref - t_query, dim=1)  # Euclidean distance, Shape: 4000

    print(d_rot, d_trans)
    # Combine distances
    d = alpha * d_rot + beta * d_trans  # Weighted distance, Shape: 4000

    # Find index of the nearest camera
    nearest_idx = torch.argmin(d).item()
    return nearest_idx

all_reference_poses = []
all_reference_poses_image_names = []
for reference_poses in os.listdir(path_reference_database):
    pose_path = path_reference_database + reference_poses
    pose = np.loadtxt(pose_path)
    pose_tensor = torch.tensor(pose, dtype=torch.float32).unsqueeze(0)
    all_reference_poses.append(pose_tensor)
    all_reference_poses_image_names.append(pose_path)
all_reference_poses = torch.cat(all_reference_poses, dim=0)
print(all_reference_poses.shape, len(all_reference_poses_image_names))
    
for query_pose in os.listdir(path_query)[:10]:
    pose_path = path_query + query_pose
    pose = np.loadtxt(pose_path)
    pose_tensor = torch.tensor(pose, dtype=torch.float32).unsqueeze(0)
    
    idx = find_nearest_camera(pose_tensor, all_reference_poses)
    # t_err, r_err = compute_pose_error_new(pose_tensor, all_reference_poses)
    # print(r_err/torch.max(r_err), t_err/torch.max(t_err))
    # d = r_err/torch.max(r_err) + t_err/torch.max(t_err)  # Weighted distance, Shape: 4000
    # # Find index of the nearest camera
    # idx = torch.argmin(d).item()
    
    print(pose_path)
    print(all_reference_poses_image_names[idx])
    
    
    