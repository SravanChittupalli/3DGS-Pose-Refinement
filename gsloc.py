import sys, os
marepo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'public_marepo')
mast3r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'public_mast3r')

# Add the marepo folder to the system path
sys.path.insert(0, marepo_path)
sys.path.insert(0, mast3r_path)

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


from public_marepo.marepo_inference import PoseEstimator
from public_mast3r.mast3r_inference import ImageMatcher

class CameraPoseRefinement:
    def __init__(self, encoder_path, head_network_path, transformer_path, transformer_json, scene_path, mast3r_model_path):
        """
        Initializes the pose refinement pipeline with paths to required models and datasets.
        
        :param encoder_path: Path to MARePo encoder model.
        :param head_network_path: Path to the head network model.
        :param transformer_path: Path to transformer model.
        :param transformer_json: Path to transformer configuration JSON file.
        :param scene_path: Path to the scene dataset.
        :param mast3r_model_path: Path to the MASt3R model checkpoint.
        """
        # Instantiate and load the PoseEstimator model for MARePo
        self.marepo_model = PoseEstimator(encoder_path, head_network_path, transformer_path, transformer_json, scene_path)

        # Instantiate the ImageMatcher model for MASt3R
        self.mast3r_model = ImageMatcher(mast3r_model_path)

    def rescale_keypoints(self, keypoints, original_shape, resized_shape):
        """
        Rescale keypoints from resized image dimensions to original image dimensions.
        
        Args:
        - keypoints (np.ndarray): Keypoints array of shape (n, 2) with (x, y) coordinates.
        - original_shape (tuple): Original shape of the image (H_original, W_original).
        - resized_shape (tuple): Resized shape of the image (H_resized, W_resized).
        
        Returns:
        - rescaled_keypoints (np.ndarray): Rescaled keypoints of shape (n, 2).
        """
        # Extract the original and resized dimensions
        H_original, W_original = original_shape
        H_resized, W_resized = resized_shape

        # Calculate the scaling factors
        scale_x = W_original / W_resized
        scale_y = H_original / H_resized

        # Rescale keypoints
        rescaled_keypoints = keypoints.copy()
        rescaled_keypoints[:, 0] = keypoints[:, 0] * scale_x  # Scale x-coordinates
        rescaled_keypoints[:, 1] = keypoints[:, 1] * scale_y  # Scale y-coordinates

        return rescaled_keypoints

    def visualize_keypoint_matching(self, query_image, reference_image, query_kps, reference_kps, save_path=None, n_viz=20):
        """
        Visualize keypoint matches between query and reference images.
        
        Args:
        - query_image (np.ndarray): The original query image as a NumPy array (H, W, C).
        - reference_image (np.ndarray): The original reference image as a NumPy array (H, W, C).
        - query_kps (np.ndarray): Keypoints from the query image of shape (n, 2).
        - reference_kps (np.ndarray): Keypoints from the reference image of shape (n, 2).
        - matches (list or np.ndarray): Array of shape (n, 2) containing matching indices for query and reference keypoints.
        - save_path (str, optional): Path to save the output image.
        - n_viz (int): Number of matches to visualize.
        """
        # Prepare for visualization
        H0, W0, _ = query_image.shape
        H1, W1, _ = reference_image.shape

        # Create a combined image by concatenating the query and reference images
        img0 = np.pad(query_image, ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(reference_image, ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img_combined = np.concatenate((img0, img1), axis=1)

        # Prepare visualization
        plt.figure(figsize=(15, 10))
        plt.imshow(img_combined)
        cmap = plt.get_cmap('jet')

        match_idx_to_viz = np.round(np.linspace(0, query_kps.shape[0] - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = query_kps[match_idx_to_viz], reference_kps[match_idx_to_viz]

        # Visualize matches
        for i in range(n_viz):
            (x0, y0) = viz_matches_im0[i].T  # Keypoint from query image
            (x1, y1) = viz_matches_im1[i].T  # Keypoint from reference image

            # Draw a line connecting the query and reference keypoints
            plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)

        # Save or show the result
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def project_to_3d(self, keypoints, depth_map, intrinsic_matrix):
        """
        Projects 2D keypoints to 3D points using the depth map and intrinsic matrix.
        
        :param keypoints: Array of 2D keypoints (Nx2 numpy array).
        :param depth_map: Depth map (HxW numpy array).
        :param intrinsic_matrix: Camera intrinsic matrix for this specific data.
        :return: Nx3 numpy array of 3D points.
        """
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        keypoints_3d = []
        valid_idx = []

        for x, y in keypoints:
            z = depth_map[int(y), int(x)]
            if z > 0:  # Only consider valid depth values
                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                keypoints_3d.append([X, Y, z])
                valid_idx.append(True)
            else:
                valid_idx.append(False)

        return np.array(keypoints_3d), np.array(valid_idx)

    def solve_pnp_ransac(self, query_kps, reference_kps_3d, intrinsic_matrix):
        """
        Solves PnP using RANSAC to estimate the refined pose.
        
        :param query_kps: Nx2 array of query 2D keypoints.
        :param reference_kps_3d: Nx3 array of 3D points from the reference image.
        :param intrinsic_matrix: Camera intrinsic matrix for this specific data.
        :return: Rotation vector (rvec), translation vector (tvec).
        """
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            reference_kps_3d,
            query_kps,
            intrinsic_matrix,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise RuntimeError("PnP RANSAC failed to find a solution.")
        return rvec, tvec
    def global_pose(self, reference_pose, rvec, tvec):
        # Convert rvec and tvec to transformation matrix (relative transformation)
        R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix (3x3)
        T = tvec.reshape(3, 1)       # Translation vector as column vector (3x1)
        relative_pose = np.hstack((R, T))  # Combine to form the 3x4 relative pose matrix

        # Compute the global pose of the query by multiplying the reference pose with the relative pose
        reference_pose_h = np.vstack((reference_pose, [0, 0, 0, 1]))  # Convert reference pose to 4x4 homogeneous matrix
        relative_pose_h = np.vstack((relative_pose, [0, 0, 0, 1]))    # Convert relative pose to 4x4 homogeneous matrix
        query_pose_h = np.dot(reference_pose_h, relative_pose_h)       # Global pose of the query (4x4 homogeneous)

        # Extract the 3x4 global pose from the homogeneous matrix
        query_pose = query_pose_h[:3, :]

        return query_pose
    
    def refine_pose(self, query_data, reference_data):
        """
        Refines the camera pose using feature matching and PnP.
        
        :param query_data_name: The name of the query data.
        :param reference_data_name: The name of the reference data.
        :return: Refined camera pose (rotation vector and translation vector).
        """

        # Extract intrinsic matrices from the loaded data
        query_intrinsic_matrix = query_data['intrinsics']
        reference_intrinsic_matrix = reference_data['intrinsics']

        # Ensure intrinsic matrices are the same for this setup
        if not np.allclose(query_intrinsic_matrix, reference_intrinsic_matrix):
            raise ValueError("Intrinsic matrices of query and reference images do not match!")

        # Extract intrinsic matrix to be used for PnP
        intrinsic_matrix = reference_intrinsic_matrix.numpy()[0]
        # Extract the reference global pose
        reference_pose = reference_data['pose'][0]  # Assuming reference_pose is a 3x4 matrix representing [R|t]
        # Initial pose estimation using MARePo
        initial_pose, _ = self.marepo_model.inference(query_data)

        query_image = query_data['rgb'] # (480, 640, 3)
        reference_rgbd = {
            'rgb': reference_data['rgb'],
            'depth': reference_data['depth']
        }
        query_image_shape = query_image.shape[:2]

        # Find and Match Keypoints with Mast3r.
        query_rgb = self.mast3r_model.load_image(query_image) 
        reference_rgb = self.mast3r_model.load_image(reference_rgbd['rgb']) # (1, 3, 384, 512)

        query_kps, reference_kps = self.mast3r_model.infer_and_match_tensors(query_rgb['img'], reference_rgb['img'], query_rgb['true_shape'], reference_rgb['true_shape'], visualize="test_small.png")
        query_kps = self.rescale_keypoints(query_kps, original_shape=query_image_shape, resized_shape=query_rgb['true_shape'][0])
        reference_kps = self.rescale_keypoints(reference_kps, original_shape=query_image_shape, resized_shape=reference_rgb['true_shape'][0])
        
        # Visualize (Test)
        self.visualize_keypoint_matching(query_image, reference_rgbd['rgb'], query_kps, reference_kps, save_path="test.png")

        reference_kps_3d, valid_idx = self.project_to_3d(reference_kps, reference_rgbd['depth'], intrinsic_matrix)
        query_kps_valid = query_kps[valid_idx]

        # Solve PnP using matched query 2D keypoints and 3D reference points
        rvec, tvec = self.solve_pnp_ransac(query_kps_valid.astype(np.float32), reference_kps_3d.astype(np.float32), intrinsic_matrix)
        
        # Output Global Pose.
        query_pose = self.global_pose(reference_pose, -rvec, -tvec/1000)

        return query_pose
        


# Example Usage
if __name__ == "__main__":
    # Paths to models and data
    encoder_path = "public_marepo/ace_encoder_pretrained.pt"
    head_network_path = "public_marepo/logs/pretrain/ace_models/7Scenes/7scenes_chess.pt"
    transformer_path = "public_marepo/logs/paper_model/marepo/marepo.pt"
    transformer_json = "public_marepo/transformer/config/nerf_focal_12T1R_256_homo.json"
    scene_path = "public_marepo/datasets/7scenes_chess"
    mast3r_model_path = "public_mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

    # Initialize the pose refinement pipeline
    gsloc = CameraPoseRefinement(
        encoder_path,
        head_network_path,
        transformer_path,
        transformer_json,
        scene_path,
        mast3r_model_path
    )

    # Refine pose given query and reference data names
    path_prefix = "public_marepo/datasets/7scenes_chess/train"
    query_data_name = "seq-01-frame-000276"
    reference_data_name = "seq-01-frame-000250"
    query_data = gsloc.marepo_model.load_data(path_prefix, query_data_name)
    reference_data = gsloc.marepo_model.load_data(path_prefix, reference_data_name)

    # Use marepo like this
    # predicted_pose, ground_truth_pose = gsloc.marepo_model.inference(query_data)

    refined_rvec, refined_tvec = gsloc.refine_pose(query_data, reference_data)

    print("Refined Rotation Vector:", refined_rvec)
    print("Refined Translation Vector:", refined_tvec)