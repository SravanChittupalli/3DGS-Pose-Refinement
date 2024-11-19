import json
import logging
import torch
from pathlib import Path
import numpy as np
from torchvision import transforms
from skimage import io, color
from .marepo.marepo_network import Regressor
from .dataset import CamLocDataset
from torch.cuda.amp import autocast

class PoseEstimator:
    def __init__(self, 
                 encoder_path, 
                 head_network_path, 
                 transformer_path, 
                 transformer_json, 
                 scene_path, 
                 image_resolution=480):
        """
        Initialize the PoseEstimator with required paths and parameters.

        :param encoder_path: Path to the encoder checkpoint.
        :param head_network_path: Path to the head network checkpoint.
        :param transformer_path: Path to the transformer checkpoint.
        :param transformer_json: Path to the transformer configuration JSON.
        :param scene_path: Path to the scene dataset.
        :param image_resolution: Target image resolution.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_resolution = image_resolution
        self.encoder_path = Path(encoder_path)
        self.head_network_path = Path(head_network_path)
        self.transformer_path = Path(transformer_path)
        self.transformer_json = transformer_json
        self.scene_path = Path(scene_path)

        # Load configurations and model
        self.test_dataset = CamLocDataset(
            root_dir=self.scene_path / "train", #"test"
            mode=0,
            image_height=self.image_resolution,
            center_crop=False,
            load_rgb=False
        )
        self.network = self._load_network()

    def _load_network(self):
        """Load the network with weights and configurations."""
        # Load network weights
        encoder_state_dict = torch.load(self.encoder_path, map_location="cpu")
        head_state_dict = torch.load(self.head_network_path, map_location="cpu")
        transformer_state_dict = torch.load(self.transformer_path, map_location="cpu")

        # Load transformer configuration
        with open(self.transformer_json) as f:
            config = json.load(f)
        config["transformer_pose_mean"] = torch.Tensor([0.0, 0.0, 0.0])
        default_img_H = self.test_dataset.default_img_H
        default_img_W = self.test_dataset.default_img_W
        config["default_img_HW"] = [default_img_H, default_img_W]

        network = Regressor.load_marepo_from_state_dict(
            encoder_state_dict, head_state_dict, transformer_state_dict, config
        )
        network.eval()
        network.to(self.device)

        return network

    def load_data(self, path_prefix, data_name):
        """
        Load image, pose, and intrinsics from the given paths.

        :param path_prefix: Prefix for the dataset paths.
        :param data_name: Name of the specific data instance.
        :return: Dictionary containing image, mask, pose, and intrinsics.
        """
        image_path = f"{path_prefix}/rgb/{data_name}.color.png"
        pose_path = f"{path_prefix}/poses/{data_name}.pose.txt"
        depth_path = f"{path_prefix}/depth/{data_name}.depth.png"
        intrinsics_path = f"{path_prefix}/calibration/{data_name}.calibration.txt"

        # Load image
        rgb_image = io.imread(image_path)
        depth_image = io.imread(depth_path)
        if len(rgb_image.shape) < 3:
            rgb_image = color.gray2rgb(rgb_image)

        # Load intrinsics
        k = np.loadtxt(intrinsics_path)
        if k.size == 1:
            focal_length = float(k)
            centre_point = None
        elif k.shape == (3, 3):
            k = k.tolist()
            focal_length = [k[0][0], k[1][1]]
            centre_point = [k[0][2], k[1][2]]
        else:
            raise Exception("Invalid calibration file format")

        f_scale_factor = self.test_dataset.image_height / rgb_image.shape[0]
        if centre_point:
            centre_point = [c * f_scale_factor for c in centre_point]
            focal_length = [f * f_scale_factor for f in focal_length]
        else:
            focal_length *= f_scale_factor


        pil_image = transforms.ToPILImage()(rgb_image)
        resized_image = transforms.Resize(self.test_dataset.image_height)(pil_image)
        image_tensor = self.test_dataset.image_transform(resized_image)
        image_mask = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]))

        # Load pose
        pose = np.loadtxt(pose_path)
        pose_tensor = torch.tensor(pose, dtype=torch.float32)

        # Create intrinsics matrix
        intrinsics = torch.eye(3)
        if centre_point:
            intrinsics[0, 0] = focal_length[0]
            intrinsics[1, 1] = focal_length[1]
            intrinsics[0, 2] = centre_point[0]
            intrinsics[1, 2] = centre_point[1]
        else:
            intrinsics[0, 0] = focal_length
            intrinsics[1, 1] = focal_length
            intrinsics[0, 2] = image_tensor.shape[2] / 2
            intrinsics[1, 2] = image_tensor.shape[1] / 2

        return {
            'rgb': rgb_image,
            'depth': depth_image,
            'image': image_tensor.unsqueeze(0),
            'image_mask': image_mask,
            'pose': pose_tensor.unsqueeze(0),
            'pose_inv': pose_tensor.inverse().unsqueeze(0),
            'intrinsics': intrinsics.unsqueeze(0),
        }

    def inference(self, input_data):
        """
        Perform inference given the input data.

        :param input_data: Dictionary containing image, pose, and intrinsics.
        :return: Predicted pose and ground truth pose.
        """
        with torch.no_grad():
            image_B1HW = input_data['image'].to(self.device)
            intrinsics_B33 = input_data['intrinsics'].to(self.device)
            gt_pose_B44 = input_data['pose']

            with autocast(enabled=True):
                features = self.network.get_features(image_B1HW)
                sc = self.network.get_scene_coordinates(features).float()

            with autocast(enabled=False):
                predict_pose = self.network.get_pose(sc, intrinsics_B33)
                predict_pose = predict_pose.float().cpu()

        return predict_pose, gt_pose_B44


if __name__ == "__main__":
    # Initialize paths
    encoder_path = "ace_encoder_pretrained.pt"
    head_network_path = "logs/pretrain/ace_models/7Scenes/7scenes_chess.pt"
    transformer_path = "logs/paper_model/marepo/marepo.pt"
    transformer_json = "transformer/config/nerf_focal_12T1R_256_homo.json"
    scene_path = "datasets/7scenes_chess"
    path_prefix = "datasets/7scenes_chess/train"
    data_name = "seq-03-frame-000017"

    # Instantiate and run the estimator
    estimator = PoseEstimator(encoder_path, head_network_path, transformer_path, transformer_json, scene_path)
    input_data = estimator.load_data(path_prefix, data_name)
    predicted_pose, ground_truth_pose = estimator.inference(input_data)

    print("Predicted Pose:\n", predicted_pose)
    print("Ground Truth Pose:\n", ground_truth_pose)
