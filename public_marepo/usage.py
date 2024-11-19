#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

import argparse
import logging
import time
from distutils.util import strtobool
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from marepo.marepo_network import Regressor
from dataset import CamLocDataset
from test_marepo_util import batch_frame_test_time_error_computation
import os
from skimage import io, color
from torchvision import transforms

_logger = logging.getLogger(__name__)

def _strtobool(x):
    return bool(strtobool(x))


def load_data_from_paths(image_path, pose_path, intrinsics_path, test_dataset):
    """Load image, pose, and intrinsics from file paths.

    Parameters:
        image_path: Path to the image file.
        pose_path: Path to the ground truth pose file.
        intrinsics_path: Path to the intrinsics/calibration file.
        image_height: Target image height for resizing.

    Returns:
        A dictionary containing processed image, mask, pose, intrinsics, and related data.
    """
    # Load image
    image = io.imread(image_path)
    if len(image.shape) < 3:
        image = color.gray2rgb(image)  # Convert grayscale to RGB if needed

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
        raise Exception("Calibration file must contain either a 3x3 camera "
                        "intrinsics matrix or a single float giving the focal length.")

    # Adjust focal length and principal point based on resizing
    f_scale_factor = test_dataset.image_height / image.shape[0]
    if centre_point:
        centre_point = [c * f_scale_factor for c in centre_point]
        focal_length = [f * f_scale_factor for f in focal_length]
    else:
        focal_length *= f_scale_factor

    # Resize image
    pil_image = transforms.ToPILImage()(image)
    resized_image = transforms.Resize(test_dataset.image_height)(pil_image)
    image_tensor = test_dataset.image_transform(resized_image)

    # Create binary mask for the image
    image_mask = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]))

    # Load pose
    pose = np.loadtxt(pose_path)
    pose_tensor = torch.tensor(pose, dtype=torch.float32)

    # Create the intrinsics matrix
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

    # Return all relevant data
    return {
        'image': image_tensor.unsqueeze(0),  # Add batch dimension
        'image_mask': image_mask,
        'pose': pose_tensor.unsqueeze(0),  # Add batch dimension
        'pose_inv': pose_tensor.inverse().unsqueeze(0),  # Add batch dimension
        'intrinsics': intrinsics.unsqueeze(0)
    }


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_path', type=Path,
                        default="",
                        help='path to the dataset folder, e.g. "~/storage/map_free_training_scenes/". '
                             'When this config is set, it means we are training with every scenes in the dataset')
    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')
    parser.add_argument('--head_network_path', type=Path,
                        default=Path(__file__).parent / "logs/wayspots_bears/wayspots_bears.pt",
                        help='file containing pre-trained ACE head weights')
    parser.add_argument('--dataset_head_network_path', type=Path,
                        default="",
                        help='path to the pre-trained ACE head weights of entire dataset, e.g. "logs/mapfree/". '
                             'When this config is set, it means we are training with every scenes in the dataset')
    parser.add_argument('--preprocessing', type=_strtobool, default=False,
                        help='use pretrained ACE networks to generate scene coordinate maps (Not used in testing)')
    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, '
                             'useful to separate different runs of a script')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of train set batch size')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='number of val set batch size')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='number of test set batch size')
    parser.add_argument('--use_half', type=_strtobool, default=False,
                        help='train with half precision')
    parser.add_argument('--trainskip', type=int, default=1,
                        help='uniformly subsample train set by 1/trainskip')
    parser.add_argument('--testskip', type=int, default=1,
                        help='uniformly subsample val/test set by 1/testskip')
    parser.add_argument('--transformer_json', type=str, default="../transformer/config/default.json",
                        help='file contain transformer config')
    parser.add_argument('--load_scheme2_sc_map', type=_strtobool, default=False,
                        help='use saved SC maps (subtract mean) and GT pose (subtract mean)'
                             'instead of use original SC map and GT pose')
    parser.add_argument('--datatype', type=str, default="test", choices=['train', 'val', 'test'],
                        help='dataset type: train means mapping data, test means query data')

    parser.add_argument('--center_crop', type=_strtobool, default=False,
                        help='Flag for datasetloader indicating images need center crop to make them proportional in size to MapFree data')

    parser.add_argument('--load_rgb', type=_strtobool, default=False,
                        help='Use 3 rgb channel images instead of using 1 channel gray image.')
    parser.add_argument('--data_split', type=str, default='test',
                        choices=('train', 'test'),help='data split')

    # noise jitter experiment to SC Map
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='noise ratio added to the sc map, in percentage 0-1')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='noise level added to the sc map, in cm, i.e.0.1m, 0.5m')

    device = "cuda"
    opt = parser.parse_args()
    # device = torch.device("cuda")
    # scene_path = Path(opt.scene)
    # encoder_path = Path(opt.encoder_path)
    # head_network_path = Path(opt.head_network_path)
    # transformer_path = Path(opt.network)

    repo_path = os.getcwd()
    ace_head_dir="logs/pretrain/ace_models/7Scenes"
    head_network_path = Path("logs/pretrain/ace_models/7Scenes/7scenes_chess.pt")
    transformer_path = Path("logs/paper_model/marepo/marepo.pt")
    encoder_path = Path("ace_encoder_pretrained.pt")
    scene_path = Path("datasets/7scenes_chess")
    transformer_json = "transformer/config/nerf_focal_12T1R_256_homo.json"

    test_dataset = CamLocDataset(
        root_dir=scene_path / "test",
        mode=0,  # Default for marepo, we don't need scene coordinates/RGB-D.
        image_height=opt.image_resolution, # 480
        center_crop=opt.center_crop,
        load_rgb=opt.load_rgb
    )

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")
    transformer_state_dict = torch.load(transformer_path, map_location="cpu")
    _logger.info(f"Loaded transformer weights from: {transformer_path}")

    # some configuration for the transformer
    f = open(transformer_json)
    config = json.load(f)
    f.close()
    config["transformer_pose_mean"] = torch.Tensor([0., 0., 0.])  # placeholder, load actual numbers later
    _logger.info(f"Loaded transformer config from: {transformer_json}")
    default_img_H = test_dataset.default_img_H  # we get default image H and W for position encoding
    default_img_W = test_dataset.default_img_W
    config["default_img_HW"] = [default_img_H, default_img_W]

    # Create regressor.
    network = Regressor.load_marepo_from_state_dict(encoder_state_dict, head_state_dict, transformer_state_dict, config)
    network.eval()

    # if network is trained by scheme 2, we should load the Ace Head mean stats
    print("Warning: we load mean again because we need to shift SC mean at test")
    load_scheme2_sc_map = True # For 7scenes.
    if load_scheme2_sc_map:
        network.transformer_head.transformer_pose_mean = network.heads.mean

    # Setup for evaluation.
    network = network.to("cuda")
    network.eval()

    # File paths for the image and ground truth pose
    path_prefix = "datasets/7scenes_chess/test"
    data_name = "seq-03-frame-000017"
    image_path = "{}/rgb/{}.color.png".format(path_prefix, data_name)  # Replace with actual image path
    pose_path = "{}/poses/{}.pose.txt".format(path_prefix, data_name) # Replace with actual GT pose path
    intrinsic_path = "{}/calibration/{}.calibration.txt".format(path_prefix, data_name)

    input_data = load_data_from_paths(image_path, pose_path, intrinsic_path, test_dataset)
    
    with torch.no_grad():

        image_B1HW = input_data['image']
        gt_pose_B44 = input_data['pose']
        intrinsics_B33= input_data['intrinsics']
        image_B1HW = image_B1HW.to(device, non_blocking=True)

        # Predict scene coordinates.
        with autocast(enabled=True):
            features = network.get_features(image_B1HW)
            sc = network.get_scene_coordinates(features).float() # [N,3,H,W]

        # Predict pose
        with autocast(enabled=False):
            predict_pose = network.get_pose(sc, intrinsics_B33.to(device))
            predict_pose = predict_pose.float().cpu()

        print(predict_pose, gt_pose_B44)