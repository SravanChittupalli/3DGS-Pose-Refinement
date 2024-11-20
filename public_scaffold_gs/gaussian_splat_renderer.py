import os
import time
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import torchvision
from argparse import ArgumentParser
from tqdm import tqdm
from public_scaffold_gs.arguments import ModelParams, PipelineParams, get_combined_args
from public_scaffold_gs.gaussian_renderer import GaussianModel
from public_scaffold_gs.gaussian_renderer import render, prefilter_voxel
from public_scaffold_gs.scene.cameras import Camera
from public_scaffold_gs.utils.system_utils import searchForMaxIteration
from public_scaffold_gs.utils.graphics_utils import fov2focal

class GaussianSplatRenderer:
    def __init__(self, model_path, dataset, pipeline, iteration=-1):
        """
        Initialize the Gaussian Splat Renderer.
        
        Args:
            model_path (str): Path to the model.
            dataset: Dataset object containing scene parameters.
            pipeline: Pipeline configuration object.
            iteration (int): Iteration to load the model. Default is -1 for the latest iteration.
        """
        self.model_path = model_path
        self.dataset = dataset
        self.pipeline = pipeline
        self.iteration = iteration
        self.gaussians = None
        self.background = None
        self.loaded_iter = None
        self._initialize_gaussians()

    def _initialize_gaussians(self):
        """Set up Gaussian model and load weights."""
        self.gaussians = GaussianModel(
            self.dataset.feat_dim,
            self.dataset.n_offsets,
            self.dataset.voxel_size,
            self.dataset.update_depth,
            self.dataset.update_init_factor,
            self.dataset.update_hierachy_factor,
            self.dataset.use_feat_bank,
            self.dataset.appearance_dim,
            self.dataset.ratio,
            self.dataset.add_opacity_dist,
            self.dataset.add_cov_dist,
            self.dataset.add_color_dist,
        )
        self.gaussians.eval()

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Determine iteration to load
        if self.iteration == -1:
            self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        else:
            self.loaded_iter = self.iteration

        print(f"Loading trained model at iteration {self.loaded_iter}")
        self.gaussians.set_appearance(1)

        if self.loaded_iter:
            point_cloud_path = os.path.join(self.model_path, "point_cloud", f"iteration_{self.loaded_iter}")
            self.gaussians.load_ply_sparse_gaussian(os.path.join(point_cloud_path, "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(point_cloud_path)

    def render_gauss(self, pose):
        """
        Render the Gaussian splats given a camera pose.

        Args:
            pose (list): Camera pose matrix (4x4).

        Returns:
            torch.Tensor: Rendered image.
        """
        # Convert pose to camera parameters
        C2W = np.linalg.inv(pose)
        print(C2W.shape)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]
        # T[2] -= 0.5

        # Set up the camera
        camera = Camera(
            colmap_id=1,
            R=R.reshape(3, 3),
            T=T.reshape(3,),
            FoVx=1.0927237984095843,
            FoVy=0.8558020556023004,
            image=None,
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
            data_device=self.dataset.data_device,
        )

        # Prefilter visible voxels
        torch.cuda.synchronize()
        t0 = time.time()
        voxel_visible_mask = prefilter_voxel(camera, self.gaussians, self.pipeline, self.background)
        render_pkg = render(camera, self.gaussians, self.pipeline, self.background, visible_mask=voxel_visible_mask)
        print(render_pkg.keys())
        torch.cuda.synchronize()
        t1 = time.time()

        rendering = render_pkg["render"]
        depth_rendering = render_pkg["depth"]
        print(f"Render time: {t1 - t0:.2f}s")

        # Save rendering
        output_path = os.path.join('/data5/Scaffold-GS/tmp/', f'{1:05d}.png')
        torchvision.utils.save_image(rendering, output_path)
        output_path = os.path.join('/data5/Scaffold-GS/tmp/', f'{1:05d}_depth.png')
        torchvision.utils.save_image(depth_rendering, output_path)
        print(f"Saved rendering to {output_path}")
        return render_pkg


if __name__ == "__main__":
    # Set up command-line arguments
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    dataset = model.extract(args)
    pipeline = pipeline.extract(args)

    # Create Gaussian Splat Renderer
    renderer = GaussianSplatRenderer(args.model_path, dataset, pipeline, args.iteration)

    # Define a sample pose matrix
    # pose = [
    #     [0.9991131298689656, -0.04134714821139791, -0.00796034284591748, 0.36597491486219197],
    #     [0.04032583442494017, 0.8852106285145754, 0.4634392843110763, -0.505986382736346],
    #     [-0.012115312681568126, -0.46334928131984293, 0.8860929199011879, 1.2093802227889288],
    #     [0.0, 0.0, 0.0, 1.0],
    # ]
    pose = [
        [0.693069595793482, 0.602279776169557, -0.3961232214902424, 0.532567284105696],
        [-0.14874846010243856, 0.657165665540862, 0.7389229889855835, -0.8503398989996902],
        [0.7053569528996029, -0.45320233809125926, 0.5450497314418228, 0.5532944072186876],
        [0.0, 0.0, 0.0, 1.0],
        ]

    # Render using the given pose
    renderer.render_gauss(pose)