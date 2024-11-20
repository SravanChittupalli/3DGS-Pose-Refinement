import os
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.system_utils import searchForMaxIteration

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

def render_gauss(model_path, iteration, views, gaussians, pipeline, background):
    torch.cuda.synchronize(); t0 = time.time()
    voxel_visible_mask = prefilter_voxel(views[0], gaussians, pipeline, background)
    render_pkg = render(views[0], gaussians, pipeline, background, visible_mask=voxel_visible_mask)
    torch.cuda.synchronize(); t1 = time.time()

    rendering = render_pkg["render"]
    torchvision.utils.save_image(rendering, os.path.join('/data5/Scaffold-GS/tmp/', '{0:05d}'.format(1) + ".png"))
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)
        
    dataset = model.extract(args)
    iteration = args.iteration
    pipeline = pipeline.extract(args)
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
            
        model_path = dataset.model_path
        loaded_iter = None

        if iteration:
            if iteration == -1:
                loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
            else:
                loaded_iter = iteration
                
            print("Loading trained model at iteration {}".format(loaded_iter))
            
        gaussians.set_appearance(1)
        
        if loaded_iter:
            gaussians.load_ply_sparse_gaussian(os.path.join(model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(loaded_iter),
                                                           "point_cloud.ply"))
            gaussians.load_mlp_checkpoints(os.path.join(model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(loaded_iter)))
        
        
        pose =  [[0.9991131298689656, -0.04134714821139791, -0.00796034284591748, 0.36597491486219197],
                [0.04032583442494017, 0.8852106285145754, 0.4634392843110763, -0.505986382736346],
                [-0.012115312681568126, -0.46334928131984293, 0.8860929199011879, 1.2093802227889288], 
                [0.0, 0.0, 0.0, 1.0]]
        
        C2W = np.linalg.inv(pose)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]
        print(T.shape)
        # T[2] -= 0.5
        
        camera_list = []
        # R, T, fovX, fovY calculations in dataset_readers.py: readColmapCameras()
        camera_list.append(Camera(colmap_id=1, R=R.reshape(3,3), T=T.reshape(3,), 
                  FoVx=1.0927237984095843, FoVy=0.8558020556023004, 
                  image=None, gt_alpha_mask=None,
                  image_name=None, uid=0, data_device=dataset.data_device))
        
        render_gauss(dataset.model_path, loaded_iter, camera_list, gaussians, pipeline, background)
        
        
        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

    