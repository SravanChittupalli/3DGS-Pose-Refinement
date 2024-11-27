import math
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import pytorch3d.transforms as transforms
from torch.cuda.amp import autocast


def compute_pose_error_new(out_pose, gt_pose_44):
    '''
    same as ACE
    out_pose: torch.Tensor [B,3,4] or [B,4,4]
    gt_pose_44: torch.Tensor [B,3,4] or [B,4,4]
    return: torch tensor t_err [B], r_err [B]
    '''
    # torch.set_printoptions(precision=32)
    # breakpoint()
    # if out_pose.get_device() != gt_pose_44.get_device():
    #     print("we put gt_pose_44 to same device with out_pose")
    #     gt_pose_44 = gt_pose_44.to(out_pose.device)

    # Calculate translation error.
    t_err = torch.norm(gt_pose_44[:,0:3, 3] - out_pose[:,0:3, 3], dim=1).float()

    # Rotation error.
    r_err = torch.matmul(out_pose[:,:3,:3], gt_pose_44[:,:3,:3].transpose(1, 2))
    # Compute angle-axis representation.
    r_err = transforms.rotation_conversions.matrix_to_axis_angle(r_err)
    # Extract the angle.
    r_err = torch.linalg.norm(r_err, dim=1) * 180 / math.pi
    return t_err, r_err

def compute_stats_on_errors(t_err, r_err, rErrs, tErrs, pct10_5, pct5, pct2, pct1, pct500_10, pct50_5, pct25_2):
    '''
    compute stats on errors
    t_err:  torch.tensor() [B] computed translation errors
    r_err:  torch.tensor() [B] computed rotation errors
    rErrs: list records on epoch
    tErrs: list records on epoch
    pct10_5: counters
    ...
    return
    '''
    for idx, (te, re) in enumerate(zip(t_err, r_err)):
        te = te.cpu().item()
        re = re.cpu().item()
        rErrs.append(re)
        tErrs.append(te * 100)

        # check thresholds
        if re < 5 and te < 0.1: # 10cm/5deg
            pct10_5 += 1
        if re < 5 and te < 0.05:  # 5cm/5deg
            pct5 += 1
        if re < 2 and te < 0.02:  # 2cm/2deg
            pct2 += 1
        if re < 1 and te < 0.01:  # 1cm/1deg
            pct1 += 1

        # more loose thresholds
        if re < 10 and te < 5:  # 5m/10deg
            pct500_10 += 1
        if re < 5 and te < 0.5:  # 50cm/5deg
            pct50_5 += 1
        if re < 2 and te < 0.25:  # 25cm/2deg
            pct25_2 += 1
    return rErrs, tErrs, pct10_5, pct5, pct2, pct1, pct500_10, pct50_5, pct25_2

def batch_frame_test_time_error_computation(predict_pose, gt_pose_B44, rErrs, tErrs,
                                             pct10_5, pct5, pct2, pct1,
                                             pct500_10, pct50_5, pct25_2, num_batches):
    '''
    moved the previous test time functions here so that code is less ugly
    '''

    # here the t_err is in meters
    t_err, r_err = compute_pose_error_new(predict_pose[:, :3, :4], gt_pose_B44[:, :3, :4])

    # the tErrs is in centimeters because of te * 100 in compute_stats_on_errors()
    rErrs, tErrs, pct10_5, pct5, pct2, pct1, pct500_10, pct50_5, pct25_2 = \
        compute_stats_on_errors(t_err, r_err, rErrs, tErrs, pct10_5, pct5, pct2, pct1, pct500_10, pct50_5, pct25_2)
    
    num_batches += 1

    return rErrs, tErrs, num_batches, \
        pct10_5, pct5, pct2, pct1, \
        pct500_10, pct50_5, pct25_2



def evaluate_batch(predict_pose, gt_pose_B44):

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    # Percentage of frames predicted within certain thresholds from their GT pose.
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    # more loose thresholds
    pct500_10 = 0
    pct50_5 = 0
    pct25_2 = 0

    with torch.no_grad():
        batch_start_time = time.time()

        rErrs, tErrs, avg_batch_time, num_batches, \
            pct10_5, pct5, pct2, pct1, \
            pct500_10, pct50_5, pct25_2 \
            = batch_frame_test_time_error_computation(predict_pose, gt_pose_B44, rErrs, tErrs,
                                                        pct10_5, pct5, pct2, pct1,
                                                        pct500_10, pct50_5, pct25_2,
                                                        avg_batch_time, batch_start_time, num_batches)
        
import sys
def compute_error(rErrs, tErrs, total_frames, pct10_5, pct5, pct2, pct1, pct500_10, pct50_5, pct25_2): 
    # Compute median errors.
    median_rErr = np.median(rErrs)
    median_tErr = np.median(tErrs)
    mean_rErr = np.mean(rErrs)
    mean_tErr = np.mean(tErrs)


    # Compute final metrics.
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    pct500_10 = pct500_10 / total_frames * 100
    pct50_5 = pct50_5 / total_frames * 100
    pct25_2 = pct25_2 / total_frames * 100

    output = (
    "===================================================\n"
    f"Tested {total_frames} frames.\n\n"
    "Accuracy:\n"
    f"\t5m/10deg: {pct500_10:.2f}%\n"
    f"\t0.5m/5deg: {pct50_5:.2f}%\n"
    f"\t0.25m/2deg: {pct25_2:.2f}%\n"
    f"\t10cm/5deg: {pct10_5:.2f}%\n"
    f"\t5cm/5deg: {pct5:.2f}%\n"
    f"\t2cm/2deg: {pct2:.2f}%\n"
    f"\t1cm/1deg: {pct1:.2f}%\n\n"
    f"Median Error: {median_rErr:.2f} deg, {median_tErr:.2f} cm\n"
    f"Mean Error: {mean_rErr:.2f} deg, {mean_tErr:.2f} cm\n\n"
    f"{total_frames} {median_rErr:.2f} {median_tErr:.2f}\n"
    )

    # ANSI escape sequence to clear the screen
    sys.stdout.write("\033[2J\033[H")  # Clear screen and move cursor to top-left
    sys.stdout.write(output)
    sys.stdout.flush()