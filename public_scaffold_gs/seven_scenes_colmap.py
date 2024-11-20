import os
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

# Configuration
project_dir = "project"
images_dir = "/home/sravanc/Projects/7scenes_ace/pgt_7scenes_chess/train/rgb"  # Path to your images folder
poses_dir = "/home/sravanc/Projects/7scenes_ace/pgt_7scenes_chess/train/poses"    # Path to your 4x4 pose txt files
camera_model = "PINHOLE"       # Camera model
image_width = 640             # Image width
image_height = 480            # Image height
camera_params = [532.57, 531.54, 320.0, 240.0]  # fx, fy, cx, cy

# Create workspace directories
os.makedirs(os.path.join(project_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(project_dir, "sparse"), exist_ok=True)

# Copy images to the project directory (if necessary)
# for image_name in os.listdir(images_dir):
#     src = os.path.join(images_dir, image_name)
#     dest = os.path.join(project_dir, "images", image_name)
#     if not os.path.exists(dest):
#         os.symlink(src, dest)  # Symlink to save space; use shutil.copy for actual copy

# Initialize a COLMAP reconstruction object
reconstruction = pycolmap.Reconstruction()

# Add camera to the reconstruction
camera_id = 1
# Create the Camera object
camera = pycolmap.Camera(
    camera_id=camera_id,
    model=camera_model,
    width=image_width,
    height=image_height,
    params=camera_params
)

# Add the camera to the reconstruction
reconstruction.add_camera(camera)

# Add images with known poses
image_id = 1
for image_name in sorted(os.listdir(images_dir))[:5]:
    pose_path = os.path.join(poses_dir, f"{os.path.splitext(image_name)[0]}.txt".replace('color', 'pose'))
    print(pose_path)
    if not os.path.exists(pose_path):
        print(f"Pose file for {image_name} not found. Skipping.")
        continue

    # Load 4x4 pose matrix
    pose_matrix = np.loadtxt(pose_path)
    
    # Extract quaternion and translation from 4x4 pose matrix
    rotation_matrix = pose_matrix[:3, :3]
    translation_vector = pose_matrix[:3, 3]
    # quaternion = pycolmap.rotation_to_quaternion(rotation_matrix)
    r = Rotation.from_matrix(rotation_matrix)
    quaternion = r.as_quat()

    # Create the Image object
    image = pycolmap.Image(
        image_id=image_id,
        name=image_name,
        camera_id=camera_id,
        cam_from_world=pycolmap.Rigid3d(rotation=pycolmap.Rotation3d(quat=quaternion), translation=translation_vector)
    )

    # Add the image to the reconstruction
    reconstruction.add_image(image)
    image_id += 1

# Write reconstruction to disk
reconstruction.write(os.path.join(project_dir, "sparse"))

# Perform feature extraction
pycolmap.extract_features(
    database_path=os.path.join(project_dir, "database.db"),
    image_path=os.path.join(project_dir, "images"),
    camera_mode="SINGLE",
)

# Perform feature matching
pycolmap.match_sequential(
    database_path=os.path.join(project_dir, "database.db"),
)

# Run sparse reconstruction using known poses
# pycolmap.incremental_mapping(
#     database_path=os.path.join(project_dir, "database.db"),
#     image_path=os.path.join(project_dir, "images"),
#     output_path=os.path.join(project_dir, "sparse"),
#     known_poses=True,
# )

print("Sparse reconstruction completed successfully!")