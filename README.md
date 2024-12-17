# **Geometry-based Robust Camera Pose Refinement with 3D Gaussian Splatting** : [(Final Report)](https://drive.google.com/file/d/17Ol0ls73b8Mw7C7Ec3Cg-oAKORLtKRU7/view?usp=sharing)

This project was done as part of course project for 16-822 Geometry based Methods for Vision at CMU.  

This repository contains an unofficial implementation of GSLoc paper [Arxiv](https://arxiv.org/abs/2408.11085). It includes modified versions of external modules and custom scripts for estimating camera pose using 3D Gaussian Splatting.

## Method

![Method](/vizualizations/gsloc.png)

---

## **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/SravanChittupalli/3DGS-Pose-Refinement.git
cd 3DGS-Pose-Refinement
```

---

### **2. Set Up the Conda Environment**

1. Ensure you have Conda installed. If not, download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. Create the environment using the `environment.yml` file provided in this repository:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate gsloc-env
   ```

4. (Optional) If you want to check the installed packages:

   ```bash
   conda list
   ```

5. Additional Installation
   ```
   cd public_scaffold_gs
   pip install submodules/diff-gaussian-rasterization

   cd submodules
   git clone https://github.com/ingra14m/diff-gaussian-rasterization-extentions.git --recursive
   git checkout filter-depth
   pip install -e .

   pip install submodules/simple-knn
   ```

---

## Datasets Setup

The **_marepo_** method has been evaluated using multiple published datasets:

- [Niantic Wayspots](https://nianticlabs.github.io/ace#dataset)
- [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)

We provide scripts in the `datasets` folder to automatically download and extract the data in a format that can be readily used by the **_marepo_** scripts.  
The format is the same used by the DSAC* codebase; see [here](https://github.com/vislearn/dsacstar#data-structure) for details.

> **Important: make sure you have checked the license terms of each dataset before using it.**

### 7-Scenes:

You can use the `datasets/setup_7scenes.py` scripts to download the data. To download and prepare the datasets:

```bash
cd datasets
# Downloads the data to datasets/7scenes_{chess, fire, ...}
./setup_7scenes.py
```

### **3. Download and Use Checkpoints**

Checkpoints should be downloaded into the directory `public_mast3r/checkpoints`.

#### **Checkpoints**


| Modelname   | Training resolutions | Head | Encoder | Decoder |
|-------------|----------------------|------|---------|---------|
| [`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | CatMLP+DPT | ViT-L | ViT-B |

You can check the hyperparameters we used to train these models in the [section: Our Hyperparameters](#our-hyperparameters).  
Make sure to check the licenses of the datasets we used.

To download a specific model, for example `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`, run the following:

```bash
mkdir -p public_mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P public_mast3r/checkpoints/
```

For these checkpoints, make sure to agree to the license of all the training datasets we used, in addition to CC-BY-NC-SA 4.0.  
The MapFree dataset license in particular is very restrictive. For more information, check [CHECKPOINTS_NOTICE](CHECKPOINTS_NOTICE).

---

#### **Pre-trained Models**

We also provide the following pre-trained models:

| Model (Linked)                                                                                  | Description                                 | 
|-------------------------------------------------------------------------------------------------|---------------------------------------------|
| ACE Heads                                                                                       |                                             |
| [wayspots_pretrain](https://storage.googleapis.com/niantic-lon-static/research/marepo/wayspots_pretrain.zip) | Pre-trained ACE Heads, Wayspots             |
| [pretrain](https://storage.googleapis.com/niantic-lon-static/research/marepo/pretrain.zip)      | Pre-trained ACE Heads, 7-Scenes & 12-Scenes |
| marepo models                                                                                   |                                             |
| [paper_model](https://storage.googleapis.com/niantic-lon-static/research/marepo/paper_model.zip) | marepo paper models                         |

To run inference with **_marepo_** on a test scene, the following components are required:

1. **ACE Encoder**:  
   The ACE encoder (`ace_encoder_pretrained.pt`) is pre-trained from the ACE paper and should be readily available in the repository by default. Download it at `public_marepo/`. 

2. **ACE Heads**:  
   - The ACE heads should be placed in either `public_marepo/logs/wayspots_pretrain/` or `public_marepo/logs/pretrain/`.  
   - We use the pre-trained ACE heads for scene-specific coordinate prediction.  

3. **marepo Pose Regression Models**:  
   - The marepo pose regression models should be placed in `logs/paper_model/`.

--- 

#### **Pre-trained Scaffold GS Models**

All the scene models trained from SfM GT pose can be downloaded from the [Link](https://drive.google.com/drive/folders/1FC8MYRbnstP82FDq_KkoRoOOwfTKN7ip?usp=sharing).  
Unzip `outputs.zip` and place the folder in `public_scaffold_gs` folder.  

If you want to train the Scaffold-GS models yourselves, we have provided the COLMAP models scaled to match the scale of GT Poses given by the authors of 7-Scenes. [Link](https://drive.google.com/drive/folders/1wz1OSRqgorcxc5IoIJjA0i6GhDTmAOZ9?usp=sharing)

### **4. Run the Code**

Run your scripts or modules as needed within the activated environment.  

`gsloc.py` is intended to run on the selected scenes in 7-Scenes and run pose refinement on MAREPO's initial guess and generate metrics. Metrics are stored under `output_metrics_v2`.  

```bash
python gsloc.py
```

**Make sure all files with paths hard-coded in `gsloc.py` are properly downloaded**  

**There might be some import errors. One of them might be from `public_scaffold_gs/gaussian_renderer/__init__.py` just change the 12th Line with your absolute library path.**

## Visualizations

![Visualization](/vizualizations/github_head_image.png)

---

## **Acknowledgments**

This project includes three external modules that have been modified for this implementation. 

- **`public_marepo`**: Based on [marepo](https://github.com/nianticlabs/marepo). 
- **`public_mast3r`**: Based on [mast3r](https://github.com/naver/mast3r).
- **`Scaffold GS`**: Based on [scaffold_gs](https://github.com/city-super/Scaffold-GS).

The modules were downloaded and adapted for the purposes of this project. Original `.git` directories have been removed to integrate them into this repository seamlessly.

---
