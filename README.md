# **GSLoc Unofficial Implementation**

This repository contains an unofficial implementation of GSLoc. It includes modified versions of external modules and custom scripts for estimating camera pose using 3D Gaussian Splatting.

---

## **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/SravanChittupalli/GSLoc-Unofficial-Implementation.git
cd GSLoc-Unofficial-Implementation
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

---

## Datasets Setup

The **_marepo_** method has been evaluated using multiple published datasets:

- [Niantic Wayspots](https://nianticlabs.github.io/ace#dataset)
- [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Stanford 12-Scenes](https://graphics.stanford.edu/projects/reloc/)

We provide scripts in the `datasets` folder to automatically download and extract the data in a format that can be readily used by the **_marepo_** scripts.  
The format is the same used by the DSAC* codebase; see [here](https://github.com/vislearn/dsacstar#data-structure) for details.

> **Important: make sure you have checked the license terms of each dataset before using it.**

### {7, 12}-Scenes:

You can use the `datasets/setup_{7,12}scenes.py` scripts to download the data. To download and prepare the datasets:

```bash
cd datasets
# Downloads the data to datasets/7scenes_{chess, fire, ...}
./setup_7scenes.py
# Downloads the data to datasets/12scenes_{apt1_kitchen, ...}
./setup_12scenes.py
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

### **4. Run the Code**

Run your scripts or modules as needed within the activated environment.

```bash
python gsloc.py
```

**Make sure all files with paths hard-coded in `gsloc.py` are properly downloaded**  

**There might be some import errors. One of them might be from `public_scaffold_gs/gaussian_renderer/__init__.py` just change the 12th Line with your absolute library path.**

---

## **Acknowledgments**

This project includes two external modules that have been modified for this implementation. 

- **`public_marepo`**: Based on [marepo](https://github.com/nianticlabs/marepo). 
- **`public_mast3r`**: Based on [mast3r](https://github.com/naver/mast3r).

The modules were downloaded and adapted for the purposes of this project. Original `.git` directories have been removed to integrate them into this repository seamlessly.

---
