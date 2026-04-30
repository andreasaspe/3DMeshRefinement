# 3D Mesh Refinement Pipeline

This repository contains a pipeline for preprocessing, segmentation, and mesh refinement of the SAROS dataset using TotalSegmentator and PyTorch3D.

---

## 1. Installation

### 1.1 Pytorch3d

Installing PyTorch3D can be tricky due to strict compatibility requirements between Python, CUDA, and PyTorch.

First, choose a CUDA version compatible with your GPU. Then install a matching PyTorch build for that CUDA version by following the guide in [Pytorch](https://pytorch.org/).

Next, install a PyTorch3D version that matches both your Python and CUDA versions. Precompiled wheels are available here:
[PyTorch3D wheels](https://miropsota.github.io/torch_packages_builder/pytorch3d/).

### Naming convention

- `cuXXX` → CUDA version (e.g., `cu128` = CUDA 12.8)  
- `cpXXX` → Python version (e.g., `cp313` = Python 3.13)

Download the wheel that matches your setup and install it as described here:  
[torch_packages_builder GitHub](https://github.com/MiroPsota/torch_packages_builder).

In this project, we use Python 3.13.0 and CUDA 12.8 (tested on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU):

```
conda create -n pyt3d python=3.13.0
conda activate pyt3d
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d-0.7.9+pt2.7.0cu128-cp313-cp313-linux_x86_64.whl
```

### 1.2 Other dependencies

---

## 2. Download SAROS dataset

Download the SAROS dataset following the official instructions:

https://github.com/UMEssen/saros-dataset

---

## 3. Run TotalSegmentator

Run TotalSegmentator on the CT images using the following classes:

- heartchambers
- total
- trunkcavities
- coronaryarteries

Place the outputs into the corresponding case folders inside the SAROS dataset directory.

After processing, the dataset structure should look like:

```
saros_dataset/
├── case_000/
│   ├── image.nii.gz
│   ├── body-regions.nii.gz
│   ├── body-parts.nii.gz
│   ├── case_000_coronaryarteries.nii.gz
│   ├── case_000_heartchambershighres.nii.gz
│   ├── case_000_total.nii.gz
│   ├── case_000_trunkcavities.nii.gz
├── case_001/
├── ...
```

---

## 4. Preprocessing Pipeline

```bash
python prepare_saros_data.py
```

This script:
- Filters cases
- Keeps only scans where the heart is visible
- Crops volumes accordingly
- Renames and reorganises files into a clean structure
- Reorients all volumes to the LPS (Left-Posterior-Superior) coordinate system


## 5. Mesh Refinement

Run the refinement stage:

```bash
python run_refinement.py
```

This performs the actual 3D mesh refinement using the preprocessed SAROS data.

---

## 6. Evaluation

### 6.1 Compute metrics

```bash
python metrics_saros.py calculate
```

### 6.2 Summarise results

```bash
python metrics_saros.py summarize --csv-path /data/awias/periseg/saros/TS_pericardium/pytorch3d/metrics/best_grid_search_result_EXCLUDEGRID_EAT0/metrics_summary_taubin.csv
```

This generates aggregated performance metrics across the dataset.

---

## Notes

- Ensure all preprocessing steps are completed before running refinement.
- Consistent LPS orientation is required for correct geometric alignment.
- Make sure TotalSegmentator outputs are correctly matched to each case ID.
