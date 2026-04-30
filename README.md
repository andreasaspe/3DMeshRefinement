# 3D Mesh Refinement Pipeline

This repository contains a pipeline for preprocessing, segmentation, and mesh refinement of the SAROS dataset using TotalSegmentator and PyTorch3D.

---

## 1. Installation

Install dependencies:

```bash
pip install pytorch3d
```

Make sure you also have all required dependencies for:
- TotalSegmentator
- nibabel
- numpy
- scipy

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

### 4.1 Organise dataset

```bash
python organise_saros_files.py
```

This script:
- Filters cases
- Keeps only scans where the heart is visible
- Crops volumes accordingly
- Renames and reorganises files into a clean structure

---

### 4.2 Reorientation

```bash
python reorient_everything_parallel.py
```

This script:
- Reorients all volumes to the LPS (Left-Posterior-Superior) coordinate system
- Runs processing in parallel for efficiency

---

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
python calculate_metrics_saros.py
```

### 6.2 Summarise results

```bash
python summarise_metrics.py
```

This generates aggregated performance metrics across the dataset.

---

## Notes

- Ensure all preprocessing steps are completed before running refinement.
- Consistent LPS orientation is required for correct geometric alignment.
- Make sure TotalSegmentator outputs are correctly matched to each case ID.
