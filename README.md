# 3DMeshRefinement

Install pytorch3d

Download the SAROS dataset by following the instructions in this github: https://github.com/UMEssen/saros-dataset

Run TotalSegmentator classes heartchambers, total, trunkcavities and coronaryarteries on the CT images in the saros-dataaset. Place them in the folders for every fold in the saros dataset as we got from downloading it. Thus we get:

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

Run the code organise_saros_files.py to filter the files, select only the ones with the heart visible, crop them accordingly and rename and move.
Run the code reorient_everything_parallel.py to reorinet to 'LPS'.
Run the code 'Run_finement.py' to do the actual refinement.
Run the code calculate_metrics_saros to calculate metrics
Run the code summarise_metrics.py to summarise the metrics.
