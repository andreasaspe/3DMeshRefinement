import os
import shutil
import SimpleITK as sitk
import tools as tools
import numpy as np

src_path = "/storage/awias/saros_raw"
dst_path = "/data/awias/periseg/saros/NIFTI_collected"

os.makedirs(dst_path, exist_ok=True)

all_subjects = [x for x in os.listdir(src_path) if x.startswith("case")]

# all_subjects = ['case_500']
# all_subjects = ['case_020']

for subject in all_subjects:
    subject_id = subject.split('_')[1]
    print(subject)
    subject_folder = os.path.join(src_path, subject)

    src_img_path = os.path.join(subject_folder, "image.nii.gz")
    src_ts_highres_path = os.path.join(subject_folder, f"{subject}_heartchambershighres.nii.gz")
    src_ts_total_path = os.path.join(subject_folder, f"{subject}_total.nii.gz")
    src_ts_coronary_path = os.path.join(subject_folder, f"{subject}_coronaryarteries.nii.gz")
    src_ts_trunk_path = os.path.join(subject_folder, f"{subject}_trunkcavities.nii.gz")
    src_bodyregions_path = os.path.join(subject_folder, f"body-regions.nii.gz")

    try:
        img_sitk = sitk.ReadImage(src_img_path)
        trunk_sitk = sitk.ReadImage(src_ts_trunk_path)
        highres_sitk = sitk.ReadImage(src_ts_highres_path)
        total_sitk = sitk.ReadImage(src_ts_total_path)
        coronary_sitk = sitk.ReadImage(src_ts_coronary_path)
        bodyregions_sitk = sitk.ReadImage(src_bodyregions_path)
    except Exception as e: # One of the files do not exist. It is because the image is private - has to apply for it via TCIA. Have tried, but not available at the moment.
        print(f"ERROR reading files for {subject}: {e}")
        continue

    # Give all files the same information as bodyregions has
    img_sitk.CopyInformation(bodyregions_sitk)
    trunk_sitk.CopyInformation(bodyregions_sitk)
    highres_sitk.CopyInformation(bodyregions_sitk)
    total_sitk.CopyInformation(bodyregions_sitk)
    coronary_sitk.CopyInformation(bodyregions_sitk)


    img_arr = sitk.GetArrayFromImage(img_sitk)  # z, y, x

    mask = img_arr != -1024

    coords = np.where(mask)
    z_min, z_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()

    start = [int(x_min), int(y_min), int(z_min)]
    size = [
        int(x_max - x_min + 1),
        int(y_max - y_min + 1),
        int(z_max - z_min + 1)
    ]


    def crop(img):
        return sitk.RegionOfInterest(img, size=size, index=start)

    img_sitk = crop(img_sitk)
    trunk_sitk = crop(trunk_sitk)
    highres_sitk = crop(highres_sitk)
    total_sitk = crop(total_sitk)
    coronary_sitk = crop(coronary_sitk)
    bodyregions_sitk = crop(bodyregions_sitk)

    
    # Cut all sitk where the img has the value -1024 along either dimension - must be artificial

    # Read trunk cavities
    trunk = sitk.GetArrayFromImage(trunk_sitk)
    # Only inspect segment 3 (pericardium)
    trunk = (trunk == 3).astype(int)

    # Read highres heart chambers
    highres = sitk.GetArrayFromImage(highres_sitk)
    # Only inspect segment 2 (left atrium)
    highres = (highres == 2).astype(int)

    # Read body regions
    bodyregions = sitk.GetArrayFromImage(bodyregions_sitk)
    # Only segment 7 == pericardium
    bodyregions = (bodyregions == 7).astype(int)

    if bodyregions.sum() == 0:
        # print(f"No pericardium in body regions of {subject}, skipping...")
        continue
    if trunk.sum() == 0:
        # print(f"No pericardium in {subject}, skipping...")
        continue
    if highres.sum() == 0:
        # print(f"No left atrium in {subject}, skipping...")
        continue

    # Check if the mask hits the image boundaries (except for the upper boundary. That is fine) - note only five borders
    if (trunk[0, :, :].sum() > 0 or 
        trunk[:, 0, :].sum() > 0 or trunk[:, -1, :].sum() > 0 or
        trunk[:, :, 0].sum() > 0 or trunk[:, :, -1].sum() > 0):
        # print(f"Trunk cavities in {subject} hit one of the forbidden image boundaries, skipping...")
        continue




    # Find the largest z-coordinate where highres == 1
    z_indices = np.where(highres == 1)[0]
    if z_indices.size > 0:
        largest_z = z_indices.max()
    else:
        largest_z = None  # or handle as needed

    # left_atrium_highest_row = highres[largest_z,:,:]
    # if left_atrium_highest_row.sum() > 10:
    #     # print(f"Highres heart chambers in {subject} hit the upper image boundary, but that is allowed, so we will keep it. Number of pixels in highres at upper boundary: {left_atrium_highest_row.sum()}")
    #     pass

    # highres_upper = highres[-1,:,:]
    # if highres_upper.sum() > 0:
    #     # print(f"Highres heart chambers in {subject} hit the upper image boundary, but that is allowed, so we will keep it. Number of pixels in highres at upper boundary: {highres_upper.sum()}")
    #     pass

    # Check if the left atrium is cut
    if (highres[0,:,:].sum() > 0 or highres[-1,:,:].sum() > 0 or
        highres[:,0,:].sum() > 0 or highres[:,-1,:].sum() > 0 or
        highres[:,:,0].sum() > 0 or highres[:,:,-1].sum() > 0):
        # print(f"Highres heart chambers in {subject} hit one of the forbidden image boundaries, skipping...")
        continue

    # Survives till here means it has pericardium in body regions, trunk cavities and highres heart chambers, and does not hit forbidden boundaries (except for the allowed upper boundary). So we will include it.
    print(f"SURVIED ALL CHECKS AND WILL BE INCLUDED: {subject}")

    #Print physical size of the image in mm in each dimension
    spacing = trunk_sitk.GetSpacing()
    size = img_sitk.GetSize()

    # Get bounding box of the trunk mask
    nz = np.where(trunk > 0)
    z_min, z_max = nz[0].min(), nz[0].max()
    y_min, y_max = nz[1].min(), nz[1].max()
    x_min, x_max = nz[2].min(), nz[2].max()

    # Convert 40mm margin to voxels (spacing is in x,y,z order, array is z,y,x)
    crop_mm = [20, 20, 20]  # margin in mm for x, y, z
    margin_vox = [int(crop_mm[i] / spacing[i]) for i in range(3)]  # x, y, z

    # Apply margin and clamp to image boundaries
    x_start = max(0, x_min - margin_vox[0])
    x_end   = min(size[0], x_max + margin_vox[0] + 1)
    y_start = max(0, y_min - margin_vox[1])
    y_end   = min(size[1], y_max + margin_vox[1] + 1)
    z_start = max(0, z_min - margin_vox[2])
    z_end   = min(size[2], z_max + margin_vox[2] + 1)
    size_x = x_end - x_start
    size_y = y_end - y_start
    size_z = z_end - z_start

    # Extract ROI using voxel indices
    cropped_img_sitk = sitk.RegionOfInterest(img_sitk,
                                size=[int(size_x), int(size_y), int(size_z)],
                                index=[int(x_start), int(y_start), int(z_start)])
    cropped_trunk_sitk = sitk.RegionOfInterest(trunk_sitk,
                                size=[int(size_x), int(size_y), int(size_z)],
                                index=[int(x_start), int(y_start), int(z_start)])
    cropped_highres_sitk = sitk.RegionOfInterest(highres_sitk,
                                size=[int(size_x), int(size_y), int(size_z)],
                                index=[int(x_start), int(y_start), int(z_start)])
    cropped_total_sitk = sitk.RegionOfInterest(total_sitk,
                                size=[int(size_x), int(size_y), int(size_z)],
                                index=[int(x_start), int(y_start), int(z_start)])
    cropped_coronary_sitk = sitk.RegionOfInterest(coronary_sitk,
                                size=[int(size_x), int(size_y), int(size_z)],
                                index=[int(x_start), int(y_start), int(z_start)])
    cropped_bodyregions_sitk = sitk.RegionOfInterest(bodyregions_sitk,
                                size=[int(size_x), int(size_y), int(size_z)],
                                index=[int(x_start), int(y_start), int(z_start)])
    

    dst_img_path = os.path.join(dst_path, f"saros_{subject_id}_img.nii.gz")
    dst_ts_highres_path = os.path.join(dst_path, f"saros_{subject_id}_ts_heartchambershighres.nii.gz")
    dst_ts_total_path = os.path.join(dst_path, f"saros_{subject_id}_ts_total.nii.gz")
    dst_ts_coronary_path = os.path.join(dst_path, f"saros_{subject_id}_ts_coronaryarteries.nii.gz")
    dst_ts_trunk_path = os.path.join(dst_path, f"saros_{subject_id}_ts_trunkcavities.nii.gz")
    dst_bodyregions_path = os.path.join(dst_path, f"saros_{subject_id}_label.nii.gz")

    sitk.WriteImage(cropped_img_sitk, dst_img_path)
    sitk.WriteImage(cropped_highres_sitk, dst_ts_highres_path)
    sitk.WriteImage(cropped_total_sitk, dst_ts_total_path)
    sitk.WriteImage(cropped_coronary_sitk, dst_ts_coronary_path)
    sitk.WriteImage(cropped_trunk_sitk, dst_ts_trunk_path)
    sitk.WriteImage(cropped_bodyregions_sitk, dst_bodyregions_path)