import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import saros_utils as utils


DEFAULT_SRC = "/storage/awias/saros_raw"
DEFAULT_DST = "/data/awias/periseg/saros/NIFTI_collected"


def collect_saros_cases(src_path, dst_path, margin_mm=20):
    os.makedirs(dst_path, exist_ok=True)

    all_subjects = [x for x in os.listdir(src_path) if x.startswith("case")]

    for subject in all_subjects:
        subject_id = subject.split("_")[1]
        print(subject)
        subject_folder = os.path.join(src_path, subject)

        src_img_path = os.path.join(subject_folder, "image.nii.gz")
        src_ts_highres_path = os.path.join(subject_folder, f"{subject}_heartchambershighres.nii.gz")
        src_ts_total_path = os.path.join(subject_folder, f"{subject}_total.nii.gz")
        src_ts_coronary_path = os.path.join(subject_folder, f"{subject}_coronaryarteries.nii.gz")
        src_ts_trunk_path = os.path.join(subject_folder, f"{subject}_trunkcavities.nii.gz")
        src_bodyregions_path = os.path.join(subject_folder, "body-regions.nii.gz")

        try:
            img_sitk = sitk.ReadImage(src_img_path)
            trunk_sitk = sitk.ReadImage(src_ts_trunk_path)
            highres_sitk = sitk.ReadImage(src_ts_highres_path)
            total_sitk = sitk.ReadImage(src_ts_total_path)
            coronary_sitk = sitk.ReadImage(src_ts_coronary_path)
            bodyregions_sitk = sitk.ReadImage(src_bodyregions_path)
        except Exception as exc:
            print(f"ERROR reading files for {subject}: {exc}")
            continue

        # Normalize metadata to match body-regions
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
            int(z_max - z_min + 1),
        ]

        def crop(img):
            return sitk.RegionOfInterest(img, size=size, index=start)

        img_sitk = crop(img_sitk)
        trunk_sitk = crop(trunk_sitk)
        highres_sitk = crop(highres_sitk)
        total_sitk = crop(total_sitk)
        coronary_sitk = crop(coronary_sitk)
        bodyregions_sitk = crop(bodyregions_sitk)

        trunk = sitk.GetArrayFromImage(trunk_sitk)
        trunk = (trunk == 3).astype(int)

        highres = sitk.GetArrayFromImage(highres_sitk)
        highres = (highres == 2).astype(int)

        bodyregions = sitk.GetArrayFromImage(bodyregions_sitk)
        bodyregions = (bodyregions == 7).astype(int)

        if bodyregions.sum() == 0:
            continue
        if trunk.sum() == 0:
            continue
        if highres.sum() == 0:
            continue

        if (
            trunk[0, :, :].sum() > 0
            or trunk[:, 0, :].sum() > 0
            or trunk[:, -1, :].sum() > 0
            or trunk[:, :, 0].sum() > 0
            or trunk[:, :, -1].sum() > 0
        ):
            continue

        if (
            highres[0, :, :].sum() > 0
            or highres[-1, :, :].sum() > 0
            or highres[:, 0, :].sum() > 0
            or highres[:, -1, :].sum() > 0
            or highres[:, :, 0].sum() > 0
            or highres[:, :, -1].sum() > 0
        ):
            continue

        print(f"SURVIVED ALL CHECKS AND WILL BE INCLUDED: {subject}")

        spacing = trunk_sitk.GetSpacing()
        size = img_sitk.GetSize()

        nz = np.where(trunk > 0)
        z_min, z_max = nz[0].min(), nz[0].max()
        y_min, y_max = nz[1].min(), nz[1].max()
        x_min, x_max = nz[2].min(), nz[2].max()

        crop_mm = [margin_mm, margin_mm, margin_mm]
        margin_vox = [int(crop_mm[i] / spacing[i]) for i in range(3)]

        x_start = max(0, x_min - margin_vox[0])
        x_end = min(size[0], x_max + margin_vox[0] + 1)
        y_start = max(0, y_min - margin_vox[1])
        y_end = min(size[1], y_max + margin_vox[1] + 1)
        z_start = max(0, z_min - margin_vox[2])
        z_end = min(size[2], z_max + margin_vox[2] + 1)
        size_x = x_end - x_start
        size_y = y_end - y_start
        size_z = z_end - z_start

        cropped_img_sitk = sitk.RegionOfInterest(
            img_sitk,
            size=[int(size_x), int(size_y), int(size_z)],
            index=[int(x_start), int(y_start), int(z_start)],
        )
        cropped_trunk_sitk = sitk.RegionOfInterest(
            trunk_sitk,
            size=[int(size_x), int(size_y), int(size_z)],
            index=[int(x_start), int(y_start), int(z_start)],
        )
        cropped_highres_sitk = sitk.RegionOfInterest(
            highres_sitk,
            size=[int(size_x), int(size_y), int(size_z)],
            index=[int(x_start), int(y_start), int(z_start)],
        )
        cropped_total_sitk = sitk.RegionOfInterest(
            total_sitk,
            size=[int(size_x), int(size_y), int(size_z)],
            index=[int(x_start), int(y_start), int(z_start)],
        )
        cropped_coronary_sitk = sitk.RegionOfInterest(
            coronary_sitk,
            size=[int(size_x), int(size_y), int(size_z)],
            index=[int(x_start), int(y_start), int(z_start)],
        )
        cropped_bodyregions_sitk = sitk.RegionOfInterest(
            bodyregions_sitk,
            size=[int(size_x), int(size_y), int(size_z)],
            index=[int(x_start), int(y_start), int(z_start)],
        )

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


def _reorient_file(file_path, direction):
    if not file_path.endswith(".nii.gz"):
        return None

    try:
        file_sitk = sitk.ReadImage(file_path)
        file_sitk_reoriented = utils.reorient_sitk(file_sitk, direction)
        sitk.WriteImage(file_sitk_reoriented, file_path)
        return os.path.basename(file_path)
    except Exception as exc:
        return f"ERROR {os.path.basename(file_path)}: {exc}"


def reorient_folder(path, direction="LPS", workers=None):
    if workers is None:
        workers = max(os.cpu_count() - 1, 1)

    all_files = [os.path.join(path, f) for f in os.listdir(path)]

    worker = partial(_reorient_file, direction=direction)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in tqdm(
            executor.map(worker, all_files),
            total=len(all_files),
            desc="Reorienting files",
        ):
            pass


def main():
    parser = argparse.ArgumentParser(description="Prepare SAROS dataset for mesh refinement.")
    parser.add_argument("--src", default=DEFAULT_SRC, help="Source SAROS dataset folder")
    parser.add_argument("--dst", default=DEFAULT_DST, help="Output folder for collected NIfTI files")
    parser.add_argument("--mode", choices=["organize", "reorient", "both"], default="both")
    parser.add_argument("--direction", default="LPS", help="Target orientation for reorienting")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers for reorienting")
    parser.add_argument("--margin-mm", type=int, default=20, help="Crop margin in mm")

    args = parser.parse_args()

    if args.mode in ("organize", "both"):
        collect_saros_cases(args.src, args.dst, margin_mm=args.margin_mm)

    if args.mode in ("reorient", "both"):
        reorient_folder(args.dst, direction=args.direction, workers=args.workers)


if __name__ == "__main__":
    main()
