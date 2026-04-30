import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import saros_utils as utils


DEFAULT_SRC = "/storage/awias/saros_raw"
DEFAULT_DST = "/data/awias/periseg/saros/NIFTI_collected_test"


# ----------------------------
# SUBJECT PROCESSING (PARALLEL)
# ----------------------------
def _process_subject(subject, src_path, dst_path, margin_mm):
    subject_folder = os.path.join(src_path, subject)
    subject_id = subject.split("_")[1]

    try:
        img_sitk = sitk.ReadImage(os.path.join(subject_folder, "image.nii.gz"))
        trunk_sitk = sitk.ReadImage(os.path.join(subject_folder, f"{subject}_trunkcavities.nii.gz"))
        highres_sitk = sitk.ReadImage(os.path.join(subject_folder, f"{subject}_heartchambershighres.nii.gz"))
        total_sitk = sitk.ReadImage(os.path.join(subject_folder, f"{subject}_total.nii.gz"))
        coronary_sitk = sitk.ReadImage(os.path.join(subject_folder, f"{subject}_coronaryarteries.nii.gz"))
        bodyregions_sitk = sitk.ReadImage(os.path.join(subject_folder, "body-regions.nii.gz"))
    except Exception:
        return None

    # Align metadata
    for img in [trunk_sitk, highres_sitk, total_sitk, coronary_sitk]:
        img.CopyInformation(bodyregions_sitk)

    img_arr = sitk.GetArrayFromImage(img_sitk)
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

    trunk = (sitk.GetArrayFromImage(trunk_sitk) == 3).astype(np.uint8)
    highres = (sitk.GetArrayFromImage(highres_sitk) == 2).astype(np.uint8)
    bodyregions = (sitk.GetArrayFromImage(bodyregions_sitk) == 7).astype(np.uint8)

    if bodyregions.sum() == 0 or trunk.sum() == 0 or highres.sum() == 0:
        return None

    # Remove border-touching cases
    if (
        trunk[0].sum() > 0 or trunk[-1].sum() > 0 or
        trunk[:, 0].sum() > 0 or trunk[:, -1].sum() > 0 or
        trunk[:, :, 0].sum() > 0 or trunk[:, :, -1].sum() > 0
    ):
        return None

    if (
        highres[0].sum() > 0 or highres[-1].sum() > 0 or
        highres[:, 0].sum() > 0 or highres[:, -1].sum() > 0 or
        highres[:, :, 0].sum() > 0 or highres[:, :, -1].sum() > 0
    ):
        return None

    spacing = trunk_sitk.GetSpacing()
    size_img = img_sitk.GetSize()

    nz = np.where(trunk > 0)
    z_min, z_max = nz[0].min(), nz[0].max()
    y_min, y_max = nz[1].min(), nz[1].max()
    x_min, x_max = nz[2].min(), nz[2].max()

    margin_vox = [int(margin_mm / s) for s in spacing]

    x_start = max(0, x_min - margin_vox[0])
    x_end = min(size_img[0], x_max + margin_vox[0] + 1)
    y_start = max(0, y_min - margin_vox[1])
    y_end = min(size_img[1], y_max + margin_vox[1] + 1)
    z_start = max(0, z_min - margin_vox[2])
    z_end = min(size_img[2], z_max + margin_vox[2] + 1)

    roi_size = [x_end - x_start, y_end - y_start, z_end - z_start]

    def roi(img):
        return sitk.RegionOfInterest(
            img,
            size=roi_size,
            index=[x_start, y_start, z_start],
        )

    return {
        "id": subject_id,
        "img": roi(img_sitk),
        "trunk": roi(trunk_sitk),
        "highres": roi(highres_sitk),
        "total": roi(total_sitk),
        "coronary": roi(coronary_sitk),
        "label": roi(bodyregions_sitk),
    }


# ----------------------------
# MAIN COLLECTION FUNCTION
# ----------------------------
def collect_saros_cases_parallel(src_path, dst_path, margin_mm=20, workers=None):
    os.makedirs(dst_path, exist_ok=True)

    subjects = [x for x in os.listdir(src_path) if x.startswith("case")]

    if workers is None:
        workers = max(os.cpu_count() - 1, 1)

    worker_fn = partial(
        _process_subject,
        src_path=src_path,
        dst_path=dst_path,
        margin_mm=margin_mm,
    )

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(
            tqdm(
                ex.map(worker_fn, subjects),
                total=len(subjects),
                desc="Collecting SAROS cases",
            )
        )

    for res in results:
        if res is None:
            continue

        sid = res["id"]

        sitk.WriteImage(res["img"], os.path.join(dst_path, f"saros_{sid}_img.nii.gz"))
        sitk.WriteImage(res["highres"], os.path.join(dst_path, f"saros_{sid}_ts_heartchambershighres.nii.gz"))
        sitk.WriteImage(res["total"], os.path.join(dst_path, f"saros_{sid}_ts_total.nii.gz"))
        sitk.WriteImage(res["coronary"], os.path.join(dst_path, f"saros_{sid}_ts_coronaryarteries.nii.gz"))
        sitk.WriteImage(res["trunk"], os.path.join(dst_path, f"saros_{sid}_ts_trunkcavities.nii.gz"))
        sitk.WriteImage(res["label"], os.path.join(dst_path, f"saros_{sid}_label.nii.gz"))


# ----------------------------
# REORIENT
# ----------------------------
def _reorient_file(file_path, direction):
    if not file_path.endswith(".nii.gz"):
        return None

    try:
        img = sitk.ReadImage(file_path)
        img = utils.reorient_sitk(img, direction)
        sitk.WriteImage(img, file_path)
        return True
    except Exception:
        return False


def reorient_folder(path, direction="LPS", workers=None):
    if workers is None:
        workers = max(os.cpu_count() - 1, 1)

    files = [os.path.join(path, f) for f in os.listdir(path)]

    worker = partial(_reorient_file, direction=direction)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(
            tqdm(
                ex.map(worker, files),
                total=len(files),
                desc="Reorienting files",
            )
        )


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--dst", default=DEFAULT_DST)
    parser.add_argument("--mode", choices=["organize", "reorient", "both"], default="both")
    parser.add_argument("--direction", default="LPS")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--margin-mm", type=int, default=20)

    args = parser.parse_args()

    if args.mode in ("organize", "both"):
        collect_saros_cases_parallel(
            args.src,
            args.dst,
            margin_mm=args.margin_mm,
            workers=args.workers,
        )

    if args.mode in ("reorient", "both"):
        reorient_folder(
            args.dst,
            direction=args.direction,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()