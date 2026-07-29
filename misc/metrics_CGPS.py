"""
Reference-only metrics script for the private CGPS (DTU-Pericardium) dataset.

This is the CGPS counterpart of ``metrics_saros.py`` (SAROS). It reuses the
same helper functions from ``saros_utils`` and mirrors its structure exactly,
but points at the internal CGPS data layout. Unlike the SAROS ground truth,
the CGPS label maps are already binary pericardium masks, so no segment
filtering is applied before computing metrics. CGPS is a private dataset, so
this script is NOT runnable as-is — see ``metrics_saros.py`` in the repo root
for the public, runnable SAROS pipeline. Kept here purely for reference /
reproducibility, alongside ``Run_refinement_CGPS.py``.
"""

import argparse
import os
import concurrent.futures

import numpy as np
import pandas as pd
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from tqdm import tqdm
import SimpleITK as sitk

import saros_utils as utils


DEFAULT_METRICS_FOLDER = "fixed_scans"
DEFAULT_PYTORCH3D_FOLDER = "/data/awias/periseg/DTU-Pericardium-04-02-2026/TS_pericardium/pytorch3d_new"
DEFAULT_DATA_FOLDER = "/data/awias/periseg/DTU-Pericardium-04-02-2026/NIFTI_collected_new"
DEFAULT_CSV_NAME = "metrics_summary_taubin.csv"


def process_series(series, pytorch3d_folder, data_folder, device):
    try:
        series_folder = os.path.join(pytorch3d_folder, series)

        ground_truth_path = os.path.join(data_folder, series + "_label.nii.gz")
        img_path = os.path.join(data_folder, series + "_img.nii.gz")
        ts_heartchambershighres_path = os.path.join(data_folder, series + "_ts_heartchambershighres.nii.gz")
        ts_total_path = os.path.join(data_folder, series + "_ts_total.nii.gz")
        ts_coronaryarteries_path = os.path.join(data_folder, series + "_ts_coronaryarteries.nii.gz")
        mesh_smoothed_path_obj = os.path.join(series_folder, series + "_smoothedsurface.obj")
        mesh_refined_path_obj = os.path.join(series_folder, series + "_refined_mesh_taubin.obj")

        verts, faces, _ = load_obj(mesh_smoothed_path_obj)
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)

        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        src_mesh = Meshes(verts=[verts], faces=[faces_idx])

        msk_highres_sitk = sitk.ReadImage(ts_heartchambershighres_path)
        spacing = msk_highres_sitk.GetSpacing()
        origin = msk_highres_sitk.GetOrigin()

        msk_highres_sitk.SetOrigin(tuple((origin - center.cpu().numpy()) / scale.cpu().numpy()))
        msk_highres_sitk.SetSpacing(tuple(spacing / scale.cpu().numpy()))
        msk_highres = sitk.GetArrayFromImage(msk_highres_sitk)

        for idx in [6, 7]:
            msk_highres[msk_highres == idx] = 0

        msk_total_sitk = sitk.ReadImage(ts_total_path)
        msk_total_sitk.SetOrigin(tuple((origin - center.cpu().numpy()) / scale.cpu().numpy()))
        msk_total_sitk.SetSpacing(tuple(spacing / scale.cpu().numpy()))
        msk_total = sitk.GetArrayFromImage(msk_total_sitk)
        mask_segment_61 = (msk_total == 61)
        for idx in [0, 51, 52, 53, 61, 62, 63]:
            msk_total[msk_total == idx] = 0

        msk_coronaryarteries_sitk = sitk.ReadImage(ts_coronaryarteries_path)
        msk_coronaryarteries_sitk.SetOrigin(tuple((origin - center.cpu().numpy()) / scale.cpu().numpy()))
        msk_coronaryarteries_sitk.SetSpacing(tuple(spacing / scale.cpu().numpy()))
        msk_coronaryarteries = sitk.GetArrayFromImage(msk_coronaryarteries_sitk)

        msk_inside = (msk_coronaryarteries > 0).copy()
        msk_inside |= (msk_highres > 0)
        msk_inside |= mask_segment_61
        msk_outside = (msk_total > 0).copy()
        msk_scaled_sitk = msk_highres_sitk

        z_cutoff = utils.get_z_cutoff_for_segment(ts_heartchambershighres_path, segment_id=2)
        gt_sitk = sitk.ReadImage(ground_truth_path)
        img_sitk = sitk.ReadImage(img_path)

        metrics_before_raw = utils.compute_all_metrics_saros(
            mesh_smoothed_path_obj, gt_sitk, img_sitk, z_cutoff=z_cutoff
        )
        metrics_before = {f"before_{k}": v for k, v in metrics_before_raw.items()}

        count_inside_before, _ = utils.count_vertices_in_mask(src_mesh, msk_inside, msk_scaled_sitk)
        count_outside_before, _ = utils.count_vertices_in_mask(src_mesh, msk_outside, msk_scaled_sitk)
        volume_overlap_inside_before, volume_overlap_outside_before = utils.count_area_overlaps(
            mesh_smoothed_path_obj, msk_inside, msk_outside, gt_sitk
        )

        verts, faces, _ = load_obj(mesh_refined_path_obj)
        faces_idx = faces.verts_idx.to(device)
        verts = verts - center
        verts = verts / scale

        refined_mesh = Meshes(verts=[verts], faces=[faces_idx])

        count_inside_after, _ = utils.count_vertices_in_mask(refined_mesh, msk_inside, msk_scaled_sitk)
        count_outside_after, _ = utils.count_vertices_in_mask(refined_mesh, msk_outside, msk_scaled_sitk)
        volume_overlap_inside_after, volume_overlap_outside_after = utils.count_area_overlaps(
            mesh_refined_path_obj, msk_inside, msk_outside, gt_sitk
        )

        metrics_after_raw = utils.compute_all_metrics_saros(
            mesh_refined_path_obj, gt_sitk, img_sitk, z_cutoff=z_cutoff
        )
        metrics_after = {f"after_{k}": v for k, v in metrics_after_raw.items()}

        row = {"series": series}
        row.update(metrics_before)
        row.update(metrics_after)

        metric_names = ["Dice", "EAT_Dice", "ASD", "ASSD", "HD", "HD95", "NSD"]

        for metric in metric_names:
            before_val = metrics_before.get(f"before_{metric}", float("nan"))
            after_val = metrics_after.get(f"after_{metric}", float("nan"))
            row[f"delta_{metric}"] = after_val - before_val

        row["before_inside_verts"] = count_inside_before
        row["before_outside_verts"] = count_outside_before
        row["after_inside_verts"] = count_inside_after
        row["after_outside_verts"] = count_outside_after

        row["before_volume_overlap_inside"] = volume_overlap_inside_before
        row["before_volume_overlap_outside"] = volume_overlap_outside_before
        row["after_volume_overlap_inside"] = volume_overlap_inside_after
        row["after_volume_overlap_outside"] = volume_overlap_outside_after
        row["delta_volume_overlap_inside"] = volume_overlap_inside_after - volume_overlap_inside_before
        row["delta_volume_overlap_outside"] = volume_overlap_outside_after - volume_overlap_outside_before

        return row

    except Exception as exc:
        print(f"Error processing {series}: {exc}")
        return {"series": series, "error": str(exc)}


def calculate_metrics(
    metrics_folder=DEFAULT_METRICS_FOLDER,
    pytorch3d_folder=DEFAULT_PYTORCH3D_FOLDER,
    data_folder=DEFAULT_DATA_FOLDER,
    csv_name=DEFAULT_CSV_NAME,
    max_workers=None,
):
    device = torch.device("cpu")

    metrics_folder = os.path.join(pytorch3d_folder, "metrics", metrics_folder)
    csv_path = os.path.join(metrics_folder, csv_name)

    os.makedirs(metrics_folder, exist_ok=True)

    all_series = [x for x in os.listdir(pytorch3d_folder) if x.startswith("Pericardium")]
    all_series = sorted(all_series, key=lambda x: int(x.split("_")[1]))
    all_series = np.unique(all_series)

    results = []

    if max_workers is None:
        max_workers = min(os.cpu_count(), 16)

    print(f"Starting parallel processing with {max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_series = {
            executor.submit(process_series, series, pytorch3d_folder, data_folder, device): series
            for series in all_series
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_series), total=len(all_series)):
            series = future_to_series[future]
            try:
                row_result = future.result()
                results.append(row_result)

                df = pd.DataFrame(results)
                df.to_csv(csv_path, index=False)
            except Exception as exc:
                print(f"{series} generated an exception: {exc}")

    print("All processing complete!")
    return csv_path


def summarize_metrics(csv_path):
    import matplotlib.pyplot as plt

    metrics = [
        "Dice",
        "EAT_Dice",
        "NSD",
        "HD",
        "HD95",
        "ASD",
        "ASSD",
        "volume_overlap_inside",
        "volume_overlap_outside",
    ]

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} series from {csv_path}\n")

    # Convert to more readable units
    for metric in ["Dice", "EAT_Dice", "NSD"]:
        for prefix in ["before_", "after_"]:
            col = f"{prefix}{metric}"
            if col in df.columns:
                df[col] *= 100

    for prefix in ["before_", "after_"]:
        for suffix in ["volume_overlap_inside", "volume_overlap_outside"]:
            col = f"{prefix}{suffix}"
            if col in df.columns:
                df[col] /= 1000

    for metric in metrics:
        b_col, a_col = f"before_{metric}", f"after_{metric}"
        if b_col not in df.columns:
            continue
        b_mean, b_std = df[b_col].mean(), df[b_col].std()
        a_mean, a_std = df[a_col].mean(), df[a_col].std()
        print(
            f"{metric:<25} before = {b_mean:7.2f} ± {b_std:5.2f}   "
            f"after = {a_mean:7.2f} ± {a_std:5.2f}   "
            f"delta = {a_mean - b_mean:+7.2f}"
        )

    plot_metrics = [m for m in metrics if f"before_{m}" in df.columns]
    ncols = 3
    nrows = -(-len(plot_metrics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, metric in zip(axes, plot_metrics):
        b_col, a_col = f"before_{metric}", f"after_{metric}"
        ax.boxplot([df[b_col].dropna(), df[a_col].dropna()], tick_labels=["before", "after"])
        ax.set_title(metric)

    for ax in axes[len(plot_metrics):]:
        ax.axis("off")

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(csv_path) or ".", "metrics_summary_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Calculate or summarize CGPS refinement metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calc_parser = subparsers.add_parser("calculate", help="Compute metrics for all series")
    calc_parser.add_argument("--metrics-folder", default=DEFAULT_METRICS_FOLDER)
    calc_parser.add_argument("--pytorch3d-folder", default=DEFAULT_PYTORCH3D_FOLDER)
    calc_parser.add_argument("--data-folder", default=DEFAULT_DATA_FOLDER)
    calc_parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME)
    calc_parser.add_argument("--max-workers", type=int, default=None)

    summ_parser = subparsers.add_parser("summarize", help="Summarize a metrics CSV")
    summ_parser.add_argument("--csv-path", required=True)

    args = parser.parse_args()

    if args.command == "calculate":
        calculate_metrics(
            metrics_folder=args.metrics_folder,
            pytorch3d_folder=args.pytorch3d_folder,
            data_folder=args.data_folder,
            csv_name=args.csv_name,
            max_workers=args.max_workers,
        )
    elif args.command == "summarize":
        summarize_metrics(args.csv_path)


if __name__ == "__main__":
    main()
