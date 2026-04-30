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


DEFAULT_METRICS_FOLDER = "best_grid_search_result_EXCLUDEGRID_EAT0"
DEFAULT_PYTORCH3D_FOLDER = "/data/awias/periseg/saros/TS_pericardium/pytorch3d"
DEFAULT_DATA_FOLDER = "/data/awias/periseg/saros/NIFTI_collected"
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
        gt_array = sitk.GetArrayFromImage(gt_sitk)
        gt_array[gt_array != 7] = 0
        gt_array[gt_array == 7] = 1
        gt_sitk = sitk.GetImageFromArray(gt_array)

        img_sitk = sitk.ReadImage(img_path)
        gt_sitk.CopyInformation(img_sitk)

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

    all_series = [x for x in os.listdir(pytorch3d_folder) if x.startswith("saros")]
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
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["savefig.dpi"] = 150
    mpl.rcParams["figure.dpi"] = 100
    mpl.rcParams["font.size"] = 10

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

    higher_is_better = {
        "Dice": True,
        "EAT_Dice": True,
        "ASD": False,
        "ASSD": False,
        "HD": False,
        "HD95": False,
        "NSD": True,
        "volume_overlap_inside": False,
        "volume_overlap_outside": False,
    }

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} series from {csv_path}\n")

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

    short_labels = ["_".join(s.split("_")[-2:]) if len(s) > 25 else s for s in df["series"]]
    x = np.arange(len(df))

    sep = "=" * 90
    print(sep)
    print(f"{'Series':<35} {'Metric':<10} {'Before':>9} {'After':>9} {'Delta':>9} {'OK?':>5}")
    print("-" * 90)

    for _, row in df.iterrows():
        for metric in metrics:
            b_col, a_col = f"before_{metric}", f"after_{metric}"
            if b_col not in df.columns:
                continue
            b, a = row[b_col], row[a_col]
            d = a - b
            if b != 0:
                performance_gain = (a - b) / abs(b) * 100 if higher_is_better[metric] else (b - a) / abs(b) * 100
            else:
                performance_gain = float("nan")

            improved = (d > 0) if higher_is_better[metric] else (d < 0)
            tag = " ✓" if improved else " ✗"
            print(
                f"{row['series']:<35} {metric:<10} {b:>9.2f} {a:>9.2f} {d:>+9.2f} {performance_gain:>+7.2f}% {tag:>5}"
            )

    if len(df) > 1:
        print("-" * 120)
        for metric in metrics:
            b_col, a_col = f"before_{metric}", f"after_{metric}"
            if b_col not in df.columns:
                continue
            b_mean = df[b_col].mean()
            a_mean = df[a_col].mean()
            b_std = df[b_col].std()
            a_std = df[a_col].std()
            d = a_mean - b_mean

            if b_mean != 0:
                performance_gain = (a_mean - b_mean) / abs(b_mean) * 100 if higher_is_better[metric] else (b_mean - a_mean) / abs(b_mean) * 100
            else:
                performance_gain = float("nan")

            improved = (d > 0) if higher_is_better[metric] else (d < 0)
            tag = " ✓" if improved else " ✗"
            print(
                f"{'MEAN':<20} {metric:<10} {b_mean:>9.2f} ± {b_std:<8.2f} {a_mean:>9.2f} ± {a_std:<8.2f} {d:>+9.2f}  {performance_gain:>+7.2f}% {tag:>5}"
            )

    print(sep)

    name_map = {
        "Dice": r"\text{DSC}\, (\%)",
        "EAT_Dice": r"\text{DSC}_{\text{EAT}}\, (\%)",
        "NSD": r"\text{NSD}\, (\%)",
        "HD": r"\text{HD}\, (\text{mm})",
        "HD95": r"\text{HD95}\, (\text{mm})",
        "ASSD": r"\text{ASSD}\, (\text{mm})",
        "volume_overlap_inside": r"\text{Internal Violation}\, (\text{cm}^3)",
        "volume_overlap_outside": r"\text{External Violation}\, (\text{cm}^3)",
    }

    metrics_to_include = [
        "Dice",
        "EAT_Dice",
        "NSD",
        "HD95",
        "ASSD",
        "volume_overlap_inside",
        "volume_overlap_outside",
    ]

    median_metrics = ["volume_overlap_inside", "volume_overlap_outside"]

    if len(df) > 1:
        for metric in metrics_to_include:
            b_col, a_col = f"before_{metric}", f"after_{metric}"
            if b_col not in df.columns:
                continue

            higher = higher_is_better[metric]
            arrow = r"$\uparrow$" if higher else r"$\downarrow$"
            name = name_map.get(metric, rf"\text{{{metric}}}")

            if metric in median_metrics:
                b_vals = df[b_col].values
                a_vals = df[a_col].values

                b_median = np.median(b_vals)
                a_median = np.median(a_vals)

                b_q25 = np.percentile(b_vals, 25)
                b_q75 = np.percentile(b_vals, 75)

                a_q25 = np.percentile(a_vals, 25)
                a_q75 = np.percentile(a_vals, 75)

                if b_median != 0:
                    performance_gain = (
                        (a_median - b_median) / abs(b_median) * 100
                        if higher
                        else (b_median - a_median) / abs(b_median) * 100
                    )
                else:
                    performance_gain = float("nan")

                if higher:
                    before_str = (
                        rf"\mathbf{{{b_median:.2f}}} \ [{b_q25:.1f}, {b_q75:.1f}]"
                        if b_median >= a_median
                        else rf"{b_median:.2f} \ [{b_q25:.1f}, {b_q75:.1f}]"
                    )
                    after_str = (
                        rf"\mathbf{{{a_median:.2f}}} \ [{a_q25:.1f}, {a_q75:.1f}]"
                        if a_median > b_median
                        else rf"{a_median:.2f} \ [{a_q25:.1f}, {a_q75:.1f}]"
                    )
                else:
                    before_str = (
                        rf"\mathbf{{{b_median:.2f}}} \ [{b_q25:.1f}, {b_q75:.1f}]"
                        if b_median <= a_median
                        else rf"{b_median:.2f} \ [{b_q25:.1f}, {b_q75:.1f}]"
                    )
                    after_str = (
                        rf"\mathbf{{{a_median:.2f}}} \ [{a_q25:.1f}, {a_q75:.1f}]"
                        if a_median < b_median
                        else rf"{a_median:.2f} \ [{a_q25:.1f}, {a_q75:.1f}]"
                    )

            else:
                b_mean = df[b_col].mean()
                a_mean = df[a_col].mean()
                b_std = df[b_col].std()
                a_std = df[a_col].std()

                if b_mean != 0:
                    performance_gain = (
                        (a_mean - b_mean) / abs(b_mean) * 100
                        if higher
                        else (b_mean - a_mean) / abs(b_mean) * 100
                    )
                else:
                    performance_gain = float("nan")

                if higher:
                    before_str = (
                        rf"\mathbf{{{b_mean:.2f}}} \pm {b_std:.2f}"
                        if b_mean >= a_mean
                        else rf"{b_mean:.2f} \pm {b_std:.2f}"
                    )
                    after_str = (
                        rf"\mathbf{{{a_mean:.2f}}} \pm {a_std:.2f}"
                        if a_mean > b_mean
                        else rf"{a_mean:.2f} \pm {a_std:.2f}"
                    )
                else:
                    before_str = (
                        rf"\mathbf{{{b_mean:.2f}}} \pm {b_std:.2f}"
                        if b_mean <= a_mean
                        else rf"{b_mean:.2f} \pm {b_std:.2f}"
                    )
                    after_str = (
                        rf"\mathbf{{{a_mean:.2f}}} \pm {a_std:.2f}"
                        if a_mean < b_mean
                        else rf"{a_mean:.2f} \pm {a_std:.2f}"
                    )

            sign = "+" if performance_gain >= 0 else ""

            row = (
                f"& {arrow} ${name}$ "
                f"& ${before_str}$ "
                f"& ${after_str}$ "
                f"& ${sign}{performance_gain:.2f}$\\,\\% \\\\"
            )

            print(row)

    anatomical_metrics = {
        "volume_overlap_outside": "External Violation",
        "volume_overlap_inside": "Internal Violation",
    }

    print("\n% --- ANATOMICAL VIOLATION TABLE ---")
    if len(df) > 1:
        for metric, display_name in anatomical_metrics.items():
            b_col, a_col = f"before_{metric}", f"after_{metric}"
            if b_col not in df.columns:
                continue

            reductions = []
            improved_count = 0

            for _, row in df.iterrows():
                b_val, a_val = row[b_col], row[a_col]
                if b_val > 0:
                    reduction = (b_val - a_val) / b_val * 100
                    reductions.append(reduction)
                    if a_val < b_val:
                        improved_count += 1
                elif b_val == 0 and a_val == 0:
                    improved_count += 1

            if reductions:
                median_reduction = np.median(reductions)
                q25 = np.percentile(reductions, 25)
                q75 = np.percentile(reductions, 75)
            else:
                median_reduction = 0
                q25 = 0
                q75 = 0

            row_tex = f"& {display_name} & {median_reduction:.1f} [{q25:.1f}, {q75:.1f}] \\\\"
            print(row_tex)

    print("\n% --- ANATOMICAL VIOLATION TABLE NEW ---")
    if len(df) > 1:
        for metric, display_name in anatomical_metrics.items():
            b_col, a_col = f"before_{metric}", f"after_{metric}"
            if b_col not in df.columns:
                continue

            before_values = df[b_col].values
            median_before = np.median(before_values)
            q25_before = np.percentile(before_values, 25)
            q75_before = np.percentile(before_values, 75)

            after_values = df[a_col].values
            median_after = np.median(after_values)
            q25_after = np.percentile(after_values, 25)
            q75_after = np.percentile(after_values, 75)

            performance_gain = (
                (median_after - median_before) / abs(median_before) * 100
                if median_before != 0
                else float("nan")
            )
            tag = " ✓" if (median_after < median_before) else " ✗"

            print(
                f"{'median':<20} {metric:<10} {median_before:>9.2f} [{q25_before:.1f}, {q75_before:.1f}] "
                f"{median_after:>9.2f} [{q25_after:.1f}, {q75_after:.1f}] {performance_gain:>+7.2f}% {tag:>5}"
            )

    return df


def main():
    parser = argparse.ArgumentParser(description="Calculate or summarize SAROS refinement metrics.")
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
