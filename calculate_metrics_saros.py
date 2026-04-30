import torch
import sys
import os
import pandas as pd
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from my_functions import *
import concurrent.futures
from processing_tools import read_nifti_itk_to_vtk

# ═══════════════════════════════════════════════════════════════════════
#                        WORKER FUNCTION
# ═══════════════════════════════════════════════════════════════════════
def process_series(series, pytorch3d_folder, data_folder, device):
    """Processes a single series and returns the metrics row."""
    try:
        series_folder = os.path.join(pytorch3d_folder, series)
        
        ground_truth_path = os.path.join(data_folder, series + "_label.nii.gz")
        img_path = os.path.join(data_folder, series + "_img.nii.gz")
        ts_heartchambershighres_path = os.path.join(data_folder, series + "_ts_heartchambershighres.nii.gz")
        ts_total_path = os.path.join(data_folder, series + "_ts_total.nii.gz")
        TS_coronaryarteries_path = os.path.join(data_folder, series + "_ts_coronaryarteries.nii.gz")
        mesh_smoothed_path_obj = os.path.join(series_folder, series + "_smoothedsurface.obj")
        mesh_refined_path_obj = os.path.join(series_folder, series + "_refined_mesh_taubin.obj")

        # ── Load smoothed mesh ───────────────────────────────────────────
        verts, faces, aux = load_obj(mesh_smoothed_path_obj)
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)

        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        src_mesh = Meshes(verts=[verts], faces=[faces_idx])

        # ── Load masks ───────────────────────────────────────────────────
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

        msk_coronaryarteries_sitk = sitk.ReadImage(TS_coronaryarteries_path)
        msk_coronaryarteries_sitk.SetOrigin(tuple((origin - center.cpu().numpy()) / scale.cpu().numpy()))
        msk_coronaryarteries_sitk.SetSpacing(tuple(spacing / scale.cpu().numpy()))
        msk_coronaryarteries = sitk.GetArrayFromImage(msk_coronaryarteries_sitk)

        # ── Inside / Outside logic ───────────────────────────────────────
        msk_inside = (msk_coronaryarteries > 0).copy()
        msk_inside |= (msk_highres > 0)
        msk_inside |= mask_segment_61
        msk_outside = (msk_total > 0).copy()
        msk_scaled_sitk = msk_highres_sitk

        # ── Metrics BEFORE ───────────────────────────────────────────────
        z_cutoff = get_z_cutoff_for_segment(ts_heartchambershighres_path, segment_id=2)
        gt_sitk  = sitk.ReadImage(ground_truth_path)
        # Only look at segment 7 (pericardium)
        gt_array = sitk.GetArrayFromImage(gt_sitk)
        gt_array[gt_array != 7] = 0
        gt_array[gt_array == 7] = 1
        gt_sitk = sitk.GetImageFromArray(gt_array)

        img_sitk = sitk.ReadImage(img_path)
        gt_sitk.CopyInformation(img_sitk)  # Ensure same spacing and origin as masks

        metrics_before_raw = compute_all_metrics_saros(mesh_smoothed_path_obj, gt_sitk, img_sitk, z_cutoff=z_cutoff)
        metrics_before = {f"before_{k}": v for k, v in metrics_before_raw.items()}

        count_inside_before, _  = count_vertices_in_mask(src_mesh, msk_inside,  msk_scaled_sitk)
        count_outside_before, _ = count_vertices_in_mask(src_mesh, msk_outside, msk_scaled_sitk)
        volume_overlap_inside_before, volume_overlap_outside_before = count_area_overlaps(mesh_smoothed_path_obj, msk_inside, msk_outside, gt_sitk)  # sanity check - should be similar to below


        # ── Load refined mesh ────────────────────────────────────────────
        verts, faces, aux = load_obj(mesh_refined_path_obj)
        faces_idx = faces.verts_idx.to(device)
        verts = verts - center
        verts = verts / scale

        refined_mesh = Meshes(verts=[verts], faces=[faces_idx])

        count_inside_after, _  = count_vertices_in_mask(refined_mesh, msk_inside,  msk_scaled_sitk)
        count_outside_after, _ = count_vertices_in_mask(refined_mesh, msk_outside, msk_scaled_sitk)
        volume_overlap_inside_after, volume_overlap_outside_after = count_area_overlaps(mesh_refined_path_obj, msk_inside, msk_outside, gt_sitk)  # sanity check - should be similar to below

        metrics_after_raw = compute_all_metrics_saros(mesh_refined_path_obj, gt_sitk, img_sitk, z_cutoff=z_cutoff)
        metrics_after = {f"after_{k}": v for k, v in metrics_after_raw.items()}

        # ── Compile row ──────────────────────────────────────────────────
        row = {"series": series}

        row.update(metrics_before)
        row.update(metrics_after)

        # ── Delta calculation (FIXED) ────────────────────────────────────
        metric_names = ['Dice', 'EAT_Dice', 'ASD', 'ASSD', 'HD', 'HD95', 'NSD']

        for m in metric_names:
            before_val = metrics_before.get(f"before_{m}", float('nan'))
            after_val  = metrics_after.get(f"after_{m}", float('nan'))
            row[f"delta_{m}"] = after_val - before_val

        # ── Vertex counts ────────────────────────────────────────────────
        row["before_inside_verts"]  = count_inside_before
        row["before_outside_verts"] = count_outside_before
        row["after_inside_verts"]   = count_inside_after
        row["after_outside_verts"]  = count_outside_after

        # ── Volume overlaps ───────────────────────────────────────────────
        row["before_volume_overlap_inside"] = volume_overlap_inside_before
        row["before_volume_overlap_outside"] = volume_overlap_outside_before
        row["after_volume_overlap_inside"] = volume_overlap_inside_after
        row["after_volume_overlap_outside"] = volume_overlap_outside_after
        row["delta_volume_overlap_inside"] = volume_overlap_inside_after - volume_overlap_inside_before
        row["delta_volume_overlap_outside"] = volume_overlap_outside_after - volume_overlap_outside_before

        return row

    except Exception as e:
        print(f"Error processing {series}: {e}")
        return {"series": series, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#                          MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    device = torch.device("cpu")

    metrics_folder = 'best_grid_search_result_EXCLUDEGRID_EAT0'  # subfolder for metrics results
    pytorch3d_folder = "/data/awias/periseg/saros/TS_pericardium/pytorch3d"
    data_folder = "/data/awias/periseg/saros/NIFTI_collected"

    metrics_folder = os.path.join(pytorch3d_folder, 'metrics', metrics_folder)
    csv_path = os.path.join(metrics_folder, "metrics_summary_taubin.csv")

    os.makedirs(metrics_folder, exist_ok=True)

    all_series = [x for x in os.listdir(pytorch3d_folder) if x.startswith("saros")]
    all_series = sorted(all_series, key=lambda x: int(x.split('_')[1]))
    all_series = np.unique(all_series)

    # all_series = all_series[:-1]

    # TEMP
    # all_series = all_series[:2]

    results = []

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