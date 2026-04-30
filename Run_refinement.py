from time import time

import torch
import sys
import os
import torch
import pandas as pd
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import saros_utils as utils
import SimpleITK as sitk
from saros_utils import *
import vtk
from scipy.interpolate import RegularGridInterpolator
import torch
import torch.nn.functional as F
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
from tqdm import tqdm
import traceback
import edt



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


output_folder_root = f"/data/awias/periseg/saros/TS_pericardium/pytorch3d_testwith10"
data_folder = f"/data/awias/periseg/saros/NIFTI_collected"

csv_path = "/data/awias/periseg/saros/series.csv"

# Vector fields folder
vf_folder = f"/data/awias/periseg/saros/TS_pericardium/vector_fields"

# ═══════════════════════════════════════════════════════════════════════
#                        MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
df = pd.read_csv(csv_path)
all_series = df.loc[df['grid_search'] != 1, 'all_series_name'].tolist()

all_series = sorted(all_series, key=lambda x: int(x.split('_')[1]))
all_series = np.unique(all_series)

os.makedirs(vf_folder, exist_ok=True)

# all_series = ['Pericardium-04-02-2026_0016_SERIES0009']
# all_series = ['Pericardium-04-02-2026_0001_SERIES0018']

# all_series = all_series[:20]  # TEMP: limit to first 20 for testing

# all_series = ['saros_311']
# all_series = ['saros_221']
# all_series = ['saros_154']
# all_series = ['saros_310']
# all_series = ['saros_304']

for series in tqdm(all_series):

    output_folder = os.path.join(output_folder_root, series)
    output_path = os.path.join(output_folder, series + "_refined_mesh.obj")
    
    os.makedirs(output_folder, exist_ok=True)

    ground_truth_path = os.path.join(data_folder, series + "_label.nii.gz")
    img_path = os.path.join(data_folder, series + "_img.nii.gz")
    ts_heartchambershighres_path = os.path.join(data_folder, series + "_ts_heartchambershighres.nii.gz")
    ts_total_path = os.path.join(data_folder, series + "_ts_total.nii.gz")
    TS_trunkcavities_path = os.path.join(data_folder, series + "_ts_trunkcavities.nii.gz")
    TS_coronaryarteries_path = os.path.join(data_folder, series + "_ts_coronaryarteries.nii.gz")
    mesh_path = os.path.join(output_folder, series + "_rawsurface.vtk")
    mesh_smoothed_path = os.path.join(output_folder, series + "_smoothedsurface.vtk")
    mesh_smoothed_path_obj = os.path.join(output_folder, series + "_smoothedsurface.obj")

    # if os.path.exists(output_path):
    #     print(f"Refined mesh already exists for {series}, skipping...")
    #     continue

    # if not os.path.exists(mesh_smoothed_path_obj):
    
    utils.convert_label_map_to_surface_file(TS_trunkcavities_path, mesh_path, segment_id=3)
    decimate_and_smooth_ALOT(mesh_path, mesh_smoothed_path)
    utils.convert_vtk_to_obj(mesh_smoothed_path, mesh_smoothed_path_obj)


    # Load the smoothed mesh
    verts, faces, aux = load_obj(mesh_smoothed_path_obj)
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
    direction = msk_highres_sitk.GetDirection()
    msk_highres_sitk.SetOrigin(tuple((origin - center.cpu().numpy()) / scale.cpu().numpy()))
    msk_highres_sitk.SetSpacing(tuple(spacing / scale.cpu().numpy()))
    msk_highres = sitk.GetArrayFromImage(msk_highres_sitk)
    structures_not_to_include = [6,7]  # example structure IDs to exclude
    for idx in structures_not_to_include:
        msk_highres[msk_highres == idx] = 0
    msk_highres_bin = np.isin(msk_highres, [1,2,3,4,5])
    COM_zyx = center_of_mass(msk_highres_bin)
    COM_xyz_index = (COM_zyx[2], COM_zyx[1], COM_zyx[0])
    COM_phys = np.array(msk_highres_sitk.TransformContinuousIndexToPhysicalPoint(COM_xyz_index))

    img_sitk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_sitk)

    msk_total_sitk = sitk.ReadImage(ts_total_path)
    msk_total_sitk.SetOrigin(tuple((origin - center.cpu().numpy()) / scale.cpu().numpy()))
    msk_total_sitk.SetSpacing(tuple(spacing / scale.cpu().numpy()))
    msk_total = sitk.GetArrayFromImage(msk_total_sitk)
    mask_segment_61 = (msk_total == 61)
    structures_not_to_include = [0, 51, 52, 53, 61, 62, 63]  # example structure IDs to exclude
    for idx in structures_not_to_include:
        msk_total[msk_total == idx] = 0

    msk_coronaryarteries_sitk = sitk.ReadImage(TS_coronaryarteries_path)
    msk_coronaryarteries_sitk.SetOrigin(tuple((origin - center.cpu().numpy()) / scale.cpu().numpy()))
    msk_coronaryarteries_sitk.SetSpacing(tuple(spacing / scale.cpu().numpy()))
    msk_coronaryarteries = sitk.GetArrayFromImage(msk_coronaryarteries_sitk)

    # Inside
    msk_inside = (msk_coronaryarteries > 0).copy()
    hr_mask = msk_highres > 0
    msk_inside |= hr_mask
    msk_inside |= mask_segment_61


    # Outside
    msk_outside = (msk_total > 0).copy()

    msk_nonscaled_sitk = sitk.ReadImage(ts_total_path)
    msk_scaled_sitk = msk_highres_sitk


    # Vector field file paths
    # ── Setup Single Path ────────────────────────────────────────────────
    # Storing everything in one (6, Z, Y, X) float32 file
    vf_combined_path = os.path.join(vf_folder, f"{series}_VF_all.npy")

    if os.path.exists(vf_combined_path):
        print(f"🚀 Loading data for {series}")
        # mmap_mode='r' makes this nearly instant (0.6s total read based on your test)
        # We use .copy() to bring it into RAM fully since you'll be converting to tensors

        start_time = time()
        combined_data = np.load(vf_combined_path).astype(np.float32)
        end_time = time()
        print(f"Loaded combined vector field in {end_time - start_time:.2f} seconds")
        
        # Split the views (indices 0-2 are outside, 3-5 are inside)
        Vx_outside, Vy_outside, Vz_outside = combined_data[0], combined_data[1], combined_data[2]
        Vx_inside,  Vy_inside,  Vz_inside  = combined_data[3], combined_data[4], combined_data[5]

    else:
        print(f"⚙️ Creating vector fields for {series}...")
        #time the vector field creation to see how long it takes
        
        start_1 = time()
        Vx_outside, Vy_outside, Vz_outside = create_internal_external_vector_fields_fast(msk_outside, COM_zyx, spacing, region="external")
        end_1 = time()
        print(f"Outside vector field created in {end_1 - start_1:.2f} seconds")

        start_2 = time()
        Vx_inside, Vy_inside, Vz_inside = create_internal_external_vector_fields_fast(msk_inside, COM_zyx, spacing, region="internal")
        end_2 = time()
        print(f"Inside vector field created in {end_2 - start_2:.2f} seconds")   

        # Consolidate into one 4D array: Shape (6, Z, Y, X)
        # This keeps your disk clean and makes loading 21x faster than HDD
        combined_data = np.stack([
            Vx_outside, Vy_outside, Vz_outside,
            Vx_inside,  Vy_inside,  Vz_inside
        ], axis=0).astype(np.float32)
        
        print(f"💾 Saving consolidated field to {vf_combined_path}")
        np.save(vf_combined_path, combined_data)

    # Combine
    # Start from the lowest-priority field (total)
    Vx_combined = Vx_outside.copy()
    Vy_combined = Vy_outside.copy()
    Vz_combined = Vz_outside.copy()
    msk_combined = (msk_outside > 0).copy()

    # Overwrite with highres where its mask is active (highest priority)
    inside_mask = msk_inside > 0
    Vx_combined[inside_mask] = Vx_inside[inside_mask]
    Vy_combined[inside_mask] = Vy_inside[inside_mask]
    Vz_combined[inside_mask] = Vz_inside[inside_mask]
    msk_combined[inside_mask] = 1

    vector_field_inside = np.stack([Vx_inside, Vy_inside, Vz_inside], axis=0)
    vector_field_inside_tensor = torch.from_numpy(vector_field_inside).float().to(device)

    vector_field_outside = np.stack([Vx_outside, Vy_outside, Vz_outside], axis=0)
    vector_field_outside_tensor = torch.from_numpy(vector_field_outside).float().to(device)




    # ── Define self parameters ────────────────────────────────────────────────────────────────
    phase1_iters              = 1000
    phase2_length             = 500
    phase3_length             = 500
    Niter                     = phase1_iters + phase2_length + phase3_length
    w_edge_p1                 = 0.001 #1.0
    w_laplacian_p1            = 0.015
    w_normal_p1               = 0.001
    w_vf_inside_p1            = 1 #0.5 * 16
    w_vf_outside_p1           = 0.35 #0.5 * 1.17
    w_edge_p2                 = w_edge_p1 #2.0
    w_laplacian_p2            = w_laplacian_p1*2
    w_normal_p2               = 0.1
    lr_init                   = 0.00007
    lr_min                    = 1e-6
    weight_decay              = 0.002
    max_grad_norm             = 1



    # LOOP
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([deform_verts], lr=lr_init, weight_decay=weight_decay)

    # CosineAnnealingLR: LR decays smoothly to lr_min over all Niter steps
    # This pairs well with the phase schedule — LR is naturally low in Phase 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Niter, eta_min=lr_min
    )

    # ── Loss history lists ────────────────────────────────────────────────────────
    edge_losses, normal_losses, laplacian_losses = [], [], []
    vectorfield_losses_inside, vectorfield_losses_outside = [], []
    edge_losses_magnitude, normal_losses_magnitude, laplacian_losses_magnitude = [], [], []
    vectorfield_losses_inside_magnitude, vectorfield_losses_outside_magnitude = [], []
    lr_history = []

    # ── Initial diagnostics ───────────────────────────────────────────────────────
    loop = tqdm(range(Niter))

    count_inside, _  = count_vertices_in_mask(src_mesh, msk_inside,  msk_scaled_sitk)
    count_outside, _ = count_vertices_in_mask(src_mesh, msk_outside, msk_scaled_sitk)
    print(f"Initial vertex count in inside mask:  {count_inside}")
    print(f"Initial vertex count in outside mask: {count_outside}")

    # ── Optimization loop ─────────────────────────────────────────────────────────
    for i in loop:

        # ------------------------------------------------------------------
        # 1.  Compute per-phase weight schedule
        # ------------------------------------------------------------------


        if i < phase1_iters:
            # ── Phase 1: vector-field dominant, regularization at base level ──
            w_edge       = w_edge_p1
            w_normal     = w_normal_p1
            w_laplacian  = w_laplacian_p1
            w_vf_inside  = w_vf_inside_p1 
            w_vf_outside = w_vf_outside_p1

        elif i < phase1_iters + phase2_length:
            # ── Phase 2: cosine blend from Phase-1 → Phase-2 targets ──────────
            t     = (i - phase1_iters) / phase2_length          # 0 → 1
            blend = 0.5 * (1.0 - np.cos(np.pi * t))             # smooth 0 → 1

            w_edge       = w_edge_p1      + (w_edge_p2      - w_edge_p1)      * blend
            w_normal     = w_normal_p1    + (w_normal_p2    - w_normal_p1)    * blend
            w_laplacian  = w_laplacian_p1 + (w_laplacian_p2 - w_laplacian_p1) * blend
            w_vf_inside  = w_vf_inside_p1  + (w_vf_inside_p1  - w_vf_inside_p1)  * blend
            w_vf_outside = w_vf_outside_p1 + (w_vf_outside_p1 - w_vf_outside_p1) * blend

        else:
            # ── Phase 3: fully relaxed — high regularization, vector field off ─
            w_edge       = w_edge_p2
            w_normal     = w_normal_p2
            w_laplacian  = w_laplacian_p2
            w_vf_inside  = w_vf_inside_p1
            w_vf_outside = w_vf_outside_p1

        # ------------------------------------------------------------------
        # 2.  Forward pass
        # ------------------------------------------------------------------
        optimizer.zero_grad()

        new_src_mesh = src_mesh.offset_verts(deform_verts)

        loss_edge      = mesh_edge_loss(new_src_mesh)
        loss_normal    = mesh_normal_consistency(new_src_mesh)
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="cot")

        loss_vectorfield_inside = vector_field_loss_stable_directional(
            new_src_mesh, vector_field_inside_tensor, msk_scaled_sitk)
        
        loss_vectorfield_outside = vector_field_loss_stable_directional(
            new_src_mesh, vector_field_outside_tensor, msk_scaled_sitk)

        loss_deform = torch.mean(deform_verts ** 2)

        loss = (
            w_edge       * loss_edge      +
            w_normal     * loss_normal    +
            w_laplacian  * loss_laplacian +
            w_vf_inside  * loss_vectorfield_inside  +
            w_vf_outside * loss_vectorfield_outside
        )

        # ------------------------------------------------------------------
        # 3.  Logging
        # ------------------------------------------------------------------
        edge_losses_weighted    = float(w_edge      * loss_edge.detach().cpu())
        normal_losses_weighted  = float(w_normal    * loss_normal.detach().cpu())
        lap_losses_weighted     = float(w_laplacian * loss_laplacian.detach().cpu())
        vf_in_weighted          = float(w_vf_inside  * loss_vectorfield_inside.detach().cpu())
        vf_out_weighted         = float(w_vf_outside * loss_vectorfield_outside.detach().cpu())

        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))
        vectorfield_losses_inside.append(float(loss_vectorfield_inside.detach().cpu()))
        vectorfield_losses_outside.append(float(loss_vectorfield_outside.detach().cpu()))

        edge_losses_magnitude.append(edge_losses_weighted)
        normal_losses_magnitude.append(normal_losses_weighted)
        laplacian_losses_magnitude.append(lap_losses_weighted)
        vectorfield_losses_inside_magnitude.append(vf_in_weighted)
        vectorfield_losses_outside_magnitude.append(vf_out_weighted)

        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # Determine current phase label for tqdm
        if i < phase1_iters:
            phase_label = "P1-vf"
        elif i < phase1_iters + phase2_length:
            phase_label = "P2-blend"
        else:
            phase_label = "P3-relax"

        # If you want to count number of vertices
        # if (i + 1) % plot_period == 0 or i == 0:
        # count_inside, _  = count_vertices_in_mask(new_src_mesh, msk_inside,  msk_scaled_sitk)
        # count_outside, _ = count_vertices_in_mask(new_src_mesh, msk_outside, msk_scaled_sitk)
        # loop.set_description(
        #     f'loss={loss:.4f}, phase_label = {phase_label} ,count_inside={count_inside}, count_outside={count_outside}, edge={edge_losses_weighted:.4f}, normal={normal_losses_weighted:.4f}, '
        #     f'laplacian={loss_laplacian:.4f}' f'lr={current_lr:.6f}')
        
        loop.set_description(
            f'loss={loss:.4f}, phase_label = {phase_label}')

        # ------------------------------------------------------------------
        # 4.  Backward + gradient clip + step
        # ------------------------------------------------------------------
        loss.backward()
        torch.nn.utils.clip_grad_norm_([deform_verts], max_norm=max_grad_norm)
        optimizer.step()
        if i >= phase1_iters:
            scheduler.step()


    # ── Save refined mesh ─────────────────────────────────────────────────────────
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
    final_verts = final_verts * scale + center
    save_obj(output_path, final_verts, final_faces)

    # ---- Post-processing: Taubin smoothing ----
    taubin_iters = 50        # number of Taubin iterations (each = shrink + inflate)
    taubin_lambda = 0.5      # positive (shrink) factor
    taubin_mu = -0.53        # negative (inflate) factor; |mu| > lambda avoids shrinkage


    # ---- Post-processing: Taubin smoothing to remove residual oscillations ----
    if taubin_iters > 0:
        print(f"Applying Taubin smoothing ({taubin_iters} iters, λ={taubin_lambda}, μ={taubin_mu})...")
        final_verts_taubin = taubin_smoothing(
            new_src_mesh, n_iters=taubin_iters,
            lambda_pos=taubin_lambda, lambda_neg=taubin_mu
        )

    # Scale normalize back to the original target size
    final_verts_taubin = final_verts_taubin * scale + center

    # Store the predicted mesh using save_obj
    output_path_taubin = os.path.join(output_folder, series + "_refined_mesh_taubin.obj") 
    save_obj(output_path_taubin, final_verts_taubin, final_faces)



    # Plot graphs
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    ax.plot(edge_losses, label="edge loss")
    ax.plot(normal_losses, label="normal loss")
    ax.plot(laplacian_losses, label="laplacian loss")
    ax.plot(vectorfield_losses_inside, label="vector field loss")
    ax.plot(vectorfield_losses_outside, label="vector field loss (outside)")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")
    plt.savefig(os.path.join(output_folder, series + "_loss_curves.png"))
    # plt.show()



    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    ax.plot(edge_losses_magnitude, label="edge loss (weighted)", color="blue")
    ax.plot(normal_losses_magnitude, label="normal loss (weighted)", color="orange")
    ax.plot(laplacian_losses_magnitude, label="laplacian loss (weighted)", color="green")
    ax.plot(vectorfield_losses_inside_magnitude, label="vector field loss (inside)", color="red")
    ax.plot(vectorfield_losses_outside_magnitude, label="vector field loss (outside)", color="black")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss magnitude", fontsize="16")
    ax.set_title("Loss magnitude vs iterations", fontsize="16")
    plt.savefig(os.path.join(output_folder, series + "_loss_magnitude_curves.png"))
    # plt.show()

    fig = plt.figure(figsize=(13, 3))
    ax = fig.gca()
    ax.plot(lr_history, label="learning rate", color="purple")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("LR", fontsize="16")
    ax.set_title("Learning rate vs iterations", fontsize="16")
    ax.legend(fontsize="16")
    plt.savefig(os.path.join(output_folder, series + "_learning_rate_curve.png"))
    # plt.show()