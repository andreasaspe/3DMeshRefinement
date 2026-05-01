import os

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from scipy.ndimage import center_of_mass
from tqdm import tqdm

import saros_utils as utils

mpl.rcParams["savefig.dpi"] = 80
mpl.rcParams["figure.dpi"] = 80



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


output_folder_root = f"/data/awias/periseg/saros/TS_pericardium/pytorch3d_test"
data_folder = f"/data/awias/periseg/saros/NIFTI_collected_test"

csv_path = "3DMeshRefinement/series.csv"

# Vector fields folder
vf_folder = f"/data/awias/periseg/saros/TS_pericardium/vector_fields"

# ═══════════════════════════════════════════════════════════════════════
#                        MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
# Load series
df = pd.read_csv(csv_path)
all_series = df.loc[df['grid_search'] != 1, 'all_series_name'].tolist()
all_series = sorted(all_series, key=lambda x: int(x.split('_')[1]))
all_series = np.unique(all_series)
os.makedirs(vf_folder, exist_ok=True)

for series in tqdm(all_series):

    # Define output paths
    output_folder = os.path.join(output_folder_root, series)
    output_path = os.path.join(output_folder, series + "_refined_mesh.obj")
    os.makedirs(output_folder, exist_ok=True)

    ts_heartchambershighres_path = os.path.join(data_folder, series + "_ts_heartchambershighres.nii.gz")
    ts_total_path = os.path.join(data_folder, series + "_ts_total.nii.gz")
    TS_trunkcavities_path = os.path.join(data_folder, series + "_ts_trunkcavities.nii.gz")
    TS_coronaryarteries_path = os.path.join(data_folder, series + "_ts_coronaryarteries.nii.gz")
    mesh_path = os.path.join(output_folder, series + "_rawsurface.vtk")
    mesh_smoothed_path = os.path.join(output_folder, series + "_smoothedsurface.vtk")
    mesh_smoothed_path_obj = os.path.join(output_folder, series + "_smoothedsurface.obj")


    utils.convert_label_map_to_surface_file(TS_trunkcavities_path, mesh_path, segment_id=3)
    utils.decimate_and_smooth(mesh_path, mesh_smoothed_path)
    utils.convert_vtk_to_obj(mesh_smoothed_path, mesh_smoothed_path_obj)

    # Load the smoothed mesh and normalize
    verts, faces, aux = load_obj(mesh_smoothed_path_obj)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    # Create a Meshes object for the source mesh
    src_mesh = Meshes(verts=[verts], faces=[faces_idx])

    # Load heartchambers_highres mask
    msk_highres_sitk = sitk.ReadImage(ts_heartchambershighres_path)
    spacing = msk_highres_sitk.GetSpacing()
    origin = msk_highres_sitk.GetOrigin()
    # Scale
    msk_highres_sitk.SetOrigin(tuple((origin - center.cpu().numpy()) / scale.cpu().numpy()))
    msk_highres_sitk.SetSpacing(tuple(spacing / scale.cpu().numpy()))
    msk_highres = sitk.GetArrayFromImage(msk_highres_sitk)
    # Define binary masks
    structures_not_to_include = [6,7]  # example structure IDs to exclude
    for idx in structures_not_to_include:
        msk_highres[msk_highres == idx] = 0
    msk_highres_bin = np.isin(msk_highres, [1,2,3,4,5])
    # Compute COM
    COM_zyx = center_of_mass(msk_highres_bin)

    # Load total mask
    msk_total_sitk = sitk.ReadImage(ts_total_path)
    # Scale
    msk_total_sitk.SetOrigin(tuple((origin - center.cpu().numpy()) / scale.cpu().numpy()))
    msk_total_sitk.SetSpacing(tuple(spacing / scale.cpu().numpy()))
    msk_total = sitk.GetArrayFromImage(msk_total_sitk)
    # Define binary masks
    mask_segment_61 = (msk_total == 61)
    structures_not_to_include = [0, 51, 52, 53, 61, 62, 63]  # example structure IDs to exclude
    for idx in structures_not_to_include:
        msk_total[msk_total == idx] = 0

    # Load coronary arteries mask (already binary)
    msk_coronaryarteries_sitk = sitk.ReadImage(TS_coronaryarteries_path)
    # Scale
    msk_coronaryarteries_sitk.SetOrigin(tuple((origin - center.cpu().numpy()) / scale.cpu().numpy()))
    msk_coronaryarteries_sitk.SetSpacing(tuple(spacing / scale.cpu().numpy()))
    msk_coronaryarteries = sitk.GetArrayFromImage(msk_coronaryarteries_sitk)

    # internal
    msk_internal = (msk_coronaryarteries > 0).copy()
    hr_mask = msk_highres > 0
    msk_internal |= hr_mask
    msk_internal |= mask_segment_61


    # external
    msk_external = (msk_total > 0).copy()

    # scaled_sitk
    msk_scaled_sitk = msk_highres_sitk


    # Vector field file paths
    # ── Setup Single Path ────────────────────────────────────────────────
    vf_combined_path = os.path.join(vf_folder, f"{series}_VF_all.npy")

    Vx_external, Vy_external, Vz_external, Vx_internal, Vy_internal, Vz_internal = utils.load_or_create_vector_fields(msk_internal, msk_external, COM_zyx, spacing, vf_combined_path)

    vector_field_internal = np.stack([Vx_internal, Vy_internal, Vz_internal], axis=0)
    vector_field_internal_tensor = torch.from_numpy(vector_field_internal).float().to(device)

    vector_field_external = np.stack([Vx_external, Vy_external, Vz_external], axis=0)
    vector_field_external_tensor = torch.from_numpy(vector_field_external).float().to(device)



    # ── Define self parameters ────────────────────────────────────────────────────────────────
    phase1_iters              = 1000
    phase2_length             = 500
    phase3_length             = 500
    Niter                     = phase1_iters + phase2_length + phase3_length
    w_edge_p1                 = 0.001
    w_laplacian_p1            = 0.015
    w_normal_p1               = 0.001
    w_vf_internal_p1            = 1
    w_vf_external_p1           = 0.35
    w_edge_p2                 = w_edge_p1
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
    vectorfield_losses_internal, vectorfield_losses_external = [], []
    edge_losses_magnitude, normal_losses_magnitude, laplacian_losses_magnitude = [], [], []
    vectorfield_losses_internal_magnitude, vectorfield_losses_external_magnitude = [], []
    lr_history = []

    # ── Initial diagnostics ───────────────────────────────────────────────────────
    loop = tqdm(range(Niter))

    count_internal, _ = utils.count_vertices_in_mask(src_mesh, msk_internal, msk_scaled_sitk)
    count_external, _ = utils.count_vertices_in_mask(src_mesh, msk_external, msk_scaled_sitk)
    print(f"Initial vertex count in internal mask:  {count_internal}")
    print(f"Initial vertex count in external mask: {count_external}")

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
            w_vf_internal  = w_vf_internal_p1 
            w_vf_external = w_vf_external_p1

        elif i < phase1_iters + phase2_length:
            # ── Phase 2: cosine blend from Phase-1 → Phase-2 targets ──────────
            t     = (i - phase1_iters) / phase2_length          # 0 → 1
            blend = 0.5 * (1.0 - np.cos(np.pi * t))             # smooth 0 → 1

            w_edge       = w_edge_p1      + (w_edge_p2      - w_edge_p1)      * blend
            w_normal     = w_normal_p1    + (w_normal_p2    - w_normal_p1)    * blend
            w_laplacian  = w_laplacian_p1 + (w_laplacian_p2 - w_laplacian_p1) * blend
            w_vf_internal  = w_vf_internal_p1  + (w_vf_internal_p1  - w_vf_internal_p1)  * blend
            w_vf_external = w_vf_external_p1 + (w_vf_external_p1 - w_vf_external_p1) * blend

        else:
            # ── Phase 3: fully relaxed — high regularization, vector field off ─
            w_edge       = w_edge_p2
            w_normal     = w_normal_p2
            w_laplacian  = w_laplacian_p2
            w_vf_internal  = w_vf_internal_p1
            w_vf_external = w_vf_external_p1

        # ------------------------------------------------------------------
        # 2.  Forward pass
        # ------------------------------------------------------------------
        optimizer.zero_grad()

        new_src_mesh = src_mesh.offset_verts(deform_verts)

        loss_edge      = mesh_edge_loss(new_src_mesh)
        loss_normal    = mesh_normal_consistency(new_src_mesh)
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="cot")

        loss_vectorfield_internal = utils.vector_field_loss_stable_directional(
            new_src_mesh, vector_field_internal_tensor, msk_scaled_sitk
        )

        loss_vectorfield_external = utils.vector_field_loss_stable_directional(
            new_src_mesh, vector_field_external_tensor, msk_scaled_sitk
        )

        loss_deform = torch.mean(deform_verts ** 2)

        loss = (
            w_edge       * loss_edge      +
            w_normal     * loss_normal    +
            w_laplacian  * loss_laplacian +
            w_vf_internal  * loss_vectorfield_internal  +
            w_vf_external * loss_vectorfield_external
        )

        # ------------------------------------------------------------------
        # 3.  Logging
        # ------------------------------------------------------------------
        edge_losses_weighted    = float(w_edge      * loss_edge.detach().cpu())
        normal_losses_weighted  = float(w_normal    * loss_normal.detach().cpu())
        lap_losses_weighted     = float(w_laplacian * loss_laplacian.detach().cpu())
        vf_in_weighted          = float(w_vf_internal  * loss_vectorfield_internal.detach().cpu())
        vf_out_weighted         = float(w_vf_external * loss_vectorfield_external.detach().cpu())

        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))
        vectorfield_losses_internal.append(float(loss_vectorfield_internal.detach().cpu()))
        vectorfield_losses_external.append(float(loss_vectorfield_external.detach().cpu()))

        edge_losses_magnitude.append(edge_losses_weighted)
        normal_losses_magnitude.append(normal_losses_weighted)
        laplacian_losses_magnitude.append(lap_losses_weighted)
        vectorfield_losses_internal_magnitude.append(vf_in_weighted)
        vectorfield_losses_external_magnitude.append(vf_out_weighted)

        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # Determine current phase label for tqdm
        if i < phase1_iters:
            phase_label = "P1-vf"
        elif i < phase1_iters + phase2_length:
            phase_label = "P2-blend"
        else:
            phase_label = "P3-relax"

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
    taubin_iters = 50
    taubin_lambda = 0.5
    taubin_mu = -0.53


    # ---- Post-processing: Taubin smoothing to remove residual oscillations ----
    if taubin_iters > 0:
        print(f"Applying Taubin smoothing ({taubin_iters} iters, λ={taubin_lambda}, μ={taubin_mu})...")
        final_verts_taubin = utils.taubin_smoothing(
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
    ax.plot(vectorfield_losses_internal, label="vector field loss")
    ax.plot(vectorfield_losses_external, label="vector field loss (external)")
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
    ax.plot(vectorfield_losses_internal_magnitude, label="vector field loss (internal)", color="red")
    ax.plot(vectorfield_losses_external_magnitude, label="vector field loss (external)", color="black")
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