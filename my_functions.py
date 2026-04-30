import vtk
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import center_of_mass
import edt
from pytorch3d.ops import sample_points_from_meshes
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import tools as tools
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from numpy import dot
import edt
from skimage.measure import find_contours
import warnings
warnings.filterwarnings("ignore", message="No mtl file provided") # Ignore this ugly warning.



def decimate_and_smooth(surface_in, surface_out, target_points=80000):
    """
    This function decimates (minimizes) and smooths the surface mesh
    It reads the surface mesh, decimates it and smooths it
    It saves the surface mesh in the output directory
    
    Args:
        surface_in: Path to input VTK mesh
        surface_out: Path to output VTK mesh
        target_points: Target number of vertices (default: 10000)
    """

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(surface_in)
    reader.Update()

    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(reader.GetOutput())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(conn.GetOutput())
    fill_holes.SetHoleSize(1000.0)
    fill_holes.Update()

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(fill_holes.GetOutput())
    triangle.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(triangle.GetOutput())
    cleaner.Update()

    # Calculate reduction ratio based on target number of points
    initial_points = cleaner.GetOutput().GetNumberOfPoints()
    print(f"Initial mesh has {initial_points} points")
    
    if initial_points > target_points:
        target_reduction = 1.0 - (target_points / initial_points)
        
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(cleaner.GetOutput())
        decimate.SetTargetReduction(target_reduction)
        decimate.SplittingOn()
        decimate.SetMaximumError(10)
        decimate.PreserveTopologyOn()
        decimate.Update()
        
        print(f"After decimation: {decimate.GetOutput().GetNumberOfPoints()} points")
        decimated_mesh = decimate.GetOutput()
    else:
        print(f"Mesh already has {initial_points} points (target: {target_points}), skipping decimation")
        decimated_mesh = cleaner.GetOutput()

    smooth_filter = vtk.vtkConstrainedSmoothingFilter()
    smooth_filter.SetInputData(decimated_mesh)
    smooth_filter.SetNumberOfIterations(1000)
    smooth_filter.SetRelaxationFactor(0.01)
    smooth_filter.SetConstraintDistance(5)
    smooth_filter.SetConstraintStrategyToConstraintDistance()
    smooth_filter.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(smooth_filter.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.Update()

    final_points = normals.GetOutput().GetNumberOfPoints()
    print(f"Final mesh has {final_points} points")
    print(f"Saving reduced surface mesh to {surface_out}")
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(surface_out)
    writer.SetInputData(normals.GetOutput())
    writer.Write()



def decimate_and_smooth_ALOT(surface_in, surface_out):
    """
    This function decimates (minimizes) and smooths the surface mesh
    It reads the surface mesh, decimates it and smooths it
    It saves the surface mesh in the output directory
    """

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(surface_in)
    reader.Update()

    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(reader.GetOutput())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    # print("Filling holes")
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(conn.GetOutput())
    fill_holes.SetHoleSize(1000.0)
    fill_holes.Update()

    # print("Triangle filter")
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(fill_holes.GetOutput())
    triangle.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(triangle.GetOutput())
    cleaner.Update()

    # Calculate reduction ratio based on target number of points
    initial_points = cleaner.GetOutput().GetNumberOfPoints()
    print(f"Initial mesh has {initial_points} points")

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(cleaner.GetOutput())
    decimate.SetTargetReduction(0.95)
    decimate.SplittingOn()
    decimate.SetMaximumError(10)
    decimate.PreserveTopologyOn()
    decimate.Update()

    print(f"After decimation: {decimate.GetOutput().GetNumberOfPoints()} points")

    smooth_filter = vtk.vtkConstrainedSmoothingFilter()
    smooth_filter.SetInputData(decimate.GetOutput())
    smooth_filter.SetNumberOfIterations(1000)
    smooth_filter.SetRelaxationFactor(0.01)
    smooth_filter.SetConstraintDistance(5)
    smooth_filter.SetConstraintStrategyToConstraintDistance()
    smooth_filter.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(smooth_filter.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.Update()

    print(f"Saving reduced surface mesh")
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(surface_out)
    writer.SetInputData(normals.GetOutput())
    writer.Write()



def create_sdf_from_ts_mask(ts_path, list_of_indices, output_path=None):

    """
    This function creates a signed distance function (SDF) from the TS mask
    It reads the TS mask, creates the SDF and saves it in the output directory
    """

    #Create sdf from ts mask
    ts_sitk = sitk.ReadImage(ts_path)
    ts = sitk.GetArrayFromImage(ts_sitk).astype(np.float32)

    spacing = ts_sitk.GetSpacing()

    # Combine labels
    ts_combined = np.isin(ts, list_of_indices)
    # ts_combined = (ts == 1) | (ts == 2) | (ts == 3) | (ts == 4) | (ts == 5) #| (ts == 6)

    ts_combined_sitk = sitk.GetImageFromArray(ts_combined.astype(np.uint8))
    ts_combined_sitk.CopyInformation(ts_sitk)
    sitk.WriteImage(ts_combined_sitk, output_path.replace("_sdf.nii.gz", "_combined_ts.nii.gz"))

    sdf_ts = -edt.sdf(ts_combined,
                anisotropy=[spacing[2], spacing[1], spacing[0]], #Spacing is x, y, z per default
                parallel=8  # number of threads, <= 0 sets to num CPU
                )


    sdf_ts_sitk = sitk.GetImageFromArray(sdf_ts)
    sdf_ts_sitk.CopyInformation(ts_sitk)

    if output_path is not None:
        sitk.WriteImage(sdf_ts_sitk, output_path)
        print(f"Saved SDF to {output_path}")

    return sdf_ts_sitk




import SimpleITK as sitk
import numpy as np


def create_vector_field_to_com(ts_path, list_of_indices, COM_phys, output_path=None):
    """
    Creates a vector field pointing toward a physical point (COM_phys).
    The vector field is defined only inside the combined TS mask.
    """

    # Read TS mask
    ts_sitk = sitk.ReadImage(ts_path)
    ts = sitk.GetArrayFromImage(ts_sitk).astype(np.float32)

    spacing = ts_sitk.GetSpacing()
    origin = ts_sitk.GetOrigin()
    direction = np.array(ts_sitk.GetDirection()).reshape(3, 3)

    # Combine labels
    ts_combined = np.isin(ts, list_of_indices)

    ts_combined_sitk = sitk.GetImageFromArray(ts_combined.astype(np.uint8))
    ts_combined_sitk.CopyInformation(ts_sitk)

    # Get grid of voxel indices
    z_dim, y_dim, x_dim = ts.shape
    zz, yy, xx = np.meshgrid(
        np.arange(z_dim),
        np.arange(y_dim),
        np.arange(x_dim),
        indexing="ij"
    )

    # Convert voxel indices to physical coordinates
    # IMPORTANT: sitk index order is (x,y,z), array order is (z,y,x)
    indices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    physical_coords = []
    for idx in indices:
        phys = ts_sitk.TransformIndexToPhysicalPoint(tuple(idx.tolist()))
        physical_coords.append(phys)

    physical_coords = np.array(physical_coords).reshape(z_dim, y_dim, x_dim, 3)

    # Compute vector field (COM - position)
    vector_field = COM_phys - physical_coords

    # Zero out vectors outside mask
    vector_field[~ts_combined] = 0.0

    # Convert to sitk vector image
    vector_sitk = sitk.GetImageFromArray(vector_field.astype(np.float32), isVector=True)
    vector_sitk.CopyInformation(ts_sitk)

    if output_path is not None:
        sitk.WriteImage(vector_sitk, output_path)
        print(f"Saved vector field to {output_path}")

    return vector_sitk



def count_vertices_in_mask(mesh, msk_total, sitk_ref):
    """
    Counts how many mesh vertices fall inside the region where msk_total == 1.

    Args:
        mesh: Meshes object (e.g., new_src_mesh)
        msk_total: numpy array, mask volume (1 = inside, 0 = outside)
        sitk_ref: SimpleITK image, reference for origin/spacing

    Returns:
        count: int, number of vertices inside mask
        indices: np.ndarray, indices of inside vertices
    """
    verts = mesh.verts_packed().detach().cpu().numpy()  # (V, 3) in (x, y, z)
    origin = np.array(sitk_ref.GetOrigin())  # (x, y, z)
    spacing = np.array(sitk_ref.GetSpacing())  # (x, y, z)
    size = np.array(sitk_ref.GetSize())  # (W, H, D)

    # Convert physical coordinates to voxel indices
    voxel_idx = (verts - origin) / spacing
    voxel_idx = np.round(voxel_idx).astype(int)

    # Clip indices to volume bounds
    voxel_idx[:, 0] = np.clip(voxel_idx[:, 0], 0, size[0] - 1)
    voxel_idx[:, 1] = np.clip(voxel_idx[:, 1], 0, size[1] - 1)
    voxel_idx[:, 2] = np.clip(voxel_idx[:, 2], 0, size[2] - 1)

    # msk_total shape: (D, H, W), voxel_idx: (V, 3) in (x, y, z)
    # Need to map (x, y, z) to (z, y, x)
    inside = msk_total[voxel_idx[:, 2], voxel_idx[:, 1], voxel_idx[:, 0]] == 1
    count = np.sum(inside)
    indices = np.where(inside)[0]
    return count, indices



def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()





def plot_vector_field_slice(img, msk, Vx, Vy, slice_idx, step=5, scale=0.5, figsize=(8, 8), COM_yx=None, title=None, output_path=None, min_magnitude=1e-6):
    """
    Plot a slice of the image with mask overlay and vector field arrows.
    Arrows pointing toward COM are green, arrows pointing away are red.

    Args:
        img: 3D numpy array (Z, Y, X) - the image volume
        msk: 3D numpy array (Z, Y, X) - the mask volume
        Vx: 3D numpy array (Z, Y, X) - X-component of vector field
        Vy: 3D numpy array (Z, Y, X) - Y-component of vector field
        slice_idx: int - which Z slice to visualize
        step: int - spacing between arrows (default: 5)
        scale: float - arrow scaling factor (default: 0.5)
        figsize: tuple - figure size (default: (8, 8))
        COM_yx: tuple (y, x) - center of mass in pixel coords for the slice.
                If None, all arrows are red.
        title: str or None - optional figure title. If None, no title is shown.
        output_path: str or None - if provided, save the figure at 300 DPI.
        min_magnitude: float - minimum vector magnitude to display an arrow (default: 1e-6).
    """
    img_slice = img[slice_idx, :, :]
    msk_slice = msk[slice_idx, :, :]
    Vx_slice = Vx[slice_idx, :, :]
    Vy_slice = Vy[slice_idx, :, :]

    plt.figure(figsize=figsize)
    plt.imshow(img_slice, cmap='gray', origin='lower')

    # Mask overlay
    alpha_mask = np.zeros_like(msk_slice, dtype=float)
    alpha_mask[msk_slice != 0] = 0.5
    plt.imshow(msk_slice, cmap='Blues', alpha=alpha_mask, origin='lower')

    # Quiver plot
    x = np.arange(0, img_slice.shape[1], step)
    y = np.arange(0, img_slice.shape[0], step)
    X, Y = np.meshgrid(x, y)

    U = Vx_slice[::step, ::step]
    V = Vy_slice[::step, ::step]

    # Normalize arrows to unit length so all arrows display at the same size
    mag = np.sqrt(U**2 + V**2)

    # Filter out arrows with magnitude below threshold
    keep = mag.ravel() >= min_magnitude
    X_f, Y_f = X.ravel()[keep], Y.ravel()[keep]
    mag_safe = mag.copy(); mag_safe[mag_safe == 0] = 1.0
    U_norm = (U / mag_safe).ravel()[keep]
    V_norm = (V / mag_safe).ravel()[keep]

    if COM_yx is None:
        # No COM provided — plot all arrows in red
        plt.quiver(X_f, Y_f, U_norm, V_norm, color='red', angles='xy', scale_units='xy', scale=scale)
    else:
        # Direction from each arrow origin toward COM
        dir_to_com_x = COM_yx[1] - X
        dir_to_com_y = COM_yx[0] - Y

        # Dot product: positive => toward COM (green), negative => away (red)
        dot = U * dir_to_com_x + V * dir_to_com_y

        # Build per-arrow color array (single quiver call to avoid independent scaling)
        colors = np.where(dot.ravel() >= 0, '#00CC00', 'red')[keep]
        plt.quiver(X_f, Y_f, U_norm, V_norm, color=colors,
                   angles='xy', scale_units='xy', scale=scale)

    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_vector_field_slice3d(img, msk, Vx, Vy, Vz, slice_idx, dim='z',
                              step=5, scale=0.5, figsize=(8, 8), COM_zyx=None,
                              title=None, output_path=None, min_magnitude=1e-6):
    """
    Plot a slice of the image with mask overlay and vector field arrows,
    slicing along any of the three axes.

    Arrows pointing toward COM (projected onto the slice plane) are green,
    arrows pointing away are red.  If COM_zyx is None, all arrows are red.

    The volume axes are (Z, Y, X).  `dim` selects which axis to slice:
        dim='z' → slice axis 0  (Y-X plane, arrows from Vx & Vy)
        dim='y' → slice axis 1  (Z-X plane, arrows from Vx & Vz)
        dim='x' → slice axis 2  (Z-Y plane, arrows from Vy & Vz)

    Args:
        img:  3D numpy array (Z, Y, X) - the image volume
        msk:  3D numpy array (Z, Y, X) - the mask volume
        Vx:   3D numpy array (Z, Y, X) - X-component of vector field
        Vy:   3D numpy array (Z, Y, X) - Y-component of vector field
        Vz:   3D numpy array (Z, Y, X) - Z-component of vector field
        slice_idx: int - index along the chosen dimension
        dim:   str  - slicing dimension: 'z' (default), 'y', or 'x'
        step:  int   - spacing between arrows (default: 5)
        scale: float - arrow scaling factor (default: 0.5)
        figsize: tuple - figure size (default: (8, 8))
        COM_zyx: tuple (z, y, x) - 3D center of mass in voxel coords.
                 The 2D projection onto the slice plane is computed
                 automatically.  If None, all arrows are red.
        title: str or None - optional figure title. If None, no title is shown.
        output_path: str or None - if provided, save the figure at 300 DPI.
        min_magnitude: float - minimum vector magnitude to display an arrow (default: 1e-6).
    """
    dim = dim.lower()
    if dim not in ('z', 'y', 'x'):
        raise ValueError(f"dim must be 'z', 'y', or 'x', got '{dim}'")

    # --- slice volumes and pick the two in-plane vector components ---
    if dim == 'z':
        # axis 0 → (Y, X) plane; rows=Y, cols=X
        img_slice = img[slice_idx, :, :]
        msk_slice = msk[slice_idx, :, :]
        U_full = Vx[slice_idx, :, :]   # horizontal (X)
        V_full = Vy[slice_idx, :, :]   # vertical   (Y)
        axis_labels = ('X', 'Y')
        # COM projection: (row, col) = (y, x)
        com_2d = (COM_zyx[1], COM_zyx[2]) if COM_zyx is not None else None
    elif dim == 'y':
        # axis 1 → (Z, X) plane; rows=Z, cols=X
        img_slice = img[:, slice_idx, :]
        msk_slice = msk[:, slice_idx, :]
        U_full = Vx[:, slice_idx, :]   # horizontal (X)
        V_full = Vz[:, slice_idx, :]   # vertical   (Z)
        axis_labels = ('X', 'Z')
        # COM projection: (row, col) = (z, x)
        com_2d = (COM_zyx[0], COM_zyx[2]) if COM_zyx is not None else None
    else:  # dim == 'x'
        # axis 2 → (Z, Y) plane; rows=Z, cols=Y
        img_slice = img[:, :, slice_idx]
        msk_slice = msk[:, :, slice_idx]
        U_full = Vy[:, :, slice_idx]   # horizontal (Y)
        V_full = Vz[:, :, slice_idx]   # vertical   (Z)
        axis_labels = ('Y', 'Z')
        # COM projection: (row, col) = (z, y)
        com_2d = (COM_zyx[0], COM_zyx[1]) if COM_zyx is not None else None

    plt.figure(figsize=figsize)
    plt.imshow(img_slice, cmap='gray', origin='lower')

    # Mask overlay
    alpha_mask = np.zeros_like(msk_slice, dtype=float)
    alpha_mask[msk_slice != 0] = 0.5
    plt.imshow(msk_slice, cmap='Blues', alpha=alpha_mask, origin='lower')

    # Quiver grid
    cols = np.arange(0, img_slice.shape[1], step)
    rows = np.arange(0, img_slice.shape[0], step)
    C, R = np.meshgrid(cols, rows)

    U = U_full[::step, ::step]
    V = V_full[::step, ::step]

    # Normalize arrows to unit length so all arrows display at the same size
    mag = np.sqrt(U**2 + V**2)

    # Filter out arrows with magnitude below threshold
    keep = mag.ravel() >= min_magnitude
    C_f, R_f = C.ravel()[keep], R.ravel()[keep]
    mag_safe = mag.copy(); mag_safe[mag_safe == 0] = 1.0
    U_norm = (U / mag_safe).ravel()[keep]
    V_norm = (V / mag_safe).ravel()[keep]

    if com_2d is None:
        plt.quiver(C_f, R_f, U_norm, V_norm, color='red', angles='xy', scale_units='xy', scale=scale)
    else:
        dir_to_com_c = com_2d[1] - C
        dir_to_com_r = com_2d[0] - R

        dot = U * dir_to_com_c + V * dir_to_com_r

        # Build per-arrow color array (single quiver call to avoid independent scaling)
        colors = np.where(dot.ravel() > 0, '#00CC00', 'red')[keep]
        plt.quiver(C_f, R_f, U_norm, V_norm, color=colors,
                   angles='xy', scale_units='xy', scale=scale)

    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()







def plot_vector_field_slice3d_3dprojectedcoloring(img, msk, Vx, Vy, Vz, slice_idx, dim='z',
                              step=5, scale=0.5, figsize=(8, 8), COM_zyx=None,
                              title=None, output_path=None, min_magnitude=1e-6):
    # The other function with almost the same name just projects the arrows in 2D. So in the 2D image they are in fact projectet such that they point away and thus red arrows. But in 3D, they point the right way. This function colors according the the real 3D value direction, not the 2D projection direction. So if the 3D vector points toward COM, it is green, if it points away, it is red, even if the 2D projection might look different.
    dim = dim.lower()
    if dim not in ('z', 'y', 'x'):
        raise ValueError(f"dim must be 'z', 'y', or 'x', got '{dim}'")

    # ── Precompute 3D dot product with toward-COM direction ───────────────
    if COM_zyx is not None:
        Z, Y, X = np.ogrid[:Vx.shape[0], :Vx.shape[1], :Vx.shape[2]]
        dz = COM_zyx[0] - Z
        dy = COM_zyx[1] - Y
        dx = COM_zyx[2] - X
        dist = np.sqrt(dz**2 + dy**2 + dx**2) + 1e-8
        # 3D dot: Vx*dx/dist + Vy*dy/dist + Vz*dz/dist
        dot_3d = Vx * (dx / dist) + Vy * (dy / dist) + Vz * (dz / dist)  # (Z, Y, X)

    # ── Slice volumes and pick in-plane vector components ─────────────────
    if dim == 'z':
        img_slice = img[slice_idx, :, :]
        msk_slice = msk[slice_idx, :, :]
        U_full    = Vx[slice_idx, :, :]
        V_full    = Vy[slice_idx, :, :]
        dot_slice = dot_3d[slice_idx, :, :] if COM_zyx is not None else None
    elif dim == 'y':
        img_slice = img[:, slice_idx, :]
        msk_slice = msk[:, slice_idx, :]
        U_full    = Vx[:, slice_idx, :]
        V_full    = Vz[:, slice_idx, :]
        dot_slice = dot_3d[:, slice_idx, :] if COM_zyx is not None else None
    else:
        img_slice = img[:, :, slice_idx]
        msk_slice = msk[:, :, slice_idx]
        U_full    = Vy[:, :, slice_idx]
        V_full    = Vz[:, :, slice_idx]
        dot_slice = dot_3d[:, :, slice_idx] if COM_zyx is not None else None

    plt.figure(figsize=figsize)
    plt.imshow(img_slice, cmap='gray', origin='lower')

    alpha_mask = np.zeros_like(msk_slice, dtype=float)
    alpha_mask[msk_slice != 0] = 0.5
    plt.imshow(msk_slice, cmap='Blues', alpha=alpha_mask, origin='lower')

    cols = np.arange(0, img_slice.shape[1], step)
    rows = np.arange(0, img_slice.shape[0], step)
    C, R = np.meshgrid(cols, rows)

    U   = U_full[::step, ::step]
    V   = V_full[::step, ::step]
    mag = np.sqrt(U**2 + V**2)

    keep     = mag.ravel() >= min_magnitude
    mag_safe = mag.copy(); mag_safe[mag_safe == 0] = 1.0
    U_norm   = (U / mag_safe).ravel()[keep]
    V_norm   = (V / mag_safe).ravel()[keep]
    C_f, R_f = C.ravel()[keep], R.ravel()[keep]

    if COM_zyx is None:
        plt.quiver(C_f, R_f, U_norm, V_norm, color='red',
                   angles='xy', scale_units='xy', scale=scale)
    else:
        # Use the 3D dot product for coloring — not the 2D projection
        dot_sampled = dot_slice[::step, ::step].ravel()[keep]
        colors = np.where(dot_sampled > 0, '#00CC00', 'red')
        plt.quiver(C_f, R_f, U_norm, V_norm, color=colors,
                   angles='xy', scale_units='xy', scale=scale)

    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()




def plot_vector_field_slice3d_3dprojectedcoloring_nonorm(img, msk, Vx, Vy, Vz, slice_idx, dim='z',
                              step=5, scale=0.5, figsize=(8, 8), COM_zyx=None,
                              title=None, output_path=None, min_magnitude=1e-6, hardcode_parameters = True):
    # The other function with almost the same name just projects the arrows in 2D. So in the 2D image they are in fact projectet such that they point away and thus red arrows. But in 3D, they point the right way. This function colors according the the real 3D value direction, not the 2D projection direction. So if the 3D vector points toward COM, it is green, if it points away, it is red, even if the 2D projection might look different.
    dim = dim.lower()
    if dim not in ('z', 'y', 'x'):
        raise ValueError(f"dim must be 'z', 'y', or 'x', got '{dim}'")

    # ── Precompute 3D dot product with toward-COM direction ───────────────
    if COM_zyx is not None:
        Z, Y, X = np.ogrid[:Vx.shape[0], :Vx.shape[1], :Vx.shape[2]]
        dz = COM_zyx[0] - Z
        dy = COM_zyx[1] - Y
        dx = COM_zyx[2] - X
        dist = np.sqrt(dz**2 + dy**2 + dx**2) + 1e-8
        # 3D dot: Vx*dx/dist + Vy*dy/dist + Vz*dz/dist
        dot_3d = Vx * (dx / dist) + Vy * (dy / dist) + Vz * (dz / dist)  # (Z, Y, X)

    # ── Slice volumes and pick in-plane vector components ─────────────────
    if dim == 'z':
        img_slice = img[slice_idx, :, :]
        msk_slice = msk[slice_idx, :, :]
        U_full    = Vx[slice_idx, :, :]
        V_full    = Vy[slice_idx, :, :]
        dot_slice = dot_3d[slice_idx, :, :] if COM_zyx is not None else None
    elif dim == 'y':
        img_slice = img[:, slice_idx, :]
        msk_slice = msk[:, slice_idx, :]
        U_full    = Vx[:, slice_idx, :]
        V_full    = Vz[:, slice_idx, :]
        dot_slice = dot_3d[:, slice_idx, :] if COM_zyx is not None else None
    else:
        img_slice = img[:, :, slice_idx]
        msk_slice = msk[:, :, slice_idx]
        U_full    = Vy[:, :, slice_idx]
        V_full    = Vz[:, :, slice_idx]
        dot_slice = dot_3d[:, :, slice_idx] if COM_zyx is not None else None

    plt.figure(figsize=figsize)
    plt.imshow(img_slice, cmap='gray', origin='lower')

    alpha_mask = np.zeros_like(msk_slice, dtype=float)
    alpha_mask[msk_slice != 0] = 0.5
    plt.imshow(msk_slice, cmap='Blues', alpha=alpha_mask, origin='lower')

    cols = np.arange(0, img_slice.shape[1], step)
    rows = np.arange(0, img_slice.shape[0], step)
    C, R = np.meshgrid(cols, rows)

    U   = U_full[::step, ::step]
    V   = V_full[::step, ::step]
    # Calculate magnitude just to filter out the tiny ones
    mag = np.sqrt(U**2 + V**2)
    keep = mag.ravel() >= min_magnitude

    # Flatten and filter the true unnormalized components!
    U_true = U.ravel()[keep]
    V_true = V.ravel()[keep]
    C_f, R_f = C.ravel()[keep], R.ravel()[keep]

    if COM_zyx is None:
        plt.quiver(C_f, R_f, U_true, V_true, color='red',
                   angles='xy', scale_units='xy', scale=scale)
    else:
        # Use the 3D dot product for coloring — not the 2D projection
        dot_sampled = dot_slice[::step, ::step].ravel()[keep]
        colors = np.where(dot_sampled > 0, '#00CC00', 'red')
        if hardcode_parameters:
            plt.quiver(C_f, R_f, U_true, V_true, color=colors,
            angles='xy', scale_units='xy', scale=scale, width=0.003,
            headwidth=3*0.9, headlength=5*0.9, headaxislength=4.5*0.9)
        else:
            plt.quiver(C_f, R_f, U_true, V_true, color=colors,
                   angles='xy', scale_units='xy', scale=scale)


    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()




def plot_vector_field_slice3d_outline(img, msk, Vx, Vy, Vz, slice_idx, outline_mask,
                                      dim='z', step=5, scale=0.5, figsize=(8, 8),
                                      COM_zyx=None, title=None, output_path=None,
                                      outline_color='lime', outline_linewidth=2,
                                      min_magnitude=1e-6):
    """
    Same as plot_vector_field_slice3d but with an additional contour outline
    of a segmentation mask drawn on top.

    Args:
        img:  3D numpy array (Z, Y, X) - the image volume
        msk:  3D numpy array (Z, Y, X) - the mask volume (blue overlay)
        Vx, Vy, Vz: 3D numpy arrays - vector field components
        slice_idx: int - index along the chosen dimension
        outline_mask: 3D numpy array (Z, Y, X) - mask whose non-zero contour
                      is drawn as an outline on top of everything.
        dim:   str  - slicing dimension: 'z', 'y', or 'x'
        step:  int   - spacing between arrows (default: 5)
        scale: float - arrow scaling factor (default: 0.5)
        figsize: tuple - figure size (default: (8, 8))
        COM_zyx: tuple (z, y, x) or None - 3D center of mass in voxel coords.
        title: str or None - optional figure title.
        output_path: str or None - if provided, save the figure at 300 DPI.
        outline_color: str - contour color (default: 'lime').
        outline_linewidth: float - contour line width (default: 2).
        min_magnitude: float - minimum vector magnitude to display an arrow (default: 1e-6).
    """
    from skimage.measure import find_contours

    dim = dim.lower()
    if dim not in ('z', 'y', 'x'):
        raise ValueError(f"dim must be 'z', 'y', or 'x', got '{dim}'")

    if dim == 'z':
        img_slice = img[slice_idx, :, :]
        msk_slice = msk[slice_idx, :, :]
        outline_slice = outline_mask[slice_idx, :, :]
        U_full = Vx[slice_idx, :, :]
        V_full = Vy[slice_idx, :, :]
        com_2d = (COM_zyx[1], COM_zyx[2]) if COM_zyx is not None else None
    elif dim == 'y':
        img_slice = img[:, slice_idx, :]
        msk_slice = msk[:, slice_idx, :]
        outline_slice = outline_mask[:, slice_idx, :]
        U_full = Vx[:, slice_idx, :]
        V_full = Vz[:, slice_idx, :]
        com_2d = (COM_zyx[0], COM_zyx[2]) if COM_zyx is not None else None
    else:  # 'x'
        img_slice = img[:, :, slice_idx]
        msk_slice = msk[:, :, slice_idx]
        outline_slice = outline_mask[:, :, slice_idx]
        U_full = Vy[:, :, slice_idx]
        V_full = Vz[:, :, slice_idx]
        com_2d = (COM_zyx[0], COM_zyx[1]) if COM_zyx is not None else None

    plt.figure(figsize=figsize)
    plt.imshow(img_slice, cmap='gray', origin='lower')

    # Mask overlay
    alpha_mask = np.zeros_like(msk_slice, dtype=float)
    alpha_mask[msk_slice != 0] = 0.5
    plt.imshow(msk_slice, cmap='Blues', alpha=alpha_mask, origin='lower')

    # Quiver grid
    cols = np.arange(0, img_slice.shape[1], step)
    rows = np.arange(0, img_slice.shape[0], step)
    C, R = np.meshgrid(cols, rows)

    U = U_full[::step, ::step]
    V = V_full[::step, ::step]

    mag = np.sqrt(U**2 + V**2)

    # Filter out arrows with magnitude below threshold
    keep = mag.ravel() >= min_magnitude
    C_f, R_f = C.ravel()[keep], R.ravel()[keep]
    mag_safe = mag.copy(); mag_safe[mag_safe == 0] = 1.0
    U_norm = (U / mag_safe).ravel()[keep]
    V_norm = (V / mag_safe).ravel()[keep]

    if com_2d is None:
        plt.quiver(C_f, R_f, U_norm, V_norm, color='red', angles='xy', scale_units='xy', scale=scale)
    else:
        dir_to_com_c = com_2d[1] - C
        dir_to_com_r = com_2d[0] - R
        dot = U * dir_to_com_c + V * dir_to_com_r
        colors = np.where(dot.ravel() >= 0, '#00CC00', 'red')[keep]
        plt.quiver(C_f, R_f, U_norm, V_norm, color=colors,
                   angles='xy', scale_units='xy', scale=scale)

    # Draw contour outline
    binary_outline = (outline_slice > 0).astype(float)
    contours = find_contours(binary_outline, level=0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color=outline_color,
                 linewidth=outline_linewidth)

    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_vector_field_slice3d_3dprojectedcoloring_nonorm_outline(img, msk, Vx, Vy, Vz, slice_idx, outline_mask, dim='z',
                              step=5, scale=0.5, figsize=(8, 8), COM_zyx=None,
                              title=None, output_path=None, min_magnitude=1e-6, outline_width=2.0, outline_color = 'lime', hardcode_parameters = True):
    # The other function with almost the same name just projects the arrows in 2D. So in the 2D image they are in fact projectet such that they point away and thus red arrows. But in 3D, they point the right way. This function colors according the the real 3D value direction, not the 2D projection direction. So if the 3D vector points toward COM, it is green, if it points away, it is red, even if the 2D projection might look different.
    dim = dim.lower()
    if dim not in ('z', 'y', 'x'):
        raise ValueError(f"dim must be 'z', 'y', or 'x', got '{dim}'")

    # ── Precompute 3D dot product with toward-COM direction ───────────────
    if COM_zyx is not None:
        Z, Y, X = np.ogrid[:Vx.shape[0], :Vx.shape[1], :Vx.shape[2]]
        dz = COM_zyx[0] - Z
        dy = COM_zyx[1] - Y
        dx = COM_zyx[2] - X
        dist = np.sqrt(dz**2 + dy**2 + dx**2) + 1e-8
        # 3D dot: Vx*dx/dist + Vy*dy/dist + Vz*dz/dist
        dot_3d = Vx * (dx / dist) + Vy * (dy / dist) + Vz * (dz / dist)  # (Z, Y, X)

    # ── Slice volumes and pick in-plane vector components ─────────────────
    if dim == 'z':
        img_slice = img[slice_idx, :, :]
        msk_slice = msk[slice_idx, :, :]
        outline_slice = outline_mask[slice_idx, :, :]
        U_full    = Vx[slice_idx, :, :]
        V_full    = Vy[slice_idx, :, :]
        dot_slice = dot_3d[slice_idx, :, :] if COM_zyx is not None else None
    elif dim == 'y':
        img_slice = img[:, slice_idx, :]
        msk_slice = msk[:, slice_idx, :]
        outline_slice = outline_mask[:, slice_idx, :]
        U_full    = Vx[:, slice_idx, :]
        V_full    = Vz[:, slice_idx, :]
        dot_slice = dot_3d[:, slice_idx, :] if COM_zyx is not None else None
    else:
        img_slice = img[:, :, slice_idx]
        msk_slice = msk[:, :, slice_idx]
        outline_slice = outline_mask[:, :, slice_idx]
        U_full    = Vy[:, :, slice_idx]
        V_full    = Vz[:, :, slice_idx]
        dot_slice = dot_3d[:, :, slice_idx] if COM_zyx is not None else None

    plt.figure(figsize=figsize)
    plt.imshow(img_slice, cmap='gray', origin='lower')

    alpha_mask = np.zeros_like(msk_slice, dtype=float)
    alpha_mask[msk_slice != 0] = 0.5
    plt.imshow(msk_slice, cmap='Blues', alpha=alpha_mask, origin='lower')

    cols = np.arange(0, img_slice.shape[1], step)
    rows = np.arange(0, img_slice.shape[0], step)
    C, R = np.meshgrid(cols, rows)

    U   = U_full[::step, ::step]
    V   = V_full[::step, ::step]

    # Calculate magnitude just to filter out the tiny ones
    mag = np.sqrt(U**2 + V**2)
    keep = mag.ravel() >= min_magnitude

    # Flatten and filter the true unnormalized components!
    U_true = U.ravel()[keep]
    V_true = V.ravel()[keep]
    C_f, R_f = C.ravel()[keep], R.ravel()[keep]

    if COM_zyx is None:
        plt.quiver(C_f, R_f, U_true, V_true, color='red',
                   angles='xy', scale_units='xy', scale=scale)
    else:
        # Use the 3D dot product for coloring — not the 2D projection
        dot_sampled = dot_slice[::step, ::step].ravel()[keep]
        colors = np.where(dot_sampled > 0, '#00CC00', 'red')
        if hardcode_parameters:
            plt.quiver(C_f, R_f, U_true, V_true, color=colors,
            angles='xy', scale_units='xy', scale=scale, width=0.003,
            headwidth=3*0.9, headlength=5*0.9, headaxislength=4.5*0.9)
        else:
            plt.quiver(C_f, R_f, U_true, V_true, color=colors,
                   angles='xy', scale_units='xy', scale=scale)

    # Draw contour outline
    binary_outline = (outline_slice > 0).astype(float)
    contours = find_contours(binary_outline, level=0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color=outline_color,
                 linewidth=outline_width)
        
    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()





def plot_img_msk_outline(img, msk, outline_mask, slice_idx, dim='z',
                              figsize=(8, 8),title=None, output_path=None,  outline_width=2.0, outline_color = 'lime'):
    # The other function with almost the same name just projects the arrows in 2D. So in the 2D image they are in fact projectet such that they point away and thus red arrows. But in 3D, they point the right way. This function colors according the the real 3D value direction, not the 2D projection direction. So if the 3D vector points toward COM, it is green, if it points away, it is red, even if the 2D projection might look different.
    dim = dim.lower()
    if dim not in ('z', 'y', 'x'):
        raise ValueError(f"dim must be 'z', 'y', or 'x', got '{dim}'")

    if dim == 'z':
        img_slice = img[slice_idx, :, :]
        msk_slice = msk[slice_idx, :, :]
        outline_slice = outline_mask[slice_idx, :, :]
    elif dim == 'y':
        img_slice = img[:, slice_idx, :]
        msk_slice = msk[:, slice_idx, :]
        outline_slice = outline_mask[:, slice_idx, :]
    else:
        img_slice = img[:, :, slice_idx]
        msk_slice = msk[:, :, slice_idx]
        outline_slice = outline_mask[:, :, slice_idx]

    plt.figure(figsize=figsize)
    plt.imshow(img_slice, cmap='gray', origin='lower')

    alpha_mask = np.zeros_like(msk_slice, dtype=float)
    alpha_mask[msk_slice != 0] = 0.5
    plt.imshow(msk_slice, cmap='Blues', alpha=alpha_mask, origin='lower')

    # Draw contour outline
    binary_outline = (outline_slice > 0).astype(float)
    contours = find_contours(binary_outline, level=0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color=outline_color,
                 linewidth=outline_width)
        
    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_img_msk(img, msk, slice_idx, dim='z',
                              figsize=(8, 8),title=None, output_path=None):
    # The other function with almost the same name just projects the arrows in 2D. So in the 2D image they are in fact projectet such that they point away and thus red arrows. But in 3D, they point the right way. This function colors according the the real 3D value direction, not the 2D projection direction. So if the 3D vector points toward COM, it is green, if it points away, it is red, even if the 2D projection might look different.
    dim = dim.lower()
    if dim not in ('z', 'y', 'x'):
        raise ValueError(f"dim must be 'z', 'y', or 'x', got '{dim}'")

    if dim == 'z':
        img_slice = img[slice_idx, :, :]
        msk_slice = msk[slice_idx, :, :]
    elif dim == 'y':
        img_slice = img[:, slice_idx, :]
        msk_slice = msk[:, slice_idx, :]
    else:
        img_slice = img[:, :, slice_idx]
        msk_slice = msk[:, :, slice_idx]

    plt.figure(figsize=figsize)
    plt.imshow(img_slice, cmap='gray', origin='lower')

    alpha_mask = np.zeros_like(msk_slice, dtype=float)
    alpha_mask[msk_slice != 0] = 0.5
    plt.imshow(msk_slice, cmap='Blues', alpha=alpha_mask, origin='lower')

    # Draw contour outline
    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()





def plot_vector_field_magnitude_slice(Vx, Vy, Vz, dim='z', slice_idx=None, title="Vector Field Slice"):
    """
    Plots the components and magnitude of a 3D vector field at a specific slice.
    
    Args:
        Vx, Vy, Vz: 3D numpy arrays representing vector components.
        dim: The dimension along which to plot the slice ('z', 'y', or 'x').
        slice_idx: The index along the specified dimension to plot. Defaults to middle slice.
        title: Overall title for the figure.
    """
    # 1. Calculate the total magnitude
    magnitude = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    
    # 2. Default to middle slice if not provided
    if slice_idx is None:
        if dim == 'z':
            slice_idx = Vx.shape[0] // 2
        elif dim == 'y':
            slice_idx = Vx.shape[1] // 2
        else:  # 'x'
            slice_idx = Vx.shape[2] // 2


    Vx_slice = Vx[slice_idx, :, :] if dim == 'z' else (Vx[:, slice_idx, :] if dim == 'y' else Vx[:, :, slice_idx])
    Vy_slice = Vy[slice_idx, :, :] if dim == 'z' else (Vy[:, slice_idx, :] if dim == 'y' else Vy[:, :, slice_idx])
    Vz_slice = Vz[slice_idx, :, :] if dim == 'z' else (Vz[:, slice_idx, :] if dim == 'y' else Vz[:, :, slice_idx])
    magnitude_slice = magnitude[slice_idx, :, :] if dim == 'z' else (magnitude[:, slice_idx, :] if dim == 'y' else magnitude[:, :, slice_idx])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"{title} (Slice: {slice_idx})", fontsize=16)

    # Determine symmetric color limits for V-components so 0 is neutral
    v_max = max(np.abs(Vx_slice).max(), np.abs(Vy_slice).max(), np.abs(Vz_slice).max())
    # Use a small epsilon to avoid colorbar errors if everything is zero
    v_max = max(v_max, 1e-5) 

    # Plot Vx
    im0 = axes[0, 0].imshow(Vx_slice, cmap='RdBu_r', vmin=-v_max, vmax=v_max)
    axes[0, 0].set_title(r'$V_x$ (Horizontal Flow)')
    fig.colorbar(im0, ax=axes[0, 0])

    # Plot Vy
    im1 = axes[0, 1].imshow(Vy_slice, cmap='RdBu_r', vmin=-v_max, vmax=v_max)
    axes[0, 1].set_title(r'$V_y$ (Vertical Flow)')
    fig.colorbar(im1, ax=axes[0, 1])

    # Plot Vz
    im2 = axes[1, 0].imshow(Vz_slice, cmap='RdBu_r', vmin=-v_max, vmax=v_max)
    axes[1, 0].set_title(r'$V_z$ (Depth Flow)')
    fig.colorbar(im2, ax=axes[1, 0])

    # Plot Total Magnitude
    im3 = axes[1, 1].imshow(magnitude_slice, cmap='viridis')
    axes[1, 1].set_title('Total Vector Magnitude')
    fig.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])




def sample_vector_field_at_vertices(verts, vector_field, mask_sitk):
    # Get mask properties
    origin = torch.tensor(mask_sitk.GetOrigin(), device=verts.device, dtype=verts.dtype)
    spacing = torch.tensor(mask_sitk.GetSpacing(), device=verts.device, dtype=verts.dtype)
    size = torch.tensor(vector_field.shape[1:][::-1], device=verts.device, dtype=verts.dtype)

    # Calculate extreme physical coordinates
    min_coord = origin
    max_coord = origin + spacing * (size - 1)

    # Normalize to [-1, 1]
    normalized = 2.0 * (verts - min_coord) / (max_coord - min_coord) - 1.0

    # Reshape and sample
    grid = normalized.view(1, 1, 1, -1, 3) 
    vf_batched = vector_field.unsqueeze(0)
    sampled = F.grid_sample(vf_batched, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Return only the vectors (N, 3)
    vectors = sampled.squeeze().t() 

    return vectors



def vector_field_loss(mesh, vector_field, mask_sitk, mode='push'):
    """
    Compute loss based on vector field for vertices inside the masked region.

    Args:
        mesh: PyTorch3D Mesh object
        vector_field: (3, D, H, W) tensor
        mask_sitk: SimpleITK image for coordinate conversion
        mode: 'push' = encourage movement opposite to vector field (push out)
              'align' = encourage movement along vector field (pull toward COM)

    Returns:
        loss: scalar tensor
    """
    verts = mesh.verts_packed()  # (N, 3)

    # Sample vector field at vertex positions
    vectors, inside_mask = sample_vector_field_at_vertices(verts, vector_field, mask_sitk)

    if not inside_mask.any():
        return torch.tensor(0.0, device=verts.device, requires_grad=True)

    # Get vectors only for vertices inside the mask
    inside_vectors = vectors[inside_mask]  # (M, 3)

    # Loss: magnitude of vectors at inside vertices
    # Higher magnitude = deeper inside = higher penalty
    # This encourages vertices to move toward the boundary
    loss = torch.mean(torch.norm(inside_vectors, dim=1))

    return loss





def vector_field_loss_stable(mesh, vector_field, mask_sitk):
    verts = mesh.verts_packed()  # (N, 3)
    N_total = verts.shape[0]

    # Sample vector field directly (no mask returned)
    vectors = sample_vector_field_at_vertices(verts, vector_field, mask_sitk)

    # Calculate per-vertex squared L2 norm
    per_vertex_loss = torch.sum(vectors ** 2, dim=1)  # (N,)

    # Sum and divide by total vertices
    loss = per_vertex_loss.sum() / N_total

    return loss



def vector_field_loss_stable_directional(mesh, vector_field, mask_sitk):
    verts = mesh.verts_packed()  # (N, 3)

    # 1. Sample the vector field to get the direction and magnitude (the "arrows")
    vectors = sample_vector_field_at_vertices(verts, vector_field, mask_sitk)

    # 2. Define where the vertices *should* be
    # We add the vector to the current position: target = current + vector
    # CRITICAL: We use .detach() to treat these targets as fixed coordinates in space 
    # for this optimization step. This prevents PyTorch from calculating weird 
    # gradients through the grid_sample operation.
    target_verts = verts.detach() + vectors.detach()

    # 3. Compute the loss as the Mean Squared Error (MSE)
    # The optimizer will push 'verts' toward 'target_verts', naturally 
    # forcing them to slide along the direction of the vectors.
    loss = F.mse_loss(verts, target_verts)

    return loss



def taubin_smoothing(mesh, n_iters=10, lambda_pos=0.5, lambda_neg=-0.53):
    """
    Apply Taubin smoothing to a PyTorch3D mesh.
    Alternates positive (shrink) and negative (inflate) Laplacian steps
    to smooth the mesh without net shrinkage.

    Args:
        mesh: PyTorch3D Meshes object (single mesh).
        n_iters: Number of smoothing iterations (each = one shrink + one inflate).
        lambda_pos: Positive smoothing factor (shrink step). Typical: 0.3-0.7.
        lambda_neg: Negative smoothing factor (inflate step). Must satisfy:
                    |lambda_neg| > lambda_pos to avoid shrinkage.
                    Typical: -0.53 for lambda_pos=0.5.

    Returns:
        smoothed_verts: (N, 3) tensor of smoothed vertex positions.
    """
    from pytorch3d.structures import Meshes

    verts = mesh.verts_packed().clone()  # (N, 3)
    faces = mesh.faces_packed()          # (F, 3)
    N = verts.shape[0]

    # Build adjacency: for each vertex, collect its neighbors
    # Using faces to build edge list
    edges = set()
    for f in faces:
        for j in range(3):
            a, b = int(f[j]), int(f[(j + 1) % 3])
            edges.add((min(a, b), max(a, b)))

    # Build neighbor lists
    neighbors = [[] for _ in range(N)]
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)

    # Convert to sparse adjacency for fast matmul
    row_idx = []
    col_idx = []
    weights = []
    for i in range(N):
        nbrs = neighbors[i]
        if len(nbrs) == 0:
            continue
        w = 1.0 / len(nbrs)
        for nb in nbrs:
            row_idx.append(i)
            col_idx.append(nb)
            weights.append(w)

    indices = torch.tensor([row_idx, col_idx], dtype=torch.long, device=verts.device)
    values = torch.tensor(weights, dtype=verts.dtype, device=verts.device)
    L = torch.sparse_coo_tensor(indices, values, size=(N, N))

    for _ in range(n_iters):
        # Positive step (shrink)
        avg = torch.sparse.mm(L, verts)
        verts = verts + lambda_pos * (avg - verts)
        # Negative step (inflate)
        avg = torch.sparse.mm(L, verts)
        verts = verts + lambda_neg * (avg - verts)

    return verts


def project_gradient_along_vector_field(deform_verts, mesh, vector_field, mask_sitk):
    """
    Project out gradient components that would move vertices AGAINST the
    vector field direction.  Call after loss.backward() and before
    optimizer.step().

    For every vertex whose sampled vector field is non-zero, if the
    optimizer update (−grad) has a component opposing the vector field
    direction, that component is removed.  Regularization losses can
    still smooth the mesh tangentially or along the desired direction,
    but can never push a vertex backwards through the mask boundary.

    Args:
        deform_verts: Parameter tensor whose .grad has been populated.
        mesh: The current deformed PyTorch3D Mesh (new_src_mesh).
        vector_field: (3, D, H, W) tensor of desired directions.
        mask_sitk: SimpleITK image used for coordinate conversion.
    """
    if deform_verts.grad is None:
        return

    with torch.no_grad():
        verts = mesh.verts_packed()  # (N, 3)
        vectors, has_field = sample_vector_field_at_vertices(
            verts, vector_field, mask_sitk
        )

        if not has_field.any():
            return

        # Unit direction of the vector field at each relevant vertex
        vf_dirs = vectors[has_field]                          # (M, 3)
        vf_norms = torch.norm(vf_dirs, dim=1, keepdim=True).clamp(min=1e-8)
        vf_unit = vf_dirs / vf_norms                         # (M, 3)

        # Current gradient for those vertices
        g = deform_verts.grad.data[has_field]                 # (M, 3)

        # dot(grad, vf_unit):  positive means the update (−grad) would
        # push the vertex AGAINST the vector field → bad component
        dots = torch.sum(g * vf_unit, dim=1, keepdim=True)   # (M, 1)
        bad = (dots > 0).float()

        # Remove the opposing component
        g_projected = g - bad * dots * vf_unit
        deform_verts.grad.data[has_field] = g_projected


def create_vector_field_for_structure(msk_total_original, structure_id, COM_zyx, z, y, x):
    """
    Create a vector field pointing toward COM for a single structure.

    Args:
        msk_total_original: Original mask with all structure labels
        structure_id: The label value of the structure (e.g., 116)
        COM_zyx: Center of mass in (z, y, x) coordinates that the vector field should point toward
        z, y, x: Coordinate grids from meshgrid

    Returns:
        Vx, Vy, Vz: Vector field components (zero outside structure)
    """
    # Create binary mask for this structure
    structure_mask = (msk_total_original == structure_id)

    # Skip if structure not present
    if not structure_mask.any():
        return np.zeros_like(z), np.zeros_like(y), np.zeros_like(x)

    # Vector from each voxel toward COM
    Vz = COM_zyx[0] - z
    Vy = COM_zyx[1] - y
    Vx = COM_zyx[2] - x
    #Dvs. det her vil give meget tæt på 0 for alle 3:
    # Vz[int(COM_zyx[0]),int(COM_zyx[1]),int(COM_zyx[2])]
    # Vy[int(COM_zyx[0]),int(COM_zyx[1]),int(COM_zyx[2])]
    # Vx[int(COM_zyx[0]),int(COM_zyx[1]),int(COM_zyx[2])]

    # Apply mask (zero outside structure)
    Vz = Vz * structure_mask
    Vy = Vy * structure_mask
    Vx = Vx * structure_mask

    return Vx, Vy, Vz



def create_combined_vector_field(msk_total_original, COM_zyx):
    """
    Create a combined vector field for multiple structures, pointing toward COM.

    Args:
        msk_total_original: 3D numpy array (Z, Y, X) with structure labels
        structure_ids: List of structure IDs to create vector fields for (e.g., [116, 117])
        COM_zyx: Center of mass coordinates in (z, y, x) order

    Returns:
        Vx_total, Vy_total, Vz_total: Combined vector field components, scaled to [0, 1]
        msk_total: Binary mask (1 where ANY structure exists)
    """
    # Define coordinate grids
    z = np.arange(msk_total_original.shape[0])
    y = np.arange(msk_total_original.shape[1])
    x = np.arange(msk_total_original.shape[2])
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    # Initialize combined vector field
    Vx_total = np.zeros_like(X, dtype=float)
    Vy_total = np.zeros_like(Y, dtype=float)
    Vz_total = np.zeros_like(Z, dtype=float)

    structures_not_to_include = [0, 51, 52, 53, 61, 62, 63]  # example structure IDs to exclude

    structure_ids = np.arange(1, 118, 1)  # all structures except 51 and 61

    progress_bar = tqdm(structure_ids, total=len(structure_ids)-2)
    progress_bar.set_description(f'Creating combined vector field')

    # Loop through structures, scale each individually, then add
    for structure_id in progress_bar:
        if structure_id in structures_not_to_include:
            continue

        Vx, Vy, Vz = create_vector_field_for_structure(msk_total_original, structure_id, COM_zyx, Z, Y, X)

        # Desired scaling: smallest non-zero magnitude -> 0.01
        desired_min = 0.01
        max_cap = 3.0  # maximum allowed magnitude

        # Compute magnitude
        magnitude = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        nonzero_mask = magnitude > 1e-8

        if nonzero_mask.any():
            min_mag = magnitude[nonzero_mask].min()
            
            # Scale so that the smallest non-zero vector becomes desired_min
            scale_factor = np.zeros_like(magnitude)
            scale_factor[nonzero_mask] = (magnitude[nonzero_mask] / min_mag) * desired_min

            # Apply scaling
            Vx = Vx * scale_factor
            Vy = Vy * scale_factor
            Vz = Vz * scale_factor

            # Cap magnitudes at max_cap
            mag = np.sqrt(Vx**2 + Vy**2 + Vz**2)
            cap_mask = mag > max_cap
            Vx[cap_mask] = Vx[cap_mask] / mag[cap_mask] * max_cap
            Vy[cap_mask] = Vy[cap_mask] / mag[cap_mask] * max_cap
            Vz[cap_mask] = Vz[cap_mask] / mag[cap_mask] * max_cap

            # Optional: visualize
            plt.hist(Vz[nonzero_mask].flatten(), bins=50)

            # Add scaled vector field to total
            Vx_total += Vx
            Vy_total += Vy
            Vz_total += Vz

    # Create combined binary mask
    msk_total = np.isin(msk_total_original, structure_ids).astype(np.uint8)

    return Vx_total, Vy_total, Vz_total, msk_total



def create_outside_vector_field(mask_arr, COM_zyx, spacing):
    # ── 1. Signed distance field ──────────────────────────────────────────
    # Convention: negative inside, positive outside
    sdf = -edt.sdf(
        mask_arr,
        anisotropy=[spacing[2], spacing[1], spacing[0]],  
        parallel=8
    )

    # ── 2. Toward-COM unit vectors ────────────────────────────────────────
    Z, Y, X = np.ogrid[:mask_arr.shape[0], :mask_arr.shape[1], :mask_arr.shape[2]]
    dz = COM_zyx[0] - Z
    dy = COM_zyx[1] - Y
    dx = COM_zyx[2] - X
    
    dist_to_com = np.sqrt(dz**2 + dy**2 + dx**2) + 1e-8

    toward_com = np.stack([
        dz / dist_to_com, 
        dy / dist_to_com, 
        dx / dist_to_com
    ], axis=0)

    # ── 3. SDF gradient ───────────────────────────────────────────────────
    grad = np.stack(np.gradient(sdf), axis=0)
    grad_norm = np.sqrt(np.sum(grad**2, axis=0)) + 1e-8
    sdf_grad = grad / grad_norm  

    # ── 3.5 Fix direction: make gradient always point toward surface ──────
    # SDF gradient naturally points inside->outside. Flip it outside to point in.
    outside = sdf > 0
    sdf_grad[:, outside] *= -1

    # ── 4. Blend: Favoring "Toward-COM" ───────────────────────────────────
    dot = np.sum(sdf_grad * toward_com, axis=0)
    
    # Only blend inside the mask if diverging > 40 degrees
    alpha = np.where((dot < 0) & (sdf < 0), 1.0, 0.0)
    
    push_dir = (1 - alpha) * sdf_grad + alpha * toward_com
    push_dir /= (np.sqrt(np.sum(push_dir**2, axis=0, keepdims=True)) + 1e-8)

    # ── 5. Magnitude ──────────────────────────────────────────────────────
    sigma = 2.0
    weight = np.ones_like(sdf)
    inside = sdf < 0
    weight[inside] = 1.0 - np.exp(sdf[inside] / sigma)

    # ── 6. Kill (Stability Gate) ──────────────────────────────────────────
    alignment = np.sum(push_dir * toward_com, axis=0)
    soft_gate = np.maximum(0, alignment)**2 
    final_weight = weight * soft_gate
    
    # ── 7. Final Axis Mapping ─────────────────────────────────────────────
    Vz, Vy, Vx = push_dir * final_weight

    return Vx, Vy, Vz



def create_inside_vector_field(mask_arr, COM_zyx, spacing):
    # ── 1. Signed distance field ──────────────────────────────────────────
    sdf = -edt.sdf(
        mask_arr,
        anisotropy=[spacing[2], spacing[1], spacing[0]],
        parallel=8
    )

    # ── 2. AWAY-FROM-COM unit vectors ─────────────────────────────────────
    Z, Y, X = np.ogrid[:mask_arr.shape[0], :mask_arr.shape[1], :mask_arr.shape[2]]
    dz = Z - COM_zyx[0] 
    dy = Y - COM_zyx[1]
    dx = X - COM_zyx[2]
    
    dist_to_com = np.sqrt(dz**2 + dy**2 + dx**2) + 1e-8
    away_from_com = np.stack([
        dz / dist_to_com, 
        dy / dist_to_com, 
        dx / dist_to_com
    ], axis=0)

    # ── 3. SDF gradient ───────────────────────────────────────────────────
    grad = np.stack(np.gradient(sdf), axis=0)
    grad_norm = np.sqrt(np.sum(grad**2, axis=0)) + 1e-8
    sdf_grad = grad / grad_norm

    # ── 3.5 Fix direction: make gradient always point away from surface ───
    # SDF gradient naturally points inside->outside. Flip it outside to point in.
    outside = sdf > 0
    sdf_grad[:, outside] *= -1

    # ── 4. Blend: Favoring "Away-from-COM" ────────────────────────────────
    dot = np.sum(sdf_grad * away_from_com, axis=0)
    
    # Only blend inside the mask if diverging > 40 degrees
    alpha = np.where((dot < 0) & (sdf < 0), 1.0, 0.0)
    
    push_dir = (1 - alpha) * sdf_grad + alpha * away_from_com
    push_dir /= (np.sqrt(np.sum(push_dir**2, axis=0, keepdims=True)) + 1e-8)

    # ── 5. Magnitude ──────────────────────────────────────────────────────
    sigma = 2.0
    weight = np.ones_like(sdf)
    inside = sdf < 0
    weight[inside] = 1.0 - np.exp(sdf[inside] / sigma)


    # ── 6. Kill (Stability Gate) ──────────────────────────────────────────
    alignment = np.sum(push_dir * away_from_com, axis=0)
    soft_gate = np.maximum(0, alignment)**2 
    final_weight = weight * soft_gate

    # ── 7. Final Axis Mapping ─────────────────────────────────────────────
    Vz, Vy, Vx = push_dir * final_weight

    return Vx, Vy, Vz



def create_internal_external_vector_fields_fast(mask_arr, COM_zyx, spacing, region):

    # Initial check for valid type
    if region not in ("internal", "external"):
        raise ValueError(
            f"region must be one of ('both', 'internal', 'external'), got {region}"
        )
    # ── 1. SDF ────────────────────────────────────────────────────────────
    
    sdf = -edt.sdf(
        mask_arr,
        anisotropy=[spacing[2], spacing[1], spacing[0]],  
        parallel=32
    )


    # ── 2. Geometry ──────────────────────────────────────────────────────
    Z, Y, X = np.ogrid[:sdf.shape[0], :sdf.shape[1], :sdf.shape[2]]

    if region == 'internal':
        dz = (Z - COM_zyx[0]).astype(np.float32)
        dy = (Y - COM_zyx[1]).astype(np.float32)
        dx = (X - COM_zyx[2]).astype(np.float32)
    elif region == 'external':
        dz = (COM_zyx[0] - Z).astype(np.float32)
        dy = (COM_zyx[1] - Y).astype(np.float32)
        dx = (COM_zyx[2] - X).astype(np.float32)
    
    dist_to_com = np.sqrt(dz**2 + dy**2 + dx**2) + 1e-8

    dz_full = np.broadcast_to(dz, sdf.shape)
    dy_full = np.broadcast_to(dy, sdf.shape)
    dx_full = np.broadcast_to(dx, sdf.shape)

    Vz = np.empty_like(sdf, dtype=np.float32)
    Vy = np.empty_like(sdf, dtype=np.float32)
    Vx = np.empty_like(sdf, dtype=np.float32)

    inside = sdf <= 0
    outside = ~inside


    # ── 3. Inside ────────────────────────────────────────────────────────

    dist_in = dist_to_com[inside]

    sigma = 2.0
    factor_in = (1.0 - np.exp(sdf[inside] / sigma)) / dist_in

    Vz[inside] = dz_full[inside] * factor_in
    Vy[inside] = dy_full[inside] * factor_in
    Vx[inside] = dx_full[inside] * factor_in


    # ── 4. Outside ───────────────────────────────────────────────────────
    grad_z, grad_y, grad_x = np.gradient(sdf)

    gz_out = grad_z[outside]
    gy_out = grad_y[outside]
    gx_out = grad_x[outside]

    g_norm_out = np.sqrt(gz_out**2 + gy_out**2 + gx_out**2) + 1e-8

    push_z = -gz_out / g_norm_out
    push_y = -gy_out / g_norm_out
    push_x = -gx_out / g_norm_out

    dist_out = dist_to_com[outside]
    tc_z = dz_full[outside] / dist_out
    tc_y = dy_full[outside] / dist_out
    tc_x = dx_full[outside] / dist_out

    alignment = (push_z * tc_z) + (push_y * tc_y) + (push_x * tc_x)
    soft_gate = np.maximum(0, alignment)
    np.square(soft_gate, out=soft_gate)

    Vz[outside] = push_z * soft_gate
    Vy[outside] = push_y * soft_gate
    Vx[outside] = push_x * soft_gate

    return Vx, Vy, Vz



def _process_single_structure(args):
    """Worker function for parallel processing of a single structure."""
    msk_total_original, structure_id, COM_zyx, shape = args
    
    # Recreate coordinate grids inside worker (can't pickle meshgrids easily)
    z = np.arange(shape[0])
    y = np.arange(shape[1])
    x = np.arange(shape[2])
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    Vx, Vy, Vz = create_vector_field_for_structure(msk_total_original, structure_id, COM_zyx, Z, Y, X)

    desired_min = 0.01
    max_cap = 3.0

    magnitude = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    nonzero_mask = magnitude > 1e-8

    if not nonzero_mask.any():
        return None  # Skip empty structures

    min_mag = magnitude[nonzero_mask].min()
    scale_factor = np.zeros_like(magnitude)
    scale_factor[nonzero_mask] = (magnitude[nonzero_mask] / min_mag) * desired_min

    Vx = Vx * scale_factor
    Vy = Vy * scale_factor
    Vz = Vz * scale_factor

    mag = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    cap_mask = mag > max_cap
    if cap_mask.any():
        Vx[cap_mask] = Vx[cap_mask] / mag[cap_mask] * max_cap
        Vy[cap_mask] = Vy[cap_mask] / mag[cap_mask] * max_cap
        Vz[cap_mask] = Vz[cap_mask] / mag[cap_mask] * max_cap

    return Vx, Vy, Vz


def create_combined_vector_field_parallel(msk_total_original, COM_zyx, n_workers=None):
    """
    Create a combined vector field for multiple structures, pointing toward COM.
    Args:
        msk_total_original: 3D numpy array (Z, Y, X) with structure labels
        COM_zyx: Center of mass coordinates in (z, y, x) order
        n_workers: Number of parallel workers (defaults to CPU count)
    Returns:
        Vx_total, Vy_total, Vz_total: Combined vector field components
        msk_total: Binary mask (1 where ANY structure exists)
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    structures_not_to_include = {0, 51, 52, 53, 61, 62, 63}  # set for O(1) lookup
    structure_ids = [i for i in range(1, 118) if i not in structures_not_to_include]
    shape = msk_total_original.shape

    # Build args list for each worker
    args_list = [
        (msk_total_original, sid, COM_zyx, shape)
        for sid in structure_ids
    ]

    Vx_total = np.zeros(shape, dtype=float)
    Vy_total = np.zeros(shape, dtype=float)
    Vz_total = np.zeros(shape, dtype=float)

    vz_histograms = []  # collect for optional plotting after parallel work

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_single_structure, args): args[1] for args in args_list}

        with tqdm(total=len(futures), desc="Creating combined vector field") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    Vx, Vy, Vz = result
                    Vx_total += Vx
                    Vy_total += Vy
                    Vz_total += Vz
                    vz_histograms.append(Vz[Vz != 0].flatten())
                pbar.update(1)

    # Optional: plot combined histogram of all Vz non-zero values
    if vz_histograms:
        all_vz = np.concatenate(vz_histograms)
        plt.hist(all_vz, bins=50)
        plt.title("Combined Vz histogram")
        plt.show()

    msk_total = np.isin(msk_total_original, structure_ids).astype(np.uint8)
    msk_total[msk_total_original == 0] = 0
    
    return Vx_total, Vy_total, Vz_total, msk_total





def create_combined_vector_field_fast(msk_total_original, COM_zyx, desired_min=0.01, max_cap=3.0, constant_scaling=False):
    """
    Fast vectorized creation of combined vector field.
    Processes all structures in a single pass without loops.
    
    Args:
        msk_total_original: 3D numpy array (Z, Y, X) with structure labels
        COM_zyx: Center of mass coordinates in (z, y, x) order
        desired_min: Minimum magnitude after scaling (default: 0.01)
        max_cap: Maximum magnitude cap (default: 3.0)
        constant_scaling: If True, normalize all vectors to unit length (default: False)
    
    Returns:
        Vx_total, Vy_total, Vz_total: Combined vector field components
        msk_total: Binary mask (1 where ANY structure exists)
    """
    structures_not_to_include = {0, 51, 52, 53, 61, 62, 63}  # 0 = background
    
    # Create coordinate grids ONCE
    z = np.arange(msk_total_original.shape[0])
    y = np.arange(msk_total_original.shape[1])
    x = np.arange(msk_total_original.shape[2])
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    # Compute raw vectors for entire volume (toward COM)
    Vz_raw = COM_zyx[0] - Z
    Vy_raw = COM_zyx[1] - Y
    Vx_raw = COM_zyx[2] - X
    
    # Initialize output
    Vx_total = np.zeros_like(X, dtype=np.float32)
    Vy_total = np.zeros_like(Y, dtype=np.float32)
    Vz_total = np.zeros_like(Z, dtype=np.float32)
    
    # Get unique structure IDs (excluding background and excluded structures)
    unique_ids = np.unique(msk_total_original)
    unique_ids = [i for i in unique_ids if i not in structures_not_to_include]
    
    # Process all structures - but only compute scaling factors per structure
    for structure_id in tqdm(unique_ids, desc="Processing structures"):
        # Get mask for this structure
        structure_mask = msk_total_original == structure_id
        
        if not structure_mask.any():
            continue
        
        # Extract only the values where structure exists (much faster!)
        vx_struct = Vx_raw[structure_mask]
        vy_struct = Vy_raw[structure_mask]
        vz_struct = Vz_raw[structure_mask]
        
        # Compute magnitude for this structure's voxels only
        magnitude = np.sqrt(vx_struct**2 + vy_struct**2 + vz_struct**2)
        
        if constant_scaling:
            # Normalize to unit vectors
            nonzero_mask = magnitude > 1e-8
            if nonzero_mask.any():
                vx_struct[nonzero_mask] /= magnitude[nonzero_mask]
                vy_struct[nonzero_mask] /= magnitude[nonzero_mask]
                vz_struct[nonzero_mask] /= magnitude[nonzero_mask]
        else:
            # Scale: min -> desired_min, proportional scaling
            min_mag = magnitude.min()
            if min_mag > 1e-8:
                scale = (magnitude / min_mag) * desired_min
                
                vx_struct = vx_struct * scale
                vy_struct = vy_struct * scale
                vz_struct = vz_struct * scale
                
                # Cap at max
                mag_scaled = np.sqrt(vx_struct**2 + vy_struct**2 + vz_struct**2)
                cap_mask = mag_scaled > max_cap
                if cap_mask.any():
                    vx_struct[cap_mask] *= max_cap / mag_scaled[cap_mask]
                    vy_struct[cap_mask] *= max_cap / mag_scaled[cap_mask]
                    vz_struct[cap_mask] *= max_cap / mag_scaled[cap_mask]
        
        # Write back to output arrays
        Vx_total[structure_mask] = vx_struct
        Vy_total[structure_mask] = vy_struct
        Vz_total[structure_mask] = vz_struct
    
    # Create combined binary mask
    msk_total = (~np.isin(msk_total_original, list(structures_not_to_include))).astype(np.uint8)
    msk_total[msk_total_original == 0] = 0
    
    return Vx_total, Vy_total, Vz_total, msk_total


def create_combined_vector_field_fast_away(msk_total_original, COM_zyx, desired_min=0.01, max_cap=3.0, constant_scaling=False):
    """
    Fast vectorized creation of combined vector field pointing AWAY from COM.
    Processes all structures in a single pass without loops.
    
    Args:
        msk_total_original: 3D numpy array (Z, Y, X) with structure labels
        COM_zyx: Center of mass coordinates in (z, y, x) order
        desired_min: Minimum magnitude after scaling (default: 0.01)
        max_cap: Maximum magnitude cap (default: 3.0)
        constant_scaling: If True, normalize all vectors to unit length (default: False)
    
    Returns:
        Vx_total, Vy_total, Vz_total: Combined vector field components
        msk_total: Binary mask (1 where ANY structure exists)
    """
    structures_not_to_include = {0,6,7}  # 0 = background
    
    # Create coordinate grids ONCE
    z = np.arange(msk_total_original.shape[0])
    y = np.arange(msk_total_original.shape[1])
    x = np.arange(msk_total_original.shape[2])
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    # Compute raw vectors for entire volume (AWAY from COM)
    Vz_raw = Z - COM_zyx[0]
    Vy_raw = Y - COM_zyx[1]
    Vx_raw = X - COM_zyx[2]
    
    # Initialize output
    Vx_total = np.zeros_like(X, dtype=np.float32)
    Vy_total = np.zeros_like(Y, dtype=np.float32)
    Vz_total = np.zeros_like(Z, dtype=np.float32)
    
    # Get unique structure IDs (excluding background and excluded structures)
    unique_ids = np.unique(msk_total_original)
    unique_ids = [i for i in unique_ids if i not in structures_not_to_include]
    
    # Process all structures - but only compute scaling factors per structure
    for structure_id in tqdm(unique_ids, desc="Processing structures (away from COM)"):
        # Get mask for this structure
        structure_mask = msk_total_original == structure_id
        
        if not structure_mask.any():
            continue
        
        # Extract only the values where structure exists (much faster!)
        vx_struct = Vx_raw[structure_mask]
        vy_struct = Vy_raw[structure_mask]
        vz_struct = Vz_raw[structure_mask]
        
        # Compute magnitude for this structure's voxels only
        magnitude = np.sqrt(vx_struct**2 + vy_struct**2 + vz_struct**2)
        
        if constant_scaling:
            # Normalize to unit vectors
            nonzero_mask = magnitude > 1e-8
            if nonzero_mask.any():
                vx_struct[nonzero_mask] /= magnitude[nonzero_mask]
                vy_struct[nonzero_mask] /= magnitude[nonzero_mask]
                vz_struct[nonzero_mask] /= magnitude[nonzero_mask]
        else:
            # Scale: INVERSE - closer to COM (smaller magnitude) gets LARGER vectors
            max_mag = magnitude.max()
            if max_mag > 1e-8:
                scale = (max_mag / magnitude) * desired_min
                
                vx_struct = vx_struct * scale
                vy_struct = vy_struct * scale
                vz_struct = vz_struct * scale
                
                # Cap at max
                mag_scaled = np.sqrt(vx_struct**2 + vy_struct**2 + vz_struct**2)
                cap_mask = mag_scaled > max_cap
                if cap_mask.any():
                    vx_struct[cap_mask] *= max_cap / mag_scaled[cap_mask]
                    vy_struct[cap_mask] *= max_cap / mag_scaled[cap_mask]
                    vz_struct[cap_mask] *= max_cap / mag_scaled[cap_mask]
        
        # Write back to output arrays
        Vx_total[structure_mask] = vx_struct
        Vy_total[structure_mask] = vy_struct
        Vz_total[structure_mask] = vz_struct
    
    # Create combined binary mask
    msk_total_bin = (~np.isin(msk_total_original, list(structures_not_to_include))).astype(np.uint8)
    msk_total_bin[msk_total_original == 0] = 0
    
    return Vx_total, Vy_total, Vz_total, msk_total_bin

def add_fallback_vector_field_to_COM(vector_field, mask, COM_zyx, fallback_strength=0.1):
    """
    For voxels outside the mask (mask == 0), set the vector field to a small
    constant-magnitude vector pointing toward the center of mass (COM).
    This ensures escaped vertices always receive a gradient pushing them back.

    Args:
        vector_field: np.array of shape (3, Z, Y, X) — (Vx, Vy, Vz)
        mask: np.array of shape (Z, Y, X), nonzero = inside mask
        COM_zyx: tuple (z, y, x) center of mass in index space
        fallback_strength: magnitude of the fallback vectors (constant)

    Returns:
        modified vector_field (in-place)
    """
    outside = (mask == 0)

    Z, Y, X = mask.shape
    zz, yy, xx = np.meshgrid(
        np.arange(Z), np.arange(Y), np.arange(X), indexing='ij'
    )

    # Direction from each voxel toward COM (in z, y, x)
    dz = COM_zyx[0] - zz
    dy = COM_zyx[1] - yy
    dx = COM_zyx[2] - xx

    # Normalize to unit vectors
    dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
    dx_norm = dx / dist
    dy_norm = dy / dist
    dz_norm = dz / dist

    # Apply only outside the mask
    # vector_field is (3, Z, Y, X) with order (Vx, Vy, Vz)
    vector_field[0][outside] = fallback_strength * dx_norm[outside]
    vector_field[1][outside] = fallback_strength * dy_norm[outside]
    vector_field[2][outside] = fallback_strength * dz_norm[outside]

    return vector_field


def total_point_movement(deform_verts):
    return torch.norm(deform_verts, dim=1).sum().item()






# ═══════════════════════════════════════════════════════════════════════
#                        METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════

# EAT HU range
EAT_LOW, EAT_HIGH = -190, 0


def _surface_points_from_mask(binary_mask, spacing):
    """Extract surface voxel coordinates (in mm) from a binary mask."""
    eroded = binary_erosion(binary_mask, iterations=1)
    boundary = binary_mask & ~eroded
    coords_zyx = np.argwhere(boundary)  # (N, 3) in z,y,x voxel order
    # Convert to physical mm (z,y,x order) — spacing is (x,y,z)
    coords_mm = coords_zyx.astype(np.float64) * np.array([spacing[2], spacing[1], spacing[0]])
    return coords_mm


def compute_surface_distances(gt_mask, pred_mask, spacing, nsd_threshold_mm=2.0):
    """Compute ASD, ASSD, HD, HD95 between two binary masks."""
    pts_gt = _surface_points_from_mask(gt_mask, spacing)
    pts_pred = _surface_points_from_mask(pred_mask, spacing)

    if len(pts_gt) == 0 or len(pts_pred) == 0:
        return {"ASD": np.nan, "ASSD": np.nan, "HD": np.nan, "HD95": np.nan}

    tree_gt = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)

    dist_pred_to_gt, _ = tree_gt.query(pts_pred)
    dist_gt_to_pred, _ = tree_pred.query(pts_gt)

    asd = float(np.mean(dist_pred_to_gt))
    assd = float(0.5 * (np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)))
    hd = float(max(np.max(dist_pred_to_gt), np.max(dist_gt_to_pred)))
    hd95 = float(max(np.percentile(dist_pred_to_gt, 95), np.percentile(dist_gt_to_pred, 95)))
    nsd  = float(0.5 * (np.mean(dist_pred_to_gt <= nsd_threshold_mm) + np.mean(dist_gt_to_pred <= nsd_threshold_mm)))


    return {"ASD": asd, "ASSD": assd, "HD": hd, "HD95": hd95, "NSD": nsd}


def compute_dice(gt, pred):
    """Standard Dice between two binary arrays."""
    intersection = np.sum(gt & pred)
    return float(2.0 * intersection / (np.sum(gt) + np.sum(pred) + 1e-8))


def compute_eat_dice(gt_mask, pred_mask, img_array):
    """Dice for EAT voxels only (voxels inside the mask with HU in [EAT_LOW, EAT_HIGH])."""
    eat_range = (img_array >= EAT_LOW) & (img_array <= EAT_HIGH)
    gt_eat = gt_mask & eat_range
    pred_eat = pred_mask & eat_range
    return compute_dice(gt_eat, pred_eat)


def obj_to_vtk_polydata(obj_path):
    """Read an OBJ mesh with VTK and return vtkPolyData."""
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_path)
    reader.Update()
    return reader.GetOutput()


def get_z_cutoff_for_segment(mask_path, segment_id):
    """Find the highest z-coordinate (in array indices) for a given segment."""
    msk_array_sitk = sitk.ReadImage(mask_path)
    msk_array = sitk.GetArrayFromImage(msk_array_sitk)

    z_indices = np.where(msk_array == segment_id)[0]
    if len(z_indices) == 0:
        return None
    return np.max(z_indices)


def compute_all_metrics(obj_path, gt_sitk, img_sitk, z_cutoff=None):
    """
    Given a predicted mesh (OBJ), ground truth sitk image, and CT sitk image,
    compute Dice, EAT_Dice, HD, HD95, ASD, ASSD.
    Returns a dict of metrics.
    """
    gt = sitk.GetArrayFromImage(gt_sitk).astype(bool)
    img = sitk.GetArrayFromImage(img_sitk)
    spacing = gt_sitk.GetSpacing()  # (x, y, z)

    # Voxelise predicted mesh
    polydata = obj_to_vtk_polydata(obj_path)
    pred_sitk = tools.voxelize_mesh_to_sitk_image(polydata, gt_sitk)
    pred = sitk.GetArrayFromImage(pred_sitk).astype(bool)

    # --- Cut off at superior boundary of segment 2 ---
    if z_cutoff is not None:
        gt[z_cutoff+1:] = 0
        pred[z_cutoff+1:] = 0

    # Debug save
    # img[z_cutoff+1:] = -2048 # not necessary to save cut img. Only done for debug
    # gt_save = gt.astype(np.uint8)
    # pred_save = pred.astype(np.uint8)
    # gt_sitk_save = sitk.GetImageFromArray(gt_save)
    # gt_sitk_save.CopyInformation(gt_sitk)
    # pred_sitk_save = sitk.GetImageFromArray(pred_save)
    # pred_sitk_save.CopyInformation(gt_sitk)
    # outputfolder = "/data/awias/periseg/debug"
    # sitk.WriteImage(gt_sitk_save, os.path.join(outputfolder, "gt_cut.nii.gz"))
    # sitk.WriteImage(pred_sitk_save, os.path.join(outputfolder, "pred_cut.nii.gz"))
    # img_sitk_save = sitk.GetImageFromArray(img.astype(np.float32))
    # img_sitk_save.CopyInformation(gt_sitk)
    # sitk.WriteImage(img_sitk_save, os.path.join(outputfolder, "img_cut.nii.gz"))
    # print("Debug saved cut GT, pred, and img")

    # Surface distances
    sd = compute_surface_distances(gt, pred, spacing)

    # Dice
    dice = compute_dice(gt, pred)

    # EAT Dice
    eat_dice = compute_eat_dice(gt, pred, img)

    return {"Dice": dice, "EAT_Dice": eat_dice, **sd}


def get_annotated_slices(gt_mask):
    """Return z-indices where gt has any annotation."""
    return np.where(gt_mask.any(axis=(1, 2)))[0]



# def compute_all_metrics_saros(obj_path, gt_sitk, img_sitk):
#     """
#     Given a predicted mesh (OBJ), ground truth sitk image, and CT sitk image,
#     compute Dice, EAT_Dice, HD, HD95, ASD, ASSD.
#     Returns a dict of metrics.
#     """
#     gt = sitk.GetArrayFromImage(gt_sitk).astype(bool)
#     img = sitk.GetArrayFromImage(img_sitk)
#     spacing = gt_sitk.GetSpacing()  # (x, y, z)

#     # Voxelise predicted mesh
#     polydata = obj_to_vtk_polydata(obj_path)
#     pred_sitk = tools.voxelize_mesh_to_sitk_image(polydata, gt_sitk)
#     pred = sitk.GetArrayFromImage(pred_sitk).astype(bool)

#     z_slices = get_annotated_slices(gt)


#     # Surface distances
#     sd = compute_surface_distances(gt, pred, spacing)

#     # Dice
#     dice = compute_dice(gt, pred)

#     # EAT Dice
#     eat_dice = compute_eat_dice(gt, pred, img)

#     return {"Dice": dice, "EAT_Dice": eat_dice, **sd}


def compute_surface_distances_2d(gt_2d, pred_2d, spacing_2d, nsd_threshold_mm=2.0):
    """Same as compute_surface_distances but for 2D slices."""
    # Expand to 3D with a single z-slice so _surface_points_from_mask still works,
    # or just reimplement with 2D spacing
    pts_gt   = _surface_points_from_mask(gt_2d[np.newaxis],   (*spacing_2d, 1.0))
    pts_pred = _surface_points_from_mask(pred_2d[np.newaxis], (*spacing_2d, 1.0))

    if len(pts_gt) == 0 or len(pts_pred) == 0:
        return {"ASD": np.nan, "ASSD": np.nan, "HD": np.nan, "HD95": np.nan, "NSD": np.nan}

    tree_gt   = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)
    dist_pred_to_gt, _ = tree_gt.query(pts_pred)
    dist_gt_to_pred, _ = tree_pred.query(pts_gt)

    return {
        "ASD":  float(np.mean(dist_pred_to_gt)),
        "ASSD": float(0.5 * (np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred))),
        "HD":   float(max(np.max(dist_pred_to_gt), np.max(dist_gt_to_pred))),
        "HD95": float(max(np.percentile(dist_pred_to_gt, 95), np.percentile(dist_gt_to_pred, 95))),
        "NSD":  float(0.5 * (np.mean(dist_pred_to_gt <= nsd_threshold_mm) + np.mean(dist_gt_to_pred <= nsd_threshold_mm))),
    }


def compute_all_metrics_saros(obj_path, gt_sitk, img_sitk, z_cutoff=None):
    gt = sitk.GetArrayFromImage(gt_sitk).astype(bool)
    img = sitk.GetArrayFromImage(img_sitk)
    spacing = gt_sitk.GetSpacing()  # (x, y, z)

    polydata = obj_to_vtk_polydata(obj_path)
    pred_sitk = tools.voxelize_mesh_to_sitk_image(polydata, gt_sitk)
    pred = sitk.GetArrayFromImage(pred_sitk).astype(bool)

    if z_cutoff is not None:
        gt[z_cutoff+1:] = 0
        pred[z_cutoff+1:] = 0

    z_slices = get_annotated_slices(gt)
    if len(z_slices) == 0:
        return {"Dice": np.nan, "EAT_Dice": np.nan,
                "ASD": np.nan, "ASSD": np.nan, "HD": np.nan, "HD95": np.nan, "NSD": np.nan}

    slice_metrics = []
    for z in z_slices:
        gt_2d   = gt[z]    # (y, x)
        pred_2d = pred[z]
        img_2d  = img[z]

        # Skip if gt slice is empty (shouldn't happen but safe)
        if gt_2d.sum() == 0:
            continue

        if pred_2d.sum() == 0: # Pred has shrunk out of this slice. Shouldnt happen at the top, since we have z_cutoff, but can happen at the bottom. In that case, we will just skip it for now.
            continue

        # Use 2D spacing (x, y) only
        spacing_2d = spacing[:2]

        sd       = compute_surface_distances_2d(gt_2d, pred_2d, spacing_2d)
        dice     = compute_dice(gt_2d, pred_2d)
        eat_dice = compute_eat_dice(gt_2d, pred_2d, img_2d)

        slice_metrics.append({"Dice": dice, "EAT_Dice": eat_dice, **sd})

    # Aggregate across slices (mean)
    keys = slice_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in slice_metrics])) for k in keys}



def concatenate_masks(*masks, names=None):
    """
    Concatenate multiple label masks into one with globally unique label IDs.

    Background (0) in every mask stays 0.  Non-zero labels are remapped to
    unique sequential IDs across all masks.  Masks are applied in order, so
    later masks overwrite earlier ones where they overlap.

    Args:
        *masks:  Variable number of 3D numpy arrays (Z, Y, X) — label maps.
        names:   Optional list of name strings (used in the printed mapping).

    Returns:
        concatenated: (Z, Y, X) int32 array with unique labels.
        mapping:       dict  {(mask_name, original_label): new_label}
    """
    if names is None:
        names = [f'mask_{i}' for i in range(len(masks))]

    concatenated = np.zeros_like(masks[0], dtype=np.int32)
    mapping = {}
    next_id = 1

    for idx, msk in enumerate(masks):
        unique_labels = np.unique(msk)
        unique_labels = unique_labels[unique_labels != 0]  # skip background

        for orig_label in sorted(unique_labels):
            new_label = next_id
            mapping[(names[idx], int(orig_label))] = new_label
            concatenated[msk == orig_label] = new_label
            next_id += 1

    # # Print mapping
    # print("Label mapping:")
    # for (name, orig), new in mapping.items():
    #     print(f"  {name} label {orig}  ->  concatenated label {new}")

    return concatenated, mapping


def count_area_overlaps(obj_path, msk_highres_bin, msk_total_bin,  reference_sitk):
        """
        Count overlapping and non-overlapping areas between masks.

        highres = crossover (forbidden area) - how many voxels of the highres mask are outside the predicted mesh?
        total = overlap (forbidden area) - how many voxels of the total mask overlaps with the predicted mesh?

        Inputs:
            - obj_path: Path to the predicted mesh OBJ file.
            - msk_highres_bin: Binary numpy array (Z, Y, X) of the high-resolution mask (1 = inside, 0 = outside).
            - msk_total_bin: Binary numpy array (Z, Y, X) of the total mask (1 = inside, 0 = outside).
            - reference_sitk: SimpleITK image used for NON-rescaled voxel spacing. HAS to be an original sitk file.

            Outputs:
            - forbidden_region_highres: Volume (in mm³) of highres mask voxels outside the predicted mesh.
            - forbidden_region_total: Volume (in mm³) of total mask voxels overlapping with the predicted mesh.
        """


        volume_spacing = np.prod(reference_sitk.GetSpacing())
        # --- Volume overlap calculation before refinement ---
        polydata = obj_to_vtk_polydata(obj_path)
        polydata_voxelized_sitk = tools.voxelize_mesh_to_sitk_image(polydata, reference_sitk)
        polydata_voxelized = sitk.GetArrayFromImage(polydata_voxelized_sitk).astype(bool)
        # Highres crossover (how large area is NOT covered by the pericardial mesh?
        outside_voxels = np.logical_and(msk_highres_bin, ~polydata_voxelized).sum() # Number of outside voxels
        outside_voxels_volume = outside_voxels * volume_spacing
        # Total crossover - how large is the overlap with the total mask?
        overlap_voxels = np.logical_and(polydata_voxelized, msk_total_bin).sum() # Number of overlap voxels
        overlap_voxels_volume = overlap_voxels * volume_spacing

        # Easier names
        forbidden_region_highres = outside_voxels_volume
        forbidden_region_total = overlap_voxels_volume

        return forbidden_region_highres, forbidden_region_total
        




# ═══════════════════════════════════════════════════════════════════════
#                        CSV HELPERS
# ═══════════════════════════════════════════════════════════════════════
def load_completed_series(csv_path):
    """Return a set of series names already saved to the CSV."""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return set(df["series"].tolist())
        except Exception:
            return set()
    return set()


def append_row_to_csv(row, csv_path):
    """Append a single result row to the CSV, creating it with a header if needed."""
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=write_header, index=False)






def create_directional_sdf_field(mask_arr, COM_zyx, spacing, sigma_outside=5.0):
    # OLD!!!!!
    import edt

    # ── 1. Signed distance field ──────────────────────────────────────────
    # edt.sdf returns positive inside, negative outside — we flip it
    # so convention is: negative inside, positive outside
    sdf = -edt.sdf(
        mask_arr,
        anisotropy=[spacing[2], spacing[1], spacing[0]],  # X, Y, Z spacing
        parallel=8
    )

    # ── 2. Toward-COM unit vectors ────────────────────────────────────────
    # For every voxel, compute a unit vector pointing toward the COM
    Z, Y, X = np.ogrid[:mask_arr.shape[0], :mask_arr.shape[1], :mask_arr.shape[2]]
    dz = COM_zyx[0] - Z
    dy = COM_zyx[1] - Y
    dx = COM_zyx[2] - X
    dist_to_com = np.sqrt(dz**2 + dy**2 + dx**2) + 1e-8
    toward_com = np.stack([
        np.broadcast_to(dz / dist_to_com, mask_arr.shape),
        np.broadcast_to(dy / dist_to_com, mask_arr.shape),
        np.broadcast_to(dx / dist_to_com, mask_arr.shape),
    ], axis=0)  # (3, Z, Y, X)

    # ── 3. SDF gradient ───────────────────────────────────────────────────
    # The gradient of the SDF points toward the nearest surface point.
    # Inside: points outward (toward exit). Outside: points away from surface.
    # We normalize to get pure direction, magnitude comes from the SDF itself.
    gz = np.gradient(sdf, axis=0)
    gy = np.gradient(sdf, axis=1)
    gx = np.gradient(sdf, axis=2)
    grad_norm = np.sqrt(gz**2 + gy**2 + gx**2) + 1e-8
    sdf_grad = np.stack([gz, gy, gx], axis=0) / grad_norm  # (3, Z, Y, X)

    # ── 3.5 Fix direction: make gradient always point toward surface ─────────
    outside = sdf > 0
    sdf_grad[:, outside] *= -1

    # Blend with an angle
    # Option 2
    cos30 = np.cos(np.deg2rad(40))  # ≈ 0.866
    dot = np.sum(sdf_grad * toward_com, axis=0)
    # Any vector more than 30° away from toward_com gets fully replaced
    alpha = np.where(dot < cos30, 1.0, 0.0)
    # Only blend inside the mask, outside keep pure SDF gradient
    inside_mask = sdf < 0
    alpha = np.where(inside_mask, alpha, 0.0)
    push_dir = (1 - alpha) * sdf_grad + alpha * toward_com
    push_dir = push_dir / (np.sqrt(np.sum(push_dir**2, axis=0, keepdims=True)) + 1e-8)

    # ── 5. Magnitude ──────────────────────────────────────────────────────
    # Inside (sdf < 0): ramp from 0 at surface to 1 at 10mm depth
    # Outside within sigma_outside mm: weak exponential warning zone
    # Outside beyond sigma_outside mm: zero — field is silent
    inside_weight  = np.where(sdf < 0, np.clip(-sdf / 10.0, 0, 1), 0.0)
    outside_weight = np.where(sdf >= 0, np.exp(-sdf / sigma_outside),  0.0)
    weight = inside_weight + outside_weight * 0.2

    Vx = push_dir[2] * weight
    Vy = push_dir[1] * weight
    Vz = push_dir[0] * weight

    # Kill
    # Instead of: away_mask = np.sum(push_dir * toward_com, axis=0) < 0
    # Use a smooth dampening:
    alignment = np.sum(push_dir * toward_com, axis=0)

    # This multiplier is 1 if perfectly aligned, and 0 if perpendicular or away.
    # We use np.maximum(0, alignment)**2 to make the "cutoff" at 90 degrees very soft.
    soft_gate = np.maximum(0, alignment)**2 

    Vx *= soft_gate
    Vy *= soft_gate
    Vz *= soft_gate

    from scipy.ndimage import gaussian_filter

    # Smooth the final vector field components. 
    # A sigma of 1.0 to 2.0 voxels is usually enough to remove grid artifacts 
    # without destroying the overall structural flow.
    # smooth_sigma = 2.0 
    
    # Vx = gaussian_filter(Vx, sigma=smooth_sigma)
    # Vy = gaussian_filter(Vy, sigma=smooth_sigma)
    # Vz = gaussian_filter(Vz, sigma=smooth_sigma)

    return Vx, Vy, Vz



def create_repulsive_sdf_field(mask_arr, COM_zyx, spacing, sigma_outside=5.0):
    # OLD!!!!!
    import edt

    # ── 1. Signed distance field ──────────────────────────────────────────
    sdf = -edt.sdf(
        mask_arr,
        anisotropy=[spacing[2], spacing[1], spacing[0]],
        parallel=8
    )

    # ── 2. AWAY-FROM-COM unit vectors ─────────────────────────────────────
    # We flip the subtraction order: Voxel - COM = Vector pointing OUT
    Z, Y, X = np.ogrid[:mask_arr.shape[0], :mask_arr.shape[1], :mask_arr.shape[2]]
    dz = Z - COM_zyx[0] 
    dy = Y - COM_zyx[1]
    dx = X - COM_zyx[2]
    
    dist_to_com = np.sqrt(dz**2 + dy**2 + dx**2) + 1e-8
    away_from_com = np.stack([
        dz / dist_to_com,
        dy / dist_to_com,
        dx / dist_to_com,
    ], axis=0)

    # ── 3. SDF gradient ───────────────────────────────────────────────────
    # sdf_grad naturally points TOWARD exit (away from center) when inside.
    gz = np.gradient(sdf, axis=0)
    gy = np.gradient(sdf, axis=1)
    gx = np.gradient(sdf, axis=2)
    grad_norm = np.sqrt(gz**2 + gy**2 + gx**2) + 1e-8
    sdf_grad = np.stack([gz, gy, gx], axis=0) / grad_norm



    # ── 3.5 Fix direction: make gradient always point away from object center ──
    # Outside: make it point away from the surface into empty space
    outside = sdf > 0
    sdf_grad[:, outside] *= -1 # Keep existing direction (already points out)

    # ── 4. Blend: Favoring "Away-from-COM" ────────────────────────────────
    cos40 = np.cos(np.deg2rad(40))
    # Check alignment between SDF exit direction and Away-from-COM direction
    dot = np.sum(sdf_grad * away_from_com, axis=0)
    
    # If the surface normal points "inward" (rare but possible in complex shapes), 
    # replace it with the pure Away-from-COM vector.
    alpha = np.where(dot < cos40, 1.0, 0.0)
    inside_mask = sdf < 0
    alpha = np.where(inside_mask, alpha, 0.0)
    
    push_dir = (1 - alpha) * sdf_grad + alpha * away_from_com
    push_dir = push_dir / (np.sqrt(np.sum(push_dir**2, axis=0, keepdims=True)) + 1e-8)

    # ── 5. Magnitude ──────────────────────────────────────────────────────
    inside_weight  = np.where(sdf < 0, np.clip(-sdf / 10.0, 0, 1), 0.0)
    outside_weight = np.where(sdf >= 0, np.exp(-sdf / sigma_outside),  0.0)
    weight = inside_weight + outside_weight * 0.2

    # ── 6. Kill (Stability Gate) ──────────────────────────────────────────
    # Now we only allow vectors that point AWAY from the COM.
    alignment = np.sum(push_dir * away_from_com, axis=0)

    # Multiplier is 1 if pointing away, 0 if pointing toward center.
    soft_gate = np.maximum(0, alignment)**2 

    Vx = push_dir[2] * weight * soft_gate
    Vy = push_dir[1] * weight * soft_gate
    Vz = push_dir[0] * weight * soft_gate

    return Vx, Vy, Vz

