import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from matplotlib.patches import Patch
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def get_direction_code(img_sitk):
    """
    Get direction code from SimpleITK image.
    
    Args:
        img_sitk (SimpleITK.Image): Input image.
        
    Returns:
        str: Direction code as a string.
    """
    direction_code = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(img_sitk.GetDirection())
    return direction_code

def reorient_sitk(img_sitk, new_direction):
    """ Reorient a SimpleITK image to a new direction.
    Args:
        img_sitk (SimpleITK.Image): Input image to be reoriented.
        new_direction (str): New direction code (e.g., 'LPS', 'RAS').
    Returns:
        SimpleITK.Image: Reoriented image.
    """

    img_sitk_reoriented = sitk.DICOMOrient(img_sitk, new_direction)

    return img_sitk_reoriented

def get_direction_code(img_sitk):
    """ Get the direction of a SimpleITK image.
    Args:
        img_sitk (SimpleITK.Image): Input image.
    Returns:
        str: Direction code of the image.
    """
    direction_code = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(img_sitk.GetDirection())

    return direction_code

def read_mesh(mesh_path):
    """
    This function voxelizes the mesh using the reference image
    It reads the surface mesh, voxelizes it and saves it in the output directory

    Input:
        mesh_path: path to mesh
    """


    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    polydata = reader.GetOutput()
    
    return polydata

def convert_vtk_to_obj(vtk_path, obj_path):
    """
    Convert a VTK mesh to OBJ format.

    Args:
        vtk_path (str): Path to the input VTK mesh file.
        obj_path (str): Path to save the output OBJ file.
    """
    
    polydata = read_mesh(vtk_path)

    # --- Write OBJ ---
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(obj_path)
    writer.SetInputData(polydata)
    writer.Update()

def voxelize_mesh_to_sitk_image(polydata, reference_img):
    """
    Rasterizes a VTK surface mesh into a SimpleITK binary image.

    This function takes a 3D surface mesh and converts it into a binary volumetric image (SimpleITK.Image), where voxels inside the mesh are set to 1 and those outside are set to 0.
    The output image matches the spacing, origin, and size of the provided reference image.
    Args:
        polydata (vtk.vtkPolyData): The input VTK surface mesh to be voxelized.
        reference_img (sitk.Image): The reference SimpleITK image whose geometry (spacing, origin, size) will be used for the output image.
    Returns:
        sitk.Image: A SimpleITK binary image with voxels inside the mesh set to 1 and outside set to 0, matching the reference image geometry.
    """

    # Ensure reference_img is LPS. VERY IMPORTANT for vtkmesh in this function. Otherwise reorient:
    direction_code = get_direction_code(reference_img)
    if direction_code != 'LPS':
        reference_img = reorient_sitk(reference_img, 'LPS')
        # raise ValueError(f"You need to reorient scan to LPS. Right now direction is: {direction_code}")
        
    
    spacing = reference_img.GetSpacing()
    origin = reference_img.GetOrigin()
    size = reference_img.GetSize()

    # --- 2. Create a vtkImageData with same geometry ---
    white_image = vtk.vtkImageData()
    white_image.SetSpacing(spacing)
    white_image.SetOrigin(origin)
    white_image.SetDimensions(size)
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Fill with 255 (white)
    white_image.GetPointData().GetScalars().Fill(1)

    # --- 3. Convert mesh to stencil ---
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
    pol2stenc.Update()

    # --- 4. Apply stencil to image ---
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(white_image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)  # outside mesh = 0
    imgstenc.Update()

    # --- 5. Convert VTK image to NumPy ---
    vtk_img = imgstenc.GetOutput()
    dims = vtk_img.GetDimensions()
    vtk_array = vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    voxel_data = vtk_array.reshape(dims[2], dims[1], dims[0])  # Z, Y, X

    # --- 6. Convert to sitk.Image ---
    sitk_image = sitk.GetImageFromArray(voxel_data)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    
    # Reorient sitk_image to original direction:
    if direction_code != 'LPS':
        sitk_image = reorient_sitk(sitk_image, direction_code)

    return sitk_image

from typing import Literal

import os.path
from pathlib import Path
import numpy as np
import vtk
from vtk.vtkCommonCore import vtkMath
import SimpleITK as sitk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import json
import csv
from scipy.ndimage import measurements
import skimage.io
from skimage.util import img_as_ubyte
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage import color
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops, find_contours
from datetime import datetime
import edt
from scipy.interpolate import UnivariateSpline

try:
    from vmtk import vmtkscripts
except ImportError as e:
    # print(f"Failed to import vmtk name {e.name} and path {e.path}")
    pass

def write_message_to_log_file(settings, message, scan_id=None, level="warning"):
    base_dir = settings["base_dir"]
    mode = settings["mode"]
    if base_dir is None:
        return

    log_file = os.path.dirname(base_dir) + f"/{mode}_analysis_log.txt"
    log_csv = os.path.dirname(base_dir) + f"/{mode}_analysis_log.csv"
    if not os.path.isdir(os.path.dirname(log_file)):
        Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)

    now_date = datetime.strftime(datetime.now(), "%d-%m-%Y-%H-%M-%S")
    with open(log_file, "a") as file:
        file.write(f"{now_date}: {message}\n")

    if scan_id is not None:
        with open(log_csv, "a") as file:
            file.write(f"{level},{scan_id},{message},{now_date}\n")

def read_json_file(json_name):
    if os.path.exists(json_name):
        try:
            with open(json_name, 'r') as openfile:
                json_stuff = json.load(openfile)
                return json_stuff
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {json_name}")
            return None
    return None

def find_task_segment_id_in_config_file(config_file, segment_to_find):
    task_and_segment_info = read_json_file(config_file)
    if not task_and_segment_info:
        return None, None

    for segment in task_and_segment_info:
        segment_name = segment["segment"]
        task = segment["task"]
        segment_id = segment["id"]

        if segment_name == segment_to_find:
            return task, segment_id

    print(f"Did not find segment {segment_to_find} in {config_file}")
    return None, None

"""
Function to convert a SimpleITK image to a VTK image.

Written by David T. Chen from the National Institute of Allergy
and Infectious Diseases, dchen@mail.nih.gov.
It is covered by the Apache License, Version 2.0:
http://www.apache.org/licenses/LICENSE-2.0
"""

def sitk2vtk(img, flip_for_volume_rendering=False, debugOn=False):
    """Convert a SimpleITK image to a VTK image, via numpy."""

    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()

    # convert the SimpleITK image to a numpy array
    i2 = sitk.GetArrayFromImage(img)
    if debugOn:
        i2_string = i2.tostring()
        print("data string address inside sitk2vtk", hex(id(i2_string)))

    vtk_image = vtk.vtkImageData()

    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)

    if len(origin) == 2:
        origin.append(0.0)

    if len(spacing) == 2:
        spacing.append(spacing[0])

    if len(direction) == 4:
        direction = [
            direction[0],
            direction[1],
            0.0,
            direction[2],
            direction[3],
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        vtk_image.SetDirectionMatrix(direction)

    # TODO: Volume rendering does not support direction matrices (27/5-2023)
    # so sometimes the volume rendering is mirrored
    # this a brutal hack to avoid that
    if flip_for_volume_rendering:
        if direction[4] < 0:
            i2 = np.fliplr(i2)

    # depth_array = numpy_support.numpy_to_vtk(i2.ravel(), deep=True,
    #                                          array_type = vtktype)
    depth_array = numpy_to_vtk(i2.ravel(), deep=True)
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()
    #
    if debugOn:
        print("Volume object inside sitk2vtk")
        print(vtk_image)
        #        print("type = ", vtktype)
        print("num components = ", ncomp)
        print(size)
        print(origin)
        print(spacing)
        print(vtk_image.GetScalarComponentAsFloat(0, 0, 0, 0))

    return vtk_image

def filter_image_with_segmentation(img, mask_img_name, fill_val = -1000):
    i2 = sitk.GetArrayFromImage(img)

    try:
        mask = sitk.ReadImage(mask_img_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {mask_img_name}")
        return img

    m_np = sitk.GetArrayFromImage(mask)
    m_mask = m_np > 0.5
    i2[~m_mask] = fill_val
    img_o = sitk.GetImageFromArray(i2)
    img_o.CopyInformation(img)

    return img_o

def read_nifti_itk_to_vtk(file_name, img_mask_name=None, flip_for_volume_rendering=None):
    try:
        img = sitk.ReadImage(file_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {file_name}")
        return None

    if img_mask_name is not None:
        img = filter_image_with_segmentation(img, img_mask_name)

    vtk_image = sitk2vtk(img, flip_for_volume_rendering)
    return vtk_image

def read_nifti_itk_to_numpy(file_name):
    try:
        img = sitk.ReadImage(file_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {file_name}")
        return None, None, None

    i2 = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    size = img.GetSize()
    return i2, spacing, size

def preprocess_for_centerline_extraction(vtk_in):
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(vtk_in)
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

    smooth_filter = vtk.vtkSmoothPolyDataFilter()
    smooth_filter.SetInputData(cleaner.GetOutput())
    smooth_filter.SetNumberOfIterations(100)
    smooth_filter.SetRelaxationFactor(0.1)
    smooth_filter.FeatureEdgeSmoothingOff()
    smooth_filter.BoundarySmoothingOn()
    smooth_filter.Update()

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(smooth_filter.GetOutput())
    decimate.SetTargetReduction(0.90)
    decimate.PreserveTopologyOn()
    decimate.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(decimate.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()

def convert_label_map_to_surface(label_name, reset_direction_matrix=False, segment_id=1,
                                only_largest_component=False):
    debug = False
    vtk_img = read_nifti_itk_to_vtk(label_name)
    if vtk_img is None:
        return None

    # Check if there is any data
    vol_np = vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    # remember that label = -2048 denotes out of scan
    if np.sum(vol_np > 0) < 1:
        if debug:
            print(f"No valid labels in {label_name}")
        return None

    if reset_direction_matrix:
        direction = [1, 0, 0.0, 0, 1, 0.0, 0.0, 0.0, 1.0]
        vtk_img.SetDirectionMatrix(direction)

    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetNumberOfContours(1)
    mc.SetValue(0, segment_id)
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() < 10:
        if debug:
            print(f"No isosurface found in {label_name} for segment {segment_id}")
        return None

    surface = mc.GetOutput()
    if only_largest_component:
        conn = vtk.vtkConnectivityFilter()
        conn.SetInputConnection(mc.GetOutputPort())
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        surface = conn. GetOutput()
    return surface

def convert_label_map_to_surface_file(label_name, output_file, reset_direction_matrix=False, segment_id=1,
                                      only_largest_component=False):

    print(f"Generating: {output_file}")
    surface = convert_label_map_to_surface(label_name, reset_direction_matrix, segment_id, only_largest_component)

    if surface is None:
        return False

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(surface)
    writer.SetFileTypeToBinary()
    writer.SetFileName(output_file)
    writer.Write()
    return True

def compute_set_of_surfaces(settings, task, segments, segm_base_dir_in=None):
    """
    task: The TotalSegmentator task (total, body etc). Can potentially also be other segmentation algorithms.
    """
    segment_base_dir = f'{settings["segment_base_dir"]}{task}/'
    surface_dir = settings["surface_dir"]
    if segm_base_dir_in is not None:
        segment_base_dir = segm_base_dir_in

    for segment in segments:
        segm_name = f"{segment_base_dir}{segment}.nii.gz"
        surface_name = f"{surface_dir}{task}_{segment}_surface.vtk"
        if not convert_label_map_to_surface_file(segm_name, surface_name):
            print(f"Could not convert {segm_name} to surface")

def read_landmarks(filename):
    if not os.path.exists(filename):
        return None

    x, y, z = 0, 0, 0
    with open(filename) as f:
        for line in f:
            if len(line) > 1:
                temp = line.split()  # Remove whitespaces and line endings and so on
                x, y, z = np.double(temp)
    return x, y, z

def add_distances_from_landmark_to_centerline(in_center, lm_in):
    """
    Add scalar values to a center line where each value is the accumulated distance to the start point
    :param in_center: vtk center line
    :param lm_in: start landmark
    :return: centerline with scalar values
    """
    n_points = in_center.GetNumberOfPoints()
    # print(f"Number of points in centerline: {n_points}")

    cen_p_start = in_center.GetPoint(0)
    cen_p_end = in_center.GetPoint(n_points - 1)
    dist_start = np.linalg.norm(np.subtract(lm_in, cen_p_start))
    dist_end = np.linalg.norm(np.subtract(lm_in, cen_p_end))
    # print(f"Dist start: {dist_start} end: {dist_end}")

    start_idx = 0
    inc = 1
    # Go reverse
    if dist_start > dist_end:
        start_idx = n_points - 1
        inc = -1

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    scalars = vtk.vtkDoubleArray()
    scalars.SetNumberOfComponents(1)

    idx = start_idx
    p_1 = in_center.GetPoint(idx)
    accumulated_dist = 0
    pid = points.InsertNextPoint(p_1)
    scalars.InsertNextValue(accumulated_dist)

    while 0 < idx <= n_points:
        idx += inc
        p_2 = in_center.GetPoint(idx)
        dist = np.linalg.norm(np.subtract(p_1, p_2))
        # pid = points.InsertNextPoint(p_1)
        # scalars.InsertNextValue(accumulated_dist)
        accumulated_dist += dist
        pid_2 = points.InsertNextPoint(p_2)
        scalars.InsertNextValue(accumulated_dist)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(pid)
        lines.InsertCellPoint(pid_2)
        p_1 = p_2
        pid = pid_2

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    del points
    pd.SetLines(lines)
    del lines
    pd.GetPointData().SetScalars(scalars)
    del scalars

    # print(f"Number of points in refined centerline: {pd.GetNumberOfPoints()}")
    return pd

def add_start_and_end_point_to_centerline(in_center, start_point, end_point):
    """
    Add start and end point centerline
    """
    n_points = in_center.GetNumberOfPoints()
    # print(f"Number of points in centerline: {n_points}")

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    pid = points.InsertNextPoint(start_point)

    for idx in range(n_points):
        p_2 = in_center.GetPoint(idx)
        pid_2 = points.InsertNextPoint(p_2)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(pid)
        lines.InsertCellPoint(pid_2)
        pid = pid_2

    pid_2 = points.InsertNextPoint(end_point)
    lines.InsertNextCell(2)
    lines.InsertCellPoint(pid)
    lines.InsertCellPoint(pid_2)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    del points
    pd.SetLines(lines)
    del lines

    # print(f"Number of points in refined centerline: {pd.GetNumberOfPoints()}")
    return pd

def compute_spline_from_path(cl_in, cl_out_file, spline_smoothing_factor = 20, sample_spacing=0.25):
    sum_dist = 0
    n_points = cl_in.GetNumberOfPoints()

    x = []
    y_1 = []
    y_2 = []
    y_3 = []
    p = cl_in.GetPoint(n_points - 1)
    p_old = p
    # Compute the three individual components of the path
    # it is parameterised using the length along the path

    for idx in range(n_points):
        # We go backwards to fix a reverse distance problem
        p = cl_in.GetPoint(n_points-idx-1)
        d = np.linalg.norm(np.array(p) - np.array(p_old))
        sum_dist += d
        p_old = p
        x.append(sum_dist)
        y_1.append(p[0])
        y_2.append(p[1])
        y_3.append(p[2])

    min_x = 0
    max_x = sum_dist

    spl_1 = UnivariateSpline(x, y_1)
    spl_2 = UnivariateSpline(x, y_2)
    spl_3 = UnivariateSpline(x, y_3)
    spl_1.set_smoothing_factor(spline_smoothing_factor)
    spl_2.set_smoothing_factor(spline_smoothing_factor)
    spl_3.set_smoothing_factor(spline_smoothing_factor)

    # self.spline_parameters = {
    #     "min_x": self.min_x,
    #     "max_x": self.max_x,
    #     "spl_1": self.spl_1,
    #     "spl_2": self.spl_2,
    #     "spl_3": self.spl_3
    # }


# def compute_sampled_spline_curve(self):
    samp_space = sample_spacing
    spline_n_points = int(max_x / samp_space)
    print(f"Computing sampled spline path with length {max_x:.1f} and sample spacing {samp_space} "
          f"resulting in {spline_n_points} samples for smoothing")

    # Compute a polydata object with the spline points
    xs = np.linspace(min_x, max_x, spline_n_points)

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    scalars = vtk.vtkDoubleArray()
    scalars.SetNumberOfComponents(1)

    current_idx = 0
    sum_dist = 0
    sp = [spl_1(xs[current_idx]), spl_2(xs[current_idx]), spl_3(xs[current_idx])]
    pid = points.InsertNextPoint(sp)
    scalars.InsertNextValue(sum_dist)
    current_idx += 1

    while current_idx < spline_n_points:
        p_1 = [spl_1(xs[current_idx]), spl_2(xs[current_idx]), spl_3(xs[current_idx])]
        sum_dist += np.linalg.norm(np.array(p_1) - np.array(sp))
        lines.InsertNextCell(2)
        pid_2 = points.InsertNextPoint(p_1)
        scalars.InsertNextValue(sum_dist)
        lines.InsertCellPoint(pid)
        lines.InsertCellPoint(pid_2)
        pid = pid_2
        sp = p_1
        current_idx += 1

    vtk_spline = vtk.vtkPolyData()
    vtk_spline.SetPoints(points)
    del points
    vtk_spline.SetLines(lines)
    del lines
    vtk_spline.GetPointData().SetScalars(scalars)
    del scalars

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(vtk_spline)
    writer.SetFileName(cl_out_file)
    writer.SetFileTypeToBinary()
    writer.Write()

def compute_single_center_line(surface_name, cl_name, start_p_name, end_p_name):
    if os.path.exists(cl_name):
        print(f"{cl_name} already exists - skippping")
        return True
    debug = False

    cl_org_name = os.path.splitext(cl_name)[0] + "_original.vtk"
    cl_spline_name = cl_name

    # cl_spline_name = os.path.splitext(cl_name)[0] + "_spline.vtk"
    cl_surface_name = os.path.splitext(cl_name)[0] + "_surface_in.vtk"

    if not os.path.exists(surface_name):
        print(f"Could not read {surface_name}")
        return False
    print(f"Computing centerline from {surface_name}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(surface_name)
    reader.Update()
    surf_in = reader.GetOutput()

    surface = preprocess_for_centerline_extraction(surf_in)
    # surface = surf_in
    if debug:
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(surface)
        writer.SetFileName(cl_surface_name)
        writer.SetFileTypeToBinary()
        writer.Write()

    start_point = read_landmarks(start_p_name)
    end_point = read_landmarks(end_p_name)
    if start_point is None or end_point is None:
        print(f"Could not read landmarks {start_p_name} or {end_p_name}")
        return False

    # computes the centerlines using vmtk
    centerlinePolyData = vmtkscripts.vmtkCenterlines()
    centerlinePolyData.Surface = surface
    centerlinePolyData.SeedSelectorName = "pointlist"
    centerlinePolyData.SourcePoints = start_point
    # endpoints needs to be: len(end_points) mod 3 = 0
    centerlinePolyData.TargetPoints = end_point
    # centerlinePolyData.AppendEndPoints = 1
    try:
        centerlinePolyData.Execute()
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"When computing cl on {surface_name}")
        return False
    # except:
    #     print(f"Got an exception")
    #     print(f"When computing cl on {aorta_surf_name}")
    #     return False
    # print(centerlinePolyData.Centerlines)
    # print("\n--compute centerlines done--")

    if centerlinePolyData.Centerlines.GetNumberOfPoints() < 10:
        print("Something wrong with centerline")
        return False

    # Adding start and end points did not work very well. Too many errors at the ends
    # cl_added = add_start_and_end_point_to_centerline(centerlinePolyData.Centerlines, end_point, start_point)
    # cl_dists = add_distances_from_landmark_to_centerline(cl_added, start_point)
    cl_dists = add_distances_from_landmark_to_centerline(centerlinePolyData.Centerlines, start_point)

    # saves the centerlines in new vtk file
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(cl_dists)
    writer.SetFileName(cl_org_name)
    writer.SetFileTypeToBinary()
    writer.Write()

    cl_added = add_start_and_end_point_to_centerline(centerlinePolyData.Centerlines, end_point, start_point)
    compute_spline_from_path(cl_added, cl_spline_name)

    return True

def set_window_and_level_on_single_slice(img_in, img_window, img_level):
    out_min = 0
    out_max = 1
    in_min = img_level - img_window / 2
    in_max = img_level + img_window / 2
    # in_max = 800

    # https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    out_img = rescale_intensity(img_in, in_range=(in_min, in_max), out_range=(out_min, out_max))

    return out_img

def write_single_slice_with_overlay(single_slice_np, single_slice_np_img,
                                    img_window, img_level, slice_out_name, slice_out_name_cropped):
    boundary = find_boundaries(single_slice_np, mode='thick')

    single_slice_np_img = set_window_and_level_on_single_slice(single_slice_np_img, img_window, img_level)
    scaled_ubyte = img_as_ubyte(single_slice_np_img)
    scaled_2_rgb = color.gray2rgb(scaled_ubyte)
    rgb_boundary = [255, 0, 0]
    scaled_2_rgb[boundary > 0] = rgb_boundary
    skimage.io.imsave(slice_out_name, np.flipud(scaled_2_rgb))

    region_p = regionprops(img_as_ubyte(boundary))
    if len(region_p) < 1:
        print(f"No regions found for {slice_out_name}")
        return
    bbox = list(region_p[0].bbox)

    shp = boundary.shape
    # Extend bbox range.
    # TODO set value elsewhere
    extend = 20
    bbox[0] = max(0, bbox[0] - extend)
    bbox[1] = max(0, bbox[1] - extend)
    bbox[2] = min(shp[0], bbox[2] + extend)
    bbox[3] = min(shp[1], bbox[3] + extend)

    scaled_2_rgb_crop = scaled_2_rgb[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    skimage.io.imsave(slice_out_name_cropped, np.flipud(scaled_2_rgb_crop))

def extract_orthonogonal_slices_from_given_segment(settings, task, segment_name, segment_id,
                                       img_np, label_img_np, hu_stats):
    stat_dir = settings["statistics_dir"]
    image_out_dir = settings["image_out_dir"]
    slice_1_out_rgb = f"{image_out_dir}{task}_{segment_name}_slice_1_rgb.png"
    # slice_1_out_label = f"{stat_dir}{segment_name}_slice_1_label.png"
    slice_1_out_rgb_crop = f"{image_out_dir}{task}_{segment_name}_slice_1_rgb_crop.png"
    slice_2_out_rgb = f"{image_out_dir}{task}_{segment_name}_slice_2_rgb.png"
    slice_2_out_rgb_crop = f"{image_out_dir}{task}_{segment_name}_slice_2_rgb_crop.png"
    slice_3_out_rgb = f"{image_out_dir}{task}_{segment_name}_slice_3_rgb.png"
    slice_3_out_rgb_crop = f"{image_out_dir}{task}_{segment_name}_slice_3_rgb_crop.png"
    # slice_out_info = f"{stat_dir}{segment_name}_slice_info.json"
    # Default values. If the values is -10000 they should be automatically computed
    visualization_min_hu = settings["visualization_min_hu"]
    visualization_max_hu = settings["visualization_max_hu"]
    recompute_all = settings["force_recompute_all"]

    if not recompute_all and os.path.exists(slice_1_out_rgb):
        return

    mask_np = label_img_np == segment_id
    if np.sum(mask_np) < 10:
        print(f"{segment_name} from task {task} not present for orthonogonal slice extraction")
        return

    if hu_stats is not None:
        stats_check = hu_stats.get(f"{task}_{segment_name}", None)
        if stats_check is None:
            print(f"{task}_{segment_name} not present HU stats file")
            return

    print(f"Computing orthogonal slices for {segment_name} from task {task}")

    if hu_stats is not None:
        if visualization_min_hu <= -10000:
            visualization_min_hu = hu_stats[f"{task}_{segment_name}"]["q01_hu"]
        if visualization_max_hu <= -10000:
            visualization_max_hu = hu_stats[f"{task}_{segment_name}"]["q99_hu"]
    img_window = visualization_max_hu - visualization_min_hu
    img_level = (visualization_max_hu + visualization_min_hu) / 2.0
    # print(f"Computed window {img_window} and level {img_level}")

    com_np = measurements.center_of_mass(mask_np)

    rel_idx = int(com_np[0])
    single_slice_np = mask_np[rel_idx, :, :]
    single_slice_np_img = img_np[rel_idx, :, :]
    write_single_slice_with_overlay(single_slice_np, single_slice_np_img, img_window, img_level,
                                    slice_1_out_rgb, slice_1_out_rgb_crop)

    rel_idx = int(com_np[1])
    single_slice_np = mask_np[:, rel_idx, :]
    single_slice_np_img = img_np[:, rel_idx, :]
    write_single_slice_with_overlay(single_slice_np, single_slice_np_img, img_window, img_level,
                                    slice_2_out_rgb, slice_2_out_rgb_crop)

    rel_idx = int(com_np[2])
    single_slice_np = mask_np[:, :, rel_idx]
    single_slice_np_img = img_np[:, :, rel_idx]
    write_single_slice_with_overlay(single_slice_np, single_slice_np_img, img_window, img_level,
                                    slice_3_out_rgb, slice_3_out_rgb_crop)

def get_components_over_certain_size(segmentation, min_size=5000, max_number_of_components=2):
    debug = False
    labels = label(segmentation)
    bin_c = np.bincount(labels.flat, weights=segmentation.flat)
    # probably extremely unefficient
    comp_ids = []
    for c in range(max_number_of_components):
        idx = np.argmax(bin_c)
        if bin_c[idx] > min_size:
            comp_ids.append(idx)
            bin_c[idx] = 0

    if len(comp_ids) < 1:
        if debug:
            print(f"No connected components with size above {min_size} found")
        return None, None
    largest_cc = labels == comp_ids[0]
    for idx in range(1, len(comp_ids)):
        largest_cc = np.bitwise_or(largest_cc, labels == comp_ids[idx])

    return largest_cc, len(comp_ids)

def edt_based_overlap(segmentation_1, segmentation_2, spacing, radius):
    """
    Compute the overlap between two segmentations using the Euclidean distance transform
    """
    sdf_mask_1 = -edt.sdf(segmentation_1, anisotropy=[spacing[0], spacing[1], spacing[2]],
                        parallel=8)
    sdf_mask_2 = -edt.sdf(segmentation_2, anisotropy=[spacing[0], spacing[1], spacing[2]],
                        parallel=8)
    overlap_mask = (sdf_mask_1 < radius) & (sdf_mask_2 < radius)

    # overlap_mask = np.bitwise_and(sdf_mask_1 < radius, sdf_mask_2 < radius)
    return overlap_mask

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
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from numpy import dot
import edt
from skimage.measure import find_contours
import warnings
warnings.filterwarnings("ignore", message="No mtl file provided") # Ignore this ugly warning.

def decimate_and_smooth(surface_in, surface_out):
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

import SimpleITK as sitk
import numpy as np

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




def load_or_create_vector_fields(msk_inside, msk_outside, com_zyx, spacing, vf_combined_path=None):
    """Load or create inside/outside vector fields and optionally persist them."""

    # Load if file exists
    if vf_combined_path is not None and os.path.exists(vf_combined_path):
        print(f"Loading vector fields from {vf_combined_path}")
        combined_data = np.load(vf_combined_path).astype(np.float32)

        vx_outside, vy_outside, vz_outside = combined_data[0], combined_data[1], combined_data[2]
        vx_inside,  vy_inside,  vz_inside  = combined_data[3], combined_data[4], combined_data[5]

        return vx_outside, vy_outside, vz_outside, vx_inside, vy_inside, vz_inside

    # Otherwise compute
    print("Creating vector fields...")
    vx_outside, vy_outside, vz_outside = create_internal_external_vector_fields_fast(
        msk_outside, com_zyx, spacing, region="external"
    )
    vx_inside, vy_inside, vz_inside = create_internal_external_vector_fields_fast(
        msk_inside, com_zyx, spacing, region="internal"
    )

    # Combine
    combined_data = np.stack(
        [vx_outside, vy_outside, vz_outside,
         vx_inside,  vy_inside,  vz_inside],
        axis=0
    ).astype(np.float32)

    # Save ONLY if path is provided
    if vf_combined_path is not None:
        print(f"Saving vector fields to {vf_combined_path}")
        np.save(vf_combined_path, combined_data)

    return vx_outside, vy_outside, vz_outside, vx_inside, vy_inside, vz_inside


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

def get_annotated_slices(gt_mask):
    """Return z-indices where gt has any annotation."""
    return np.where(gt_mask.any(axis=(1, 2)))[0]

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
    pred_sitk = voxelize_mesh_to_sitk_image(polydata, gt_sitk)
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
        polydata_voxelized_sitk = voxelize_mesh_to_sitk_image(polydata, reference_sitk)
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



def load_and_normalize_itk(path, center, scale, spacing_override=None):
    """Loads image and applies the mesh-space transformation."""
    img = sitk.ReadImage(str(path))
    if spacing_override is None:
        spacing_override = img.GetSpacing()
        
    # Transform to match normalized mesh space
    new_origin = (np.array(img.GetOrigin()) - center) / scale
    new_spacing = np.array(spacing_override) / scale
    
    img.SetOrigin(tuple(new_origin))
    img.SetSpacing(tuple(new_spacing))
    return img