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

# VMTK only works with a very specific environment
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


def check_for_lock_file(lock_file_name):
    return os.path.exists(lock_file_name)


def generate_lock_file(lock_file_name):
    lock_file = open(lock_file_name, 'w')
    lock_file.write(f"locked")


def remove_lock_file(lock_file_name):
    try:
        os.remove(lock_file_name)
    except OSError:
        pass


def generate_status_file(status_file_name, success=True):
    status_file = open(status_file_name, "w")
    if success:
        status_file.write("success")
    else:
        status_file.write("failed")


def check_for_status_file(status_file_name):
    if not os.path.exists(status_file_name):
        return False
    return True


def check_for_status_file_with_status(status_file_name):
    if not os.path.exists(status_file_name):
        return False, False
    with open(status_file_name, "r") as file:
        status = file.read().strip()
        if status == "success":
            return True, True
        return True, False


def remove_status_file(status_file_name):
    try:
        os.remove(status_file_name)
    except OSError:
        pass

def read_file_list(file_list):
    samples = []
    if len(file_list) == 0:
        return samples

    file = open(file_list, 'r')
    lines = file.readlines()
    if len(lines) < 1:
        print(f"Could not read from {file_list}")
        return []

    for line in lines:
        ls = line.strip()
        samples.append(ls)
    return samples

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

def display_time(seconds, granularity=2):
    # intervals_full = (
    #     ('weeks', 604800),  # 60 * 60 * 24 * 7
    #     ('days', 86400),  # 60 * 60 * 24
    #     ('hours', 3600),  # 60 * 60
    #     ('minutes', 60),
    #     ('seconds', 1),
    # )

    intervals = (
        ('w', 604800),  # 60 * 60 * 24 * 7
        ('d', 86400),  # 60 * 60 * 24
        ('h', 3600),  # 60 * 60
        ('m', 60),
        ('s', 1),
    )
    result = []
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append(f"{value}{name}")
    return ' '.join(result[:granularity])

def extract_meta_information_from_csv(settings):
    meta_info_file = settings["meta_info_file"]
    scan_id = settings["scan_id"]

    scan_meta_info = {}
    scan_meta_info_txt = ""

    if meta_info_file == "":
        # print("No meta info file")
        return scan_meta_info, scan_meta_info_txt

    try:
        file = open(meta_info_file, "r", encoding='utf-8')
    except IOError:
        print(f"Cannot open {meta_info_file}")
        return scan_meta_info, scan_meta_info_txt

    # All the encoding and decoding stuff is done to show the Danish letters correctly
    meta_info = csv.DictReader(file, delimiter=",", skipinitialspace=True)
    for elem in meta_info:
        check_id = scan_id + '.nii.gz'
        if check_id == elem["filename"]:
            try:
                scan_meta_info["StudyDate"] = elem.get("StudyDate", "")
                scan_meta_info_txt = f'StudyDate: {scan_meta_info["StudyDate"]}\n'
                scan_meta_info["StudyDescription"] = elem["StudyDescription"]
                scan_meta_info_txt += f'Study: {scan_meta_info["StudyDescription"]}\n'
                scan_meta_info["SeriesDescription"] = elem["SeriesDescription"]
                scan_meta_info_txt += f'Series: {scan_meta_info["SeriesDescription"]}\n'
                scan_meta_info["ImageType"] = elem["ImageType"]
                scan_meta_info_txt += f'Type: {scan_meta_info["ImageType"]}\n'
                scan_meta_info["SliceThickness"] = elem["SliceThickness"]
                # scan_meta_info_txt += f'Study: {scan_meta_info["StudyDescription"]}\n'
                scan_meta_info["Contrast"] = elem["ContrastBolusAgent"]
                scan_meta_info_txt += f'Contrast: {scan_meta_info["Contrast"]}\n'
                scan_meta_info["Manufacturer"] = elem["Manufacturer"]
                scan_meta_info_txt += f'Manufacturer: {scan_meta_info["Manufacturer"]}\n'
                scan_meta_info["ManufacturerModelName"] = elem["ManufacturerModelName"]
                scan_meta_info_txt += f'Model: {scan_meta_info["ManufacturerModelName"]}\n'
                scan_meta_info["BodyPartExamined"] = elem["BodyPartExamined"]
                # scan_meta_info_txt += f'Study: {scan_meta_info["StudyDescription"]}\n'
                scan_meta_info["ScanOptions"] = elem["ScanOptions"]
                scan_meta_info_txt += f'Options: {scan_meta_info["ScanOptions"]}\n'
                scan_meta_info["AcquisitionType"] = elem["AcquisitionType"]
                scan_meta_info_txt += f'Acquisition: {scan_meta_info["AcquisitionType"]}\n'
                scan_meta_info["ContentTime"] = elem.get("ContentTime", 0)
                scan_meta_info["ProtocolName"] = elem["ProtocolName"]
                scan_meta_info_txt += f'Protocol: {scan_meta_info["ProtocolName"]}\n'
            except UnicodeDecodeError as e:
                print(f"Got UnidecodeError when parsing metainfo for object: {e.object}, encoding: {e.encoding},"
                      f" reason: {e.reason}")
                write_message_to_log_file(settings, f"Exception when decoding metainfo for object: {e.object}, "
                                                    f"encoding: {e.encoding},"
                      f" reason: {e.reason} for {scan_id}", scan_id,
                                          "error")
                scan_meta_info_txt += "Problems reading all meta info due to decoding error"
            except ValueError:
                print(f"Got ValueError when parsing metainfo file")
                write_message_to_log_file(settings, f"Got ValueError when parsing metadata for {scan_id}",
                                          scan_id, "error")
                scan_meta_info_txt += "Problems reading all meta info due to decoding error"

            return scan_meta_info, scan_meta_info_txt

    return scan_meta_info, scan_meta_info_txt


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


def check_if_anatomy_present_in_segmentation(task_and_segment_info_file, segm_base_dir, anatomy_name):
    task, segm_id = find_task_segment_id_in_config_file(task_and_segment_info_file, anatomy_name)
    segm_name = f"{segm_base_dir}{task}/{task}.nii.gz"

    if not os.path.exists(segm_name):
        return False

    # Square millimeters - one square centimer
    volume_threshold = 1000

    try:
        label_img = sitk.ReadImage(segm_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {segm_name}")
        return False

    spacing = label_img.GetSpacing()
    vox_size = spacing[0] * spacing[1] * spacing[2]
    label_img_np = sitk.GetArrayFromImage(label_img)
    mask_np = label_img_np == segm_id
    sum_pix = np.sum(mask_np)
    if sum_pix * vox_size < volume_threshold:
        return False
    return True



def find_closests_points_on_two_surfaces(surface_1, surface_2):
    """
    Find the two points that are closest on each other on the two surfaces
    """
    n_points_1 = surface_1.GetNumberOfPoints()
    n_points_2 = surface_2.GetNumberOfPoints()

    closest_dist = np.inf
    idx_1 = -1
    idx_2 = -1

    for i in range(n_points_1):
        for j in range(n_points_2):
            p_1 = surface_1.GetPoint(i)
            p_2 = surface_2.GetPoint(j)
            dist_squared = vtkMath.Distance2BetweenPoints(p_1, p_2)
            if dist_squared < closest_dist:
                closest_dist = dist_squared
                idx_1 = i
                idx_2 = j

    return idx_1, idx_2, np.sqrt(closest_dist)


def find_closests_points_on_two_surfaces_with_start_point(surface_1, surface_2, start_point_surface_1):
    """
    Find the two points that are closest on each other on the two surfaces
    """
    min_dist = np.Inf

    locator_1 = vtk.vtkPointLocator()
    locator_1.SetDataSet(surface_1)
    locator_1.BuildLocator()

    locator_2 = vtk.vtkPointLocator()
    locator_2.SetDataSet(surface_2)
    locator_2.BuildLocator()
    p_1 = start_point_surface_1

    idx_1 = -1
    idx_2 = -1

    stop = False
    while not stop:
        idx_2 = locator_2.FindClosestPoint(p_1)
        p_2 = surface_2.GetPoint(idx_2)
        idx_1 = locator_1.FindClosestPoint(p_2)
        p_1 = surface_1.GetPoint(idx_1)
        dist_squared = vtkMath.Distance2BetweenPoints(p_1, p_2)
        if dist_squared < min_dist:
            min_dist = dist_squared
        else:
            stop = True

    p_1 = surface_1.GetPoint(idx_1)
    p_2 = surface_2.GetPoint(idx_2)
    avg_p = np.mean(np.stack((p_1, p_2)), axis=0)

    return idx_1, idx_2, avg_p, np.sqrt(min_dist)


"""
Function to convert a SimpleITK image to a VTK image.

Written by David T. Chen from the National Institute of Allergy
and Infectious Diseases, dchen@mail.nih.gov.
It is covered by the Apache License, Version 2.0:
http://www.apache.org/licenses/LICENSE-2.0
"""


# @staticmethod
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


def refine_landmark_using_sphere(surface, lm_in):
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    closest_n = vtk.vtkIdList()
    locator.FindClosestNPoints(500, lm_in, closest_n)

    avg_p = [0.0, 0.0, 0.0]
    for i in range(closest_n.GetNumberOfIds()):
        idx = closest_n.GetId(i)
        p = surface.GetPoint(idx)
        # print(f"idx: {idx} p: {p}")
        avg_p[0] += p[0]
        avg_p[1] += p[1]
        avg_p[2] += p[2]

    # print(f"DEBUG: closest_n : {closest_n.GetNumberOfIds()} avg_p {avg_p}")
    avg_p[0] /= closest_n.GetNumberOfIds()
    avg_p[1] /= closest_n.GetNumberOfIds()
    avg_p[2] /= closest_n.GetNumberOfIds()
    return avg_p


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


def compute_set_of_surfaces_from_uunet(settings, task, segments):
    """
    Works on a combined segmentation
    """
    segm_base_dir = settings["segment_base_dir"]
    surf_output_dir = settings["surface_dir"]
    scan_id = settings["scan_id"]
    label_name = os.path.join(segm_base_dir, f"{scan_id}_{task}.nii.gz")

    vtk_img = read_nifti_itk_to_vtk(label_name)
    if vtk_img is None:
        return False

    for idx, segment_dict in enumerate(segments):
        segm_id = segment_dict["id"]
        segment = segment_dict["segment"]
        surface_name = f"{surf_output_dir}{task}_{segment}_surface.vtk"
        if os.path.exists(surface_name):
            print(f"{surface_name} exists - not recomputing")
        else:
            print(f"Generating: {surface_name}")
            mc = vtk.vtkDiscreteMarchingCubes()
            mc.SetInputData(vtk_img)
            mc.SetNumberOfContours(1)
            mc.SetValue(0, float(segm_id))
            mc.Update()

            if mc.GetOutput().GetNumberOfPoints() < 10:
                print(f"No isosurface found in {label_name}")
            else:
                writer = vtk.vtkPolyDataWriter()
                writer.SetInputConnection(mc.GetOutputPort())
                writer.SetFileTypeToBinary()
                writer.SetFileName(surface_name)
                writer.Write()
    return True


def compute_set_of_surfaces_from_combined_segmentations(settings, task, segments, segm_base_dir_in=None):
    """
    Works on a combined segmentation
    """
    segm_base_dir = settings["segment_base_dir"]
    surf_output_dir = settings["surface_dir"]
    # scan_id = settings["scan_id"]
    if segm_base_dir_in is not None:
        segm_base_dir = segm_base_dir_in
    debug = False

    label_name = os.path.join(segm_base_dir, f"{task}/{task}.nii.gz")

    vtk_img = read_nifti_itk_to_vtk(label_name)
    if vtk_img is None:
        return False

    for idx, segment_dict in enumerate(segments):
        segm_id = segment_dict["id"]
        segment = segment_dict["segment"]
        surface_name = f"{surf_output_dir}{task}_{segment}_surface.vtk"
        if os.path.exists(surface_name):
            print(f"{surface_name} exists - not recomputing")
        else:
            if debug:
                print(f"Generating: {surface_name}")
            mc = vtk.vtkDiscreteMarchingCubes()
            mc.SetInputData(vtk_img)
            mc.SetNumberOfContours(1)
            mc.SetValue(0, float(segm_id))
            mc.Update()

            if mc.GetOutput().GetNumberOfPoints() < 10:
                if debug:
                    print(f"No isosurface found in {label_name}")
                    return False
            else:
                writer = vtk.vtkPolyDataWriter()
                writer.SetInputConnection(mc.GetOutputPort())
                writer.SetFileTypeToBinary()
                writer.SetFileName(surface_name)
                writer.Write()
    return True

# def compute_landmark_relevant_surfaces(settings):
#     # segment_base_dir = settings["segment_base_dir"]
#     # surface_dir = settings["surface_dir"]
#
#     task = "total"
#     segments = ["aorta", "heart_ventricle_left", "iliac_artery_left", "iliac_artery_right"]
#     # for segment in segments:
#     #     segm_name = f"{segment_base_dir}{segment}.nii.gz"
#     #     surface_name = f"{surface_dir}total_{segment}_surface.vtk"
#     #     if not convert_label_map_to_surface(segm_name, surface_name):
#     #         print(f"Could not convert {segm_name} to surface")
#     compute_set_of_surfaces(settings, task, segments)


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


def compute_center_line(settings):
    surface_dir = settings["surface_dir"]
    lm_dir = settings["landmark_dir"]
    cl_dir = settings["centerline_dir"]
    stats_file = f'{settings["statistics_dir"]}/aorta_parts.json'
    scan_id = settings["scan_id"]
    Path(cl_dir).mkdir(parents=True, exist_ok=True)

    n_aorta_parts = 1
    parts_stats = read_json_file(stats_file)
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]

    if n_aorta_parts == 1:
        # task = "computed"
        # segments = ["aorta_left_ventricle"]
        # compute_set_of_surfaces(settings, task, segments)

        aorta_surf_name = f"{surface_dir}/aorta_left_ventricle.vtk"
        cl_name = f"{cl_dir}/aorta_centerline.vtk"
        cl_name_fail = f"{cl_dir}/aorta_centerline_failed.txt"
        start_p_file = f"{lm_dir}aorta_start_point.txt"
        end_p_file = f"{lm_dir}aorta_end_point.txt"
        if os.path.exists(cl_name_fail):
            write_message_to_log_file(settings,
                                       f"Centerline failed before on {aorta_surf_name}",
                                       scan_id, "error")
            return False
        if not compute_single_center_line(aorta_surf_name, cl_name, start_p_file, end_p_file):
            write_message_to_log_file(settings,
                                       f"Failed to compute centerline from {aorta_surf_name}",
                                       scan_id, "error")
            with open(cl_name_fail, 'w') as f:
                f.write(f"Failed to compute centerline from {aorta_surf_name}")
            return False

    elif n_aorta_parts == 2:
        # task = "computed"
        # segments = ["aorta_left_ventricle"]
        # compute_set_of_surfaces(settings, task, segments)

        # TODO: Check if start and end should be switched
        aorta_surf_name = f"{surface_dir}aorta_left_ventricle.vtk"
        cl_name = f"{cl_dir}/aorta_centerline_annulus.vtk"
        cl_name_failed = f"{cl_dir}/aorta_centerline_annulus_failed.txt"
        start_p_file = f"{lm_dir}aorta_start_point_annulus.txt"
        end_p_file = f"{lm_dir}aorta_end_point_annulus.txt"
        if os.path.exists(cl_name_failed):
            write_message_to_log_file(settings,
                                       f"Centerline failed before on {aorta_surf_name}",
                                       scan_id, "error")
            return False
        if not compute_single_center_line(aorta_surf_name, cl_name, start_p_file, end_p_file):
            write_message_to_log_file(settings,
                                       f"Failed to compute centerline from {aorta_surf_name}",
                                       scan_id, "error")
            with open(cl_name_failed, 'w') as f:
                f.write(f"Failed to compute centerline from {aorta_surf_name}")
            return False

        task = "computed"
        segments = ["aorta_lumen_descending"]
        compute_set_of_surfaces(settings, task, segments)

        aorta_surf_name = f"{surface_dir}/computed_aorta_lumen_descending_surface.vtk"
        cl_name = f"{cl_dir}/aorta_centerline_descending.vtk"
        cl_name_failed = f"{cl_dir}/aorta_centerline_descending_failed.txt"
        start_p_file = f"{lm_dir}aorta_start_point_descending.txt"
        end_p_file = f"{lm_dir}aorta_end_point_descending.txt"
        if os.path.exists(cl_name_failed):
            write_message_to_log_file(settings,
                                       f"Centerline failed before on {aorta_surf_name}",
                                       scan_id, "error")
            return False
        if not compute_single_center_line(aorta_surf_name, cl_name, start_p_file, end_p_file):
            write_message_to_log_file(settings,
                                       f"Failed to compute centerline from {aorta_surf_name}",
                                       scan_id, "error")
            with open(cl_name_failed, 'w') as f:
                f.write(f"Failed to compute centerline from {aorta_surf_name}")
            return False
    else:
        print(f"We can not handle {n_aorta_parts} aorta parts")
        write_message_to_log_file(settings,
                                     f"We can not handle {n_aorta_parts} aorta parts",
                                     scan_id, "error")
        return False

    return True


def create_3d_box_footprint(size, spacing):
    """
    size: x,y,z side lengths in mm
    spacing: x,y,z spacing in mm
    """
    # create numpy array
    shape = [1, 1, 1]
    shape[0] = (int(size[0] // spacing[0]) // 2) * 2 + 1
    shape[1] = (int(size[1] // spacing[1]) // 2) * 2 + 1
    shape[2] = (int(size[2] // spacing[2]) // 2) * 2 + 1
    footprint = np.ones(shape).astype(np.uint8)
    return footprint


def create_3d_ball_footprint(radius, spacing):
    """
    radius: radius in mm
    spacing: x,y,z spacing in mm
    """
    # create numpy array
    size = radius * 2
    shape = [1, 1, 1]
    shape[0] = (int(size // spacing[0]) // 2) * 2 + 1
    shape[1] = (int(size // spacing[1]) // 2) * 2 + 1
    shape[2] = (int(size // spacing[2]) // 2) * 2 + 1
    footprint = np.zeros(shape).astype(np.uint8)

    center = [shape[0] // 2, shape[1] // 2, shape[2] // 2]
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if ((x - center[0]) * spacing[0]) ** 2 + \
                        ((y - center[1]) * spacing[1]) ** 2 + ((z - center[2]) * spacing[2]) ** 2 < radius ** 2:
                    footprint[x, y, z] = 1

    return footprint


def check_if_segmentation_hit_sides_of_scan(segmentation, segm_id, n_slices_to_check=8):
    """
    Return the sides (if any) that the segmentation hits
    """
    shp = segmentation.shape
    bin_segm = segmentation == segm_id

    down = np.sum(bin_segm[0:n_slices_to_check, :, :])
    up = np.sum(bin_segm[shp[0]-1-n_slices_to_check:shp[0]-1, :, :])
    left = np.sum(bin_segm[:, 0:n_slices_to_check, :])
    right = np.sum(bin_segm[:, shp[1]-1-n_slices_to_check:shp[1]-1, :])
    front = np.sum(bin_segm[:, :, 0:n_slices_to_check])
    back = np.sum(bin_segm[:, :, shp[2]-1-n_slices_to_check:shp[2]-1])

    sides = set()
    if up > 0:
        sides.add("up")
    if down > 0:
        sides.add("down")
    if left > 0:
        sides.add("left")
    if right > 0:
        sides.add("right")
    if front > 0:
        sides.add("front")
    if back > 0:
        sides.add("back")

    return sides


def compute_segmentation_volume(segmentation_file, segm_id):
    """
    Compute the volume of a segmentation
    """
    try:
        segmentation = sitk.ReadImage(segmentation_file)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {segmentation_file}")
        return 0

    spacing = segmentation.GetSpacing()

    segmentation_np = sitk.GetArrayFromImage(segmentation)
    volume = np.sum(segmentation_np == segm_id)
    volume = volume * spacing[0] * spacing[1] * spacing[2]
    return volume


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


def extract_relevant_slices_based_on_labels(settings):
    scan_id = settings["scan_id"]
    recompute_all = settings["force_recompute_all"]
    segmentation_model = settings["segmentation_model"]
    ct_name = settings["input_file"]
    segm_base_dir = settings["segment_base_dir"]
    mode = settings["mode"]
    task_and_segment_info_file = f'configs/{settings["task_and_segment_config"]}'
    hu_stats_file = f'{settings["statistics_dir"]}/{mode}_hu_and_volume_statistics.json'
    image_out_dir = settings["image_out_dir"]

    print("Computing orthogonal slices from selected segmentations")

    # TODO: better check
    check_file = f'{image_out_dir}/heartchambers_highres_heart_atrium_left_slice_1_rgb_crop.png'
    if not recompute_all and os.path.isfile(check_file):
        print(f"Already computed {check_file}")
        return True

    check_file = f'{image_out_dir}/total_kidney_left_slice_1_rgb_crop.png'
    if not recompute_all and os.path.isfile(check_file):
        print(f"Already computed {check_file}")
        return True

    if not recompute_all and segmentation_model == "TotalSegmentator":
        task_and_segment_info = read_json_file(task_and_segment_info_file)
        if task_and_segment_info is None:
            print(f"Could not read {task_and_segment_info_file}")
            return False
    else:
        # print(f"I am not familiar with model {segmentation_model} but lets see how it goes")
        task_and_segment_info = read_json_file(task_and_segment_info_file)
        if task_and_segment_info is None:
            print(f"Could not read {task_and_segment_info_file}")
            return False

    img_data, spacing, size = read_nifti_itk_to_numpy(ct_name)
    if img_data is None:
        write_message_to_log_file(settings,
                                     f"compute_and_gather_multi_organ_statistics: Error reading {ct_name}",
                                     scan_id, "error")

        print(f"Could not read {ct_name}")
        return False

    Path(image_out_dir).mkdir(parents=True, exist_ok=True)

    # Path(settings["statistics_dir"]).mkdir(parents=True, exist_ok=True)

    hu_stats = read_json_file(hu_stats_file)

    last_read_label_vol = None
    segm_data = None
    for segment in task_and_segment_info:
        segment_name = str(segment["segment"])
        task = str(segment["task"])
        segment_id = segment["id"]

        if segmentation_model == "CustomImageCAS":
            segm_file_name = f'{segm_base_dir}{task}.nii.gz'
        elif segmentation_model == "BartholinatorV1":
            segm_folder = settings["bartholinator_dir"]
            segm_file_name = f"{segm_folder}{scan_id}_labels.nii.gz"
        else:
            ts_base_dir = settings["totalsegmentator_dir"]
            segm_file_name = f"{ts_base_dir}{scan_id}/segmentations/{task}/{task}.nii.gz"
            # segm_file_name = f'{segm_base_dir}{task}/{task}.nii.gz'

        # segm_file_name = f'{segm_base_dir}{task}/{task}.nii.gz'
        if segm_file_name != last_read_label_vol:
            segm_data, _, _ = read_nifti_itk_to_numpy(segm_file_name)
            if segm_data is None:
                write_message_to_log_file(settings,
                                             f"extract_relevant_slices_based_on_labels: Error reading {segm_file_name}",
                                             scan_id, "error")
                return False

        last_read_label_vol = segm_file_name
        if segm_data is not None and segm_data.sum() != 0:
            extract_orthonogonal_slices_from_given_segment(settings, task, segment_name, segment_id, img_data, segm_data,
                                                           hu_stats)

    return True

def extract_specified_slices(settings):
    """
    Only select slices specified in the settings file
    """
    scan_id = settings["scan_id"]
    recompute_all = settings["force_recompute_all"]
    segmentation_model = settings["segmentation_model"]
    ct_name = settings["input_file"]
    segm_base_dir = settings["segment_base_dir"]
    mode = settings["mode"]
    # task_and_segment_info_file = f'configs/{settings["task_and_segment_config"]}'
    hu_stats_file = f'{settings["statistics_dir"]}/{mode}_hu_and_volume_statistics.json'
    image_out_dir = settings["image_out_dir"]
    task_and_segment_info_file = f'configs/{settings["task_and_segment_config"]}'

    organs = settings["organs"]
    for organ in organs:
    #
    #
    # task_segm_list = settings["segment_to_slice"]
    # for tsl in task_segm_list:
    #     task = tsl["task"]
    #     segment_id = tsl["segment_id"]
    #     segment_name = tsl["segment_name"]


        task, segment_id = find_task_segment_id_in_config_file(task_and_segment_info_file, organ)
        segment_name = organ
        # task = settings["segment_to_slice"]["task"]
        # segment_id = settings["segment_to_slice"]["segment_id"]
        # segment_name = settings["segment_to_slice"]["segment_name"]

        print(f"Computing orthogonal slices for {segment_name}")

        check_file = f"{image_out_dir}{task}{segment_name}_1_rgb_crop.png"
        if not recompute_all and os.path.isfile(check_file):
            print(f"Already computed {check_file}")
            continue

        img_data, spacing, size = read_nifti_itk_to_numpy(ct_name)
        if img_data is None:
            write_message_to_log_file(settings,
                                         f"compute_and_gather_multi_organ_statistics: Error reading {ct_name}",
                                         scan_id, "error")

            print(f"Could not read {ct_name}")
            return False

        Path(image_out_dir).mkdir(parents=True, exist_ok=True)

        hu_stats = read_json_file(hu_stats_file)

        if segmentation_model == "CustomImageCAS":
            segm_file_name = f'{segm_base_dir}{task}.nii.gz'
        elif segmentation_model == "BartholinatorV1":
            segm_folder = settings["bartholinator_dir"]
            segm_file_name = f"{segm_folder}{scan_id}_labels.nii.gz"
        else:
            ts_base_dir = settings["totalsegmentator_dir"]
            segm_file_name = f"{ts_base_dir}{scan_id}/segmentations/{task}/{task}.nii.gz"

        segm_data, _, _ = read_nifti_itk_to_numpy(segm_file_name)
        if segm_data is None:
            write_message_to_log_file(settings,
                                         f"extract_relevant_slices_based_on_labels: Error reading {segm_file_name}",
                                         scan_id, "error")
            return False

        if segm_data.sum() == 0:
            write_message_to_log_file(settings,
                                         f"extract_relevant_slices_based_on_labels: Sum of {segm_file_name} is zero",
                                         scan_id, "error")
            continue


        extract_orthonogonal_slices_from_given_segment(settings, task, segment_name, segment_id, img_data, segm_data,
                                                       hu_stats)

    return True


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

def compute_body_segmentation(settings):
    """
    Use simple HU thresholding to compute the body segmentation
    """
    recompute_all = settings["force_recompute_all"]
    # Path(settings["landmark_dir"]).mkdir(parents=True, exist_ok=True)
    # Path(settings["surface_dir"]).mkdir(parents=True, exist_ok=True)
    segm_base_dir = settings["segment_base_dir"]
    Path(f"{segm_base_dir}computed/").mkdir(parents=True, exist_ok=True)
    segm_out_name = f'{segm_base_dir}computed/body.nii.gz'
    ct_name = settings["input_file"]
    low_thresh = -200
    high_thresh = 1500

    if not recompute_all and os.path.exists(segm_out_name):
        print(f"{segm_out_name} already exists - skipping")
        return True

    print(f"Computing {segm_out_name}")

    if not os.path.exists(ct_name):
        print(f"Could not find {ct_name}")
        write_message_to_log_file(settings, f"Could not find {ct_name}", ct_name, "error")
        return False

    try:
        ct_img = sitk.ReadImage(ct_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {ct_name}")
        write_message_to_log_file(settings, f"Could not read {ct_name}", ct_name, "error")
        return False

    ct_np = sitk.GetArrayFromImage(ct_img)

    img_mask_1 = low_thresh < ct_np
    img_mask_2 = ct_np < high_thresh
    combined_mask = np.bitwise_and(img_mask_1, img_mask_2)

    combined_mask, _ = get_components_over_certain_size(combined_mask, 5000, 1)

    img_o = sitk.GetImageFromArray(combined_mask.astype(int))
    img_o.CopyInformation(ct_img)

    # print(f"saving")
    sitk.WriteImage(img_o, segm_out_name)

    return True

def close_cavities_in_segmentations(segmentation):
    background = segmentation == 0
    labels = label(background)
    bin_c = np.bincount(labels.flat, weights=background.flat)
    n_comp = np.count_nonzero(bin_c)
    idx = np.argmax(bin_c)

    connected_background = labels == idx
    closed_segm = np.bitwise_not(connected_background)

    return closed_segm, n_comp

def compute_out_scan_field_segmentation_and_sdf(settings):
    """
    Use simple HU thresholding to compute the area of the scan that is marked as
    invalid values (very low HU). Also compute an SDF so it can be quickly determined
    if a segmentation is close to the scan side.
    """
    recompute_all = settings["force_recompute_all"]
    segm_base_dir = settings["segment_base_dir"]
    Path(f"{segm_base_dir}computed/").mkdir(parents=True, exist_ok=True)
    segm_out_name = f'{segm_base_dir}computed/out_of_scan.nii.gz'
    sdf_out_name = f'{segm_base_dir}computed/out_of_scan_sdf.nii.gz'

    ct_name = settings["input_file"]
    low_thresh = -2000
    high_thresh = 16000

    if not recompute_all and os.path.exists(segm_out_name):
        print(f"{segm_out_name} already exists - skipping")
        return True

    print(f"Computing out-of-scan-field: {segm_out_name}")

    if not os.path.exists(ct_name):
        print(f"Could not find {ct_name}")
        write_message_to_log_file(settings, f"Could not find {ct_name}", ct_name, "error")
        return False

    try:
        ct_img = sitk.ReadImage(ct_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {ct_name}")
        write_message_to_log_file(settings, f"Could not read {ct_name}", ct_name, "error")
        return False

    ct_np = sitk.GetArrayFromImage(ct_img)

    combined_mask = (low_thresh > ct_np) | (ct_np > high_thresh)

    # mark all the sides as well
    combined_mask[:, :, 0] = True
    combined_mask[:, :, -1] = True
    combined_mask[:, 0, :] = True
    combined_mask[:, -1, :] = True
    combined_mask[0, :, :] = True
    combined_mask[-1, :, :] = True

    img_o = sitk.GetImageFromArray(combined_mask.astype(int))
    img_o.CopyInformation(ct_img)
    # print(f"saving")
    sitk.WriteImage(img_o, segm_out_name)

    spacing = ct_img.GetSpacing()
    sdf_mask = -edt.sdf(combined_mask,
                   anisotropy=[spacing[2], spacing[1], spacing[0]],
                   parallel=8  # number of threads, <= 0 sets to num CPU
                   )

    # Clamp SDF to make nifti file smaller
    max_dist_to_keep = 4
    sdf_mask = np.clip(sdf_mask, -max_dist_to_keep, max_dist_to_keep)

    img_o = sitk.GetImageFromArray(sdf_mask)
    img_o.CopyInformation(ct_img)
    sitk.WriteImage(img_o, sdf_out_name)

    return True


def edt_based_opening(segmentation, spacing, radius):
    sdf_mask = -edt.sdf(segmentation, anisotropy=[spacing[0], spacing[1], spacing[2]],
                        parallel=8)
    eroded_mask = sdf_mask < -radius
    sdf_mask = -edt.sdf(eroded_mask, anisotropy=[spacing[0], spacing[1], spacing[2]],
                        parallel=8)
    opened_mask = sdf_mask < radius
    return opened_mask

def edt_based_closing(segmentation, spacing, radius):
    sdf_mask = -edt.sdf(segmentation, anisotropy=[spacing[0], spacing[1], spacing[2]],
                        parallel=8)
    dilated_mask = sdf_mask < radius
    sdf_mask = -edt.sdf(dilated_mask, anisotropy=[spacing[0], spacing[1], spacing[2]],
                        parallel=8)
    closed_mask = sdf_mask < -radius
    return closed_mask

def edt_based_dilation(segmentation, spacing, radius):
    sdf_mask = -edt.sdf(segmentation, anisotropy=[spacing[0], spacing[1], spacing[2]],
                        parallel=8)
    dilated_mask = sdf_mask < radius
    return dilated_mask

def edt_based_erosion(segmentation, spacing, radius):
    sdf_mask = -edt.sdf(segmentation, anisotropy=[spacing[0], spacing[1], spacing[2]],
                        parallel=8)
    eroded_mask = sdf_mask < -radius
    return eroded_mask


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


def edt_based_compute_landmark_from_segmentation_overlap(segmentation_1, segmentation_2, radius, segm_sitk_img,
                                               overlap_name, lm_name, only_larges_components=True, debug=False):
    if debug:
        print(f"Computing {overlap_name} and {lm_name}")

    spacing = segm_sitk_img.GetSpacing()
    spc_trans = [spacing[2], spacing[1], spacing[0]]
    overlap_mask = edt_based_overlap(segmentation_1, segmentation_2, spc_trans, radius)
    if only_larges_components:
        overlap_mask, n_comp = get_components_over_certain_size(overlap_mask, 100, 1)
        if overlap_mask is None or n_comp < 1:
            if debug:
                print(f"No components found in {overlap_name}")
            return False

    if np.sum(overlap_mask) == 0:
        print(f"No overlap found for {overlap_name}")
        return False

    com_np = measurements.center_of_mass(overlap_mask)
    com_np = [com_np[2], com_np[1], com_np[0]]

    com_phys = segm_sitk_img.TransformIndexToPhysicalPoint([int(com_np[0]), int(com_np[1]), int(com_np[2])])
    if debug:
        img_o = sitk.GetImageFromArray(overlap_mask.astype(int))
        img_o.CopyInformation(segm_sitk_img)

        print(f"saving {overlap_name}")
        sitk.WriteImage(img_o, overlap_name)

    end_p_out = open(lm_name, "w")
    end_p_out.write(f"{com_phys[0]} {com_phys[1]} {com_phys[2]}")
    end_p_out.close()
    return True
