import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from matplotlib.patches import Patch
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def load_nifti_as_numpy(img_path):
    """
    Load image as numpy array
    
    input:
        img_path: input to image path
    output:
        img: numpy image
    """
    img_sitk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    if img.ndim == 4:
        img = np.squeeze(img, axis=0)
    img = img.transpose(2, 1, 0)
    return img

def load_nifti_as_sitk(path):
    img = sitk.ReadImage(path)
    return img

def convert_sitk_to_numpy(img_sitk):
    img = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    if img.ndim == 4:
        img = np.squeeze(img, axis=0)
    img = img.transpose(2, 1, 0)
    return img

def convert_numpy_to_sitk(img_numpy):
    """
    Convert a NumPy array to a SimpleITK image.

    Args:
        img_numpy (numpy.ndarray): The input NumPy array (3D or 4D).
        
    Returns:
        sitk.Image: The converted SimpleITK image.
    """
    # Transpose the axes back to match SimpleITK's convention
    img_numpy = img_numpy.transpose(2, 1, 0)

    # Convert to SimpleITK image
    img_sitk = sitk.GetImageFromArray(img_numpy)
    
    return img_sitk
    
def save_sitk_as_nifti_from_ref(img_sitk, ref_img_sitk, output_path):
    """
    Save a SimpleITK image as a NIfTI file using a reference image for metadata.

    Args:
        img_sitk (SimpleITK.Image): The image to save.
        ref_img_sitk (SimpleITK.Image): The reference image to copy metadata from.
        output_path (str): Path to save the NIfTI file.
    """
    
    # Copy metadata from the reference image
    img_sitk.CopyInformation(ref_img_sitk)
    # Save the image
    sitk.WriteImage(img_sitk, output_path)
    # print(f"Saved image to: {output_path}")
    
    

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

def get_info(img_sitk):
    """
    Get image information from SimpleITK image.
    
    Args:
        img_sitk (SimpleITK.Image): Input image.
        
    Returns:
        dict: Dictionary containing image information.
    """
    info = {
        'size': img_sitk.GetSize(),
        'spacing': img_sitk.GetSpacing(),
        'origin': img_sitk.GetOrigin(),
        'direction': img_sitk.GetDirection(),
        # 'direction_code': get_direction_code(img_sitk),
        'pixel_type': img_sitk.GetPixelIDTypeAsString()
    }
    return print(info)
    
    

def get_spacing(img_path):
    """
    Get image spacing
    
    input:
        img_path: input to nifti image path
    output:
        spacing: image spacing
    """
            
    img_sitk = sitk.ReadImage(img_path)
    spacing = img_sitk.GetSpacing()
    spacing = np.array(spacing)
    spacing = spacing[:3] #If dimension is > 4 (an extra 1-dimensional channel, then we only want the first 3 dimensions). This is the case for RH data.
    return spacing



def plot_central_slice_img(img, spacing=None, title=None, output_path=None):
    """
    Plot central slice of image

    input:
        img: image (numpy)
        spacing: image spacing (numpy)
        title: plot title (str)
    """

    if spacing is None:
        spacing = np.ones(3) # default isotropic spacing
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(img[img.shape[0] // 2].T, cmap='gray',origin="lower")
    ax[0].set_title('Sagittal')
    ax[0].set_aspect(spacing[2] / spacing[1])
    ax[1].imshow(img[:, img.shape[1] // 2].T, cmap='gray',origin="lower")
    ax[1].set_title('Coronal')
    ax[1].set_aspect(spacing[2] / spacing[0])
    ax[2].imshow(img[:, :, img.shape[2] // 2].T, cmap='gray',origin="lower")
    ax[2].set_title('Axial')
    ax[2].set_aspect(spacing[1] / spacing[0])

    if title is not None:
        fig.suptitle(title)

    if output_path is not None:
        fig.savefig(output_path)
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_central_slice_img_zyx(img, spacing=None, title=None, output_path=None):
    """
    Plot central slice of image

    input:
        img: image (numpy) ASSUMES ZYX ORDER
        spacing: image spacing (numpy) ASSUMES ZYX ORDER 
        title: plot title (str)
    """

    if spacing is None:
        spacing = np.ones(3)
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(img[img.shape[0] // 2], cmap='gray',origin="lower")
    ax[0].set_title('Axial')
    ax[0].set_aspect(spacing[1] / spacing[2])
    ax[1].imshow(img[:, img.shape[1] // 2], cmap='gray',origin="lower")
    ax[1].set_title('Coronal')
    ax[1].set_aspect(spacing[0] / spacing[2])
    ax[2].imshow(img[:, :, img.shape[2] // 2], cmap='gray',origin="lower")
    ax[2].set_title('Sagittal')
    ax[2].set_aspect(spacing[0] / spacing[1])

    if title is not None:
        fig.suptitle(title)

    if output_path is not None:
        fig.savefig(output_path)
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_central_slice_mask(mask, spacing=None, title=None, output_path=None):
    """
    Plot central slice of mask

    input:
        mask: mask (numpy)
        spacing: image spacing (numpy)
        title: plot title (str)
    """

    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]

    colors = ["black"] + [plt.cm.tab10(i) for i in range(len(unique_classes))]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.concatenate(([0], unique_classes))
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(bounds) + 1) - 0.5, ncolors=len(bounds))

    if spacing is None:
        spacing = np.ones(3)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(mask[mask.shape[0] // 2].T, cmap=cmap, norm=norm, origin="lower")
    ax[0].set_title('Sagittal')
    ax[0].set_aspect(spacing[2] / spacing[1])
    ax[1].imshow(mask[:, mask.shape[1] // 2].T, cmap=cmap, norm=norm, origin="lower")
    ax[1].set_title('Coronal')
    ax[1].set_aspect(spacing[2] / spacing[0])
    ax[2].imshow(mask[:, :, mask.shape[2] // 2].T, cmap=cmap, norm=norm, origin="lower")
    ax[2].set_title('Axial')
    ax[2].set_aspect(spacing[1] / spacing[0])

    if title is not None:
        fig.suptitle(title)

    if output_path is not None:
        fig.savefig(output_path)
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_central_slice_mask_zyx(mask, spacing=None, title=None, output_path=None):
    """
    Plot central slice of mask (ZYX order).

    input:
        mask: mask (numpy, ZYX order)
        spacing: image spacing (numpy, ZYX order)
        title: plot title (str)
    """

    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]

    colors = ["black"] + [plt.cm.tab10(i) for i in range(len(unique_classes))]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.concatenate(([0], unique_classes))
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(bounds) + 1) - 0.5, ncolors=len(bounds))

    if spacing is None:
        spacing = np.ones(3)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(mask[mask.shape[0] // 2], cmap=cmap, norm=norm, origin="lower")
    ax[0].set_title('Axial')
    ax[0].set_aspect(spacing[1] / spacing[2])
    ax[1].imshow(mask[:, mask.shape[1] // 2], cmap=cmap, norm=norm, origin="lower")
    ax[1].set_title('Coronal')
    ax[1].set_aspect(spacing[0] / spacing[2])
    ax[2].imshow(mask[:, :, mask.shape[2] // 2], cmap=cmap, norm=norm, origin="lower")
    ax[2].set_title('Sagittal')
    ax[2].set_aspect(spacing[0] / spacing[1])

    if title is not None:
        fig.suptitle(title)

    if output_path is not None:
        fig.savefig(output_path)
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_central_slice_img_mask(img, mask, spacing=None, title=None, output_path=None, alpha=0.7):
    """
    Plot central slices of a 3D scan with a segmented mask overlay.

    Args:
        img (numpy array): 3D medical scan (CT/MRI).
        mask (numpy array): 3D segmentation mask with labeled classes.
        spacing (numpy array, optional): Physical spacing for correct aspect ratio.
        title (str, optional): Plot title.
        alpha (float, optional): Transparency level for the mask overlay. Default is 0.7.
        output_path (str, optional): Path to save the figure. If None, the figure will be shown.
    """

    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]

    colors = ["black"] + [plt.cm.tab10(i % 10) for i in range(len(unique_classes))]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.concatenate(([0], unique_classes))
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(bounds) + 1) - 0.5, ncolors=len(bounds))

    if spacing is None:
        spacing = np.ones(3)

    sagittal_idx = img.shape[0] // 2
    coronal_idx = img.shape[1] // 2
    axial_idx = img.shape[2] // 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    def create_rgba_mask(mask_slice):
        rgba_mask = np.zeros((*mask_slice.shape, 4))
        unique_vals = np.unique(mask_slice)
        for i, val in enumerate(unique_vals):
            if val == 0:
                continue
            color = cmap(bounds.tolist().index(val))[:3]
            rgba_mask[..., :3][mask_slice == val] = color
            rgba_mask[..., 3][mask_slice == val] = alpha
        return rgba_mask

    ax[0].imshow(img[sagittal_idx].T, cmap='gray', origin="lower")
    ax[0].imshow(create_rgba_mask(mask[sagittal_idx].T), origin="lower")
    ax[0].set_title('Sagittal')
    ax[0].set_aspect(spacing[2] / spacing[1])

    ax[1].imshow(img[:, coronal_idx].T, cmap='gray', origin="lower")
    ax[1].imshow(create_rgba_mask(mask[:, coronal_idx].T), origin="lower")
    ax[1].set_title('Coronal')
    ax[1].set_aspect(spacing[2] / spacing[0])

    ax[2].imshow(img[:, :, axial_idx].T, cmap='gray', origin="lower")
    ax[2].imshow(create_rgba_mask(mask[:, :, axial_idx].T), origin="lower")
    ax[2].set_title('Axial')
    ax[2].set_aspect(spacing[1] / spacing[0])

    plt.tight_layout()

    if title is not None:
        fig.suptitle(title)

    if output_path is not None:
        fig.savefig(output_path)
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_central_slice_img_mask_zyx(img, mask, spacing=None, title=None, output_path=None, alpha=0.7):
    """
    Plot central slices of a 3D scan with a segmented mask overlay (ZYX order).

    Args:
        img (numpy array): 3D medical scan (ZYX order).
        mask (numpy array): 3D segmentation mask with labeled classes (ZYX order).
        spacing (numpy array, optional): Physical spacing for correct aspect ratio (ZYX order).
        title (str, optional): Plot title.
        alpha (float, optional): Transparency level for the mask overlay. Default is 0.7.
        output_path (str, optional): Path to save the figure. If None, the figure will be shown.
    """

    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]

    colors = ["black"] + [plt.cm.tab10(i % 10) for i in range(len(unique_classes))]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.concatenate(([0], unique_classes))
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(bounds) + 1) - 0.5, ncolors=len(bounds))

    if spacing is None:
        spacing = np.ones(3)

    z_idx = img.shape[0] // 2
    y_idx = img.shape[1] // 2
    x_idx = img.shape[2] // 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    def create_rgba_mask(mask_slice):
        rgba_mask = np.zeros((*mask_slice.shape, 4))
        unique_vals = np.unique(mask_slice)
        for i, val in enumerate(unique_vals):
            if val == 0:
                continue
            color = cmap(bounds.tolist().index(val))[:3]
            rgba_mask[..., :3][mask_slice == val] = color
            rgba_mask[..., 3][mask_slice == val] = alpha
        return rgba_mask

    ax[0].imshow(img[z_idx], cmap='gray', origin="lower")
    ax[0].imshow(create_rgba_mask(mask[z_idx]), origin="lower")
    ax[0].set_title('Axial')
    ax[0].set_aspect(spacing[1] / spacing[2])

    ax[1].imshow(img[:, y_idx], cmap='gray', origin="lower")
    ax[1].imshow(create_rgba_mask(mask[:, y_idx]), origin="lower")
    ax[1].set_title('Coronal')
    ax[1].set_aspect(spacing[0] / spacing[2])

    ax[2].imshow(img[:, :, x_idx], cmap='gray', origin="lower")
    ax[2].imshow(create_rgba_mask(mask[:, :, x_idx]), origin="lower")
    ax[2].set_title('Sagittal')
    ax[2].set_aspect(spacing[0] / spacing[1])

    plt.tight_layout()

    if title is not None:
        fig.suptitle(title)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        # print(f"Saved figure to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_specific_slice_img_mask_zyx(img, mask, dim, slice_idx = None, spacing=None, title=None, output_path=None, alpha=0.7):
    """
    Plot central slices of a 3D scan with a segmented mask overlay (ZYX order).

    Args:
        img (numpy array): 3D medical scan (ZYX order).
        mask (numpy array): 3D segmentation mask with labeled classes (ZYX order).
        spacing (numpy array, optional): Physical spacing for correct aspect ratio (ZYX order).
        title (str, optional): Plot title.
        alpha (float, optional): Transparency level for the mask overlay. Default is 0.7.
        output_path (str, optional): Path to save the figure. If None, the figure will be shown.
    """

    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]

    colors = ["black"] + [plt.cm.tab10(i % 10) for i in range(len(unique_classes))]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.concatenate(([0], unique_classes))
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(bounds) + 1) - 0.5, ncolors=len(bounds))

    if spacing is None:
        spacing = np.ones(3)

    if dim == 'z':
        z_idx = slice_idx if slice_idx is not None else img.shape[0] // 2
        y_idx = img.shape[1] // 2
        x_idx = img.shape[2] // 2
    elif dim == 'y':
        z_idx = img.shape[0] // 2
        y_idx = slice_idx if slice_idx is not None else img.shape[1] // 2
        x_idx = img.shape[2] // 2
    elif dim == 'x':
        z_idx = img.shape[0] // 2
        y_idx = img.shape[1] // 2
        x_idx = slice_idx if slice_idx is not None else img.shape[2] // 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    def create_rgba_mask(mask_slice):
        rgba_mask = np.zeros((*mask_slice.shape, 4))
        unique_vals = np.unique(mask_slice)
        for i, val in enumerate(unique_vals):
            if val == 0:
                continue
            color = cmap(bounds.tolist().index(val))[:3]
            rgba_mask[..., :3][mask_slice == val] = color
            rgba_mask[..., 3][mask_slice == val] = alpha
        return rgba_mask

    ax[0].imshow(img[z_idx], cmap='gray', origin="lower")
    ax[0].imshow(create_rgba_mask(mask[z_idx]), origin="lower")
    ax[0].set_title('Axial')
    ax[0].set_aspect(spacing[1] / spacing[2])

    ax[1].imshow(img[:, y_idx], cmap='gray', origin="lower")
    ax[1].imshow(create_rgba_mask(mask[:, y_idx]), origin="lower")
    ax[1].set_title('Coronal')
    ax[1].set_aspect(spacing[0] / spacing[2])

    ax[2].imshow(img[:, :, x_idx], cmap='gray', origin="lower")
    ax[2].imshow(create_rgba_mask(mask[:, :, x_idx]), origin="lower")
    ax[2].set_title('Sagittal')
    ax[2].set_aspect(spacing[0] / spacing[1])

    plt.tight_layout()

    if title is not None:
        fig.suptitle(title)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_slice_idx_img_mask_zyx(img, mask, slice_idx, dim='z', spacing=None,
                                 title=None, output_path=None, alpha=0.7, figsize=(8, 8)):
    """
    Plot a single 2D slice of img with mask overlay at a given index and dimension.

    Args:
        img (numpy array): 3D medical scan (Z, Y, X order).
        mask (numpy array): 3D segmentation mask with labeled classes (Z, Y, X order).
        slice_idx (int): Index along the chosen dimension.
        dim (str): Slicing dimension: 'z' (axial), 'y' (coronal), or 'x' (sagittal).
        spacing (numpy array, optional): Physical spacing in (Z, Y, X) order for aspect ratio.
        title (str, optional): Plot title. If None, no title is shown.
        output_path (str, optional): If provided, save the figure at 300 DPI.
        alpha (float, optional): Transparency for the mask overlay. Default is 0.7.
        figsize (tuple, optional): Figure size. Default is (8, 8).
    """
    dim = dim.lower()
    if dim not in ('z', 'y', 'x'):
        raise ValueError(f"dim must be 'z', 'y', or 'x', got '{dim}'")

    if spacing is None:
        spacing = np.ones(3)

    if dim == 'z':
        img_slice = img[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :]
        aspect = spacing[1] / spacing[2]
    elif dim == 'y':
        img_slice = img[:, slice_idx, :]
        mask_slice = mask[:, slice_idx, :]
        aspect = spacing[0] / spacing[2]
    else:  # 'x'
        img_slice = img[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]
        aspect = spacing[0] / spacing[1]

    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]

    colors = ["black"] + [plt.cm.tab10(i % 10) for i in range(len(unique_classes))]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.concatenate(([0], unique_classes))

    def create_rgba_mask(ms):
        rgba = np.zeros((*ms.shape, 4))
        for val in np.unique(ms):
            if val == 0:
                continue
            color = cmap(bounds.tolist().index(val))[:3]
            rgba[..., :3][ms == val] = color
            rgba[..., 3][ms == val] = alpha
        return rgba

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_slice, cmap='gray', origin='lower')
    ax.imshow(create_rgba_mask(mask_slice), origin='lower')
    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_slice_idx_img_mask_outline_zyx(img, mask, outline_mask, slice_idx, dim='z',
                                         spacing=None, title=None, output_path=None,
                                         alpha=0.7, figsize=(8, 8),
                                         outline_color='lime', outline_linewidth=2):
    """
    Plot a single 2D slice of img with mask overlay AND the contour outline
    of a (possibly different) segmentation mask drawn on top.

    Args:
        img (numpy array): 3D medical scan (Z, Y, X order).
        mask (numpy array): 3D segmentation mask with labeled classes (Z, Y, X order).
                            Set to np.zeros_like(img) if you only want the outline.
        outline_mask (numpy array): 3D binary or label mask (Z, Y, X) whose
                                    non-zero region contour is drawn as an outline.
        slice_idx (int): Index along the chosen dimension.
        dim (str): Slicing dimension: 'z' (axial), 'y' (coronal), or 'x' (sagittal).
        spacing (numpy array, optional): Physical spacing in (Z, Y, X) for aspect ratio.
        title (str, optional): Plot title. If None, no title is shown.
        output_path (str, optional): If provided, save the figure at 300 DPI.
        alpha (float, optional): Transparency for the mask overlay. Default is 0.7.
        figsize (tuple, optional): Figure size. Default is (8, 8).
        outline_color (str, optional): Color for the contour outline. Default is 'lime'.
        outline_linewidth (float, optional): Line width for the contour. Default is 2.
    """
    from skimage.measure import find_contours

    dim = dim.lower()
    if dim not in ('z', 'y', 'x'):
        raise ValueError(f"dim must be 'z', 'y', or 'x', got '{dim}'")

    if spacing is None:
        spacing = np.ones(3)

    if dim == 'z':
        img_slice = img[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :]
        outline_slice = outline_mask[slice_idx, :, :]
        aspect = spacing[1] / spacing[2]
    elif dim == 'y':
        img_slice = img[:, slice_idx, :]
        mask_slice = mask[:, slice_idx, :]
        outline_slice = outline_mask[:, slice_idx, :]
        aspect = spacing[0] / spacing[2]
    else:  # 'x'
        img_slice = img[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]
        outline_slice = outline_mask[:, :, slice_idx]
        aspect = spacing[0] / spacing[1]

    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]

    colors = ["black"] + [plt.cm.tab10(i % 10) for i in range(len(unique_classes))]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.concatenate(([0], unique_classes))

    def create_rgba_mask(ms):
        rgba = np.zeros((*ms.shape, 4))
        for val in np.unique(ms):
            if val == 0:
                continue
            color = cmap(bounds.tolist().index(val))[:3]
            rgba[..., :3][ms == val] = color
            rgba[..., 3][ms == val] = alpha
        return rgba

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_slice, cmap='gray', origin='lower')
    ax.imshow(create_rgba_mask(mask_slice), origin='lower')

    # Draw contour outline of the outline_mask
    binary_outline = (outline_slice > 0).astype(float)
    contours = find_contours(binary_outline, level=0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color=outline_color,
                linewidth=outline_linewidth)

    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_one_slice_img(img, title=None):
    """
    Plot a single slice of an image.

    Args:
        img (numpy array): 2D slice.
        title (str, optional): Plot title.
    """

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img.T, cmap='gray', origin="lower")
    if title is not None:
        ax.set_title(title)
    plt.show()


def plot_one_slice_img_zyx(img, title=None):
    """
    Plot a single slice of an image (ZYX order).

    Args:
        img (numpy array): 2D slice (ZYX order).
        title (str, optional): Plot title.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap='gray', origin="lower")
    if title is not None:
        ax.set_title(title)
    plt.show()


def plot_one_slice_sdf(sdf, positive_thres=None, title=None):
    """
    Plot one slice of sdf

    input:
        sdf: Signed Distance Field (numpy)
        positive_thres: threshold for positive values
        title: plot title (str)
    """

    if positive_thres is not None:
        sdf[sdf > positive_thres] = positive_thres

    plt.figure(figsize=(10, 8))
    plt.imshow(sdf.T, cmap='viridis', origin="lower", vmin=np.min(sdf), vmax=np.max(sdf))
    plt.colorbar(label='Signed Distance')
    plt.xlabel('Y-axis')
    plt.ylabel('X-axis')
    if title is not None:
        plt.title(title)
    plt.show()


def plot_one_slice_sdf_zyx(sdf, positive_thres=None, title=None):
    """
    Plot one slice of sdf (ZYX order).

    input:
        sdf: Signed Distance Field (numpy, ZYX order)
        positive_thres: threshold for positive values
        title: plot title (str)
    """
    if positive_thres is not None:
        sdf[sdf > positive_thres] = positive_thres

    plt.figure(figsize=(10, 8))
    plt.imshow(sdf, cmap='viridis', origin="lower", vmin=np.min(sdf), vmax=np.max(sdf))
    plt.colorbar(label='Signed Distance')
    plt.xlabel('Y-axis')
    plt.ylabel('X-axis')
    if title is not None:
        plt.title(title)
    plt.show()


def plot_central_slice_sdf(sdf, positive_thres=None, spacing=None, title=None):
    """
    Plot central slice of sdf

    input:
        sdf: Signed Distance Field (numpy)
        positive_thres: threshold for positive values
        spacing: image spacing (numpy)
        title: plot title (str)
    """

    if spacing is None:
        spacing = np.ones(3)

    if positive_thres is not None:
        sdf[sdf > positive_thres] = positive_thres

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(sdf[sdf.shape[0] // 2].T, cmap='viridis',origin="lower", vmin=np.min(sdf), vmax=np.max(sdf))
    ax[0].set_title('Sagittal')
    ax[0].set_aspect(spacing[2] / spacing[1])
    ax[1].imshow(sdf[:, sdf.shape[1] // 2].T, cmap='viridis',origin="lower", vmin=np.min(sdf), vmax=np.max(sdf))
    ax[1].set_title('Coronal')
    ax[1].set_aspect(spacing[2] / spacing[0])
    ax[2].imshow(sdf[:, :, sdf.shape[2] // 2].T, cmap='viridis',origin="lower", vmin=np.min(sdf), vmax=np.max(sdf))
    ax[2].set_title('Axial')
    ax[2].set_aspect(spacing[1] / spacing[0])

    if title is not None:
        fig.suptitle(title)

    plt.show()


def plot_central_slice_sdf_zyx(sdf, positive_thres=None, spacing=None, title=None):
    """
    Plot central slice of sdf (ZYX order).

    input:
        sdf: Signed Distance Field (numpy, ZYX order)
        positive_thres: threshold for positive values
        spacing: image spacing (numpy, ZYX order)
        title: plot title (str)
    """
    if spacing is None:
        spacing = np.ones(3)

    if positive_thres is not None:
        sdf[sdf > positive_thres] = positive_thres

    z_idx = sdf.shape[0] // 2
    y_idx = sdf.shape[1] // 2
    x_idx = sdf.shape[2] // 2

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(sdf[z_idx], cmap='viridis', origin="lower", vmin=np.min(sdf), vmax=np.max(sdf))
    ax[0].set_title('Axial')
    ax[0].set_aspect(spacing[1] / spacing[2])
    ax[1].imshow(sdf[:, y_idx], cmap='viridis', origin="lower", vmin=np.min(sdf), vmax=np.max(sdf))
    ax[1].set_title('Coronal')
    ax[1].set_aspect(spacing[0] / spacing[2])
    ax[2].imshow(sdf[:, :, x_idx], cmap='viridis', origin="lower", vmin=np.min(sdf), vmax=np.max(sdf))
    ax[2].set_title('Sagittal')
    ax[2].set_aspect(spacing[0] / spacing[1])

    if title is not None:
        fig.suptitle(title)

    plt.show()


def dice_score(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice score between two binary masks.

    Args:
        y_true (numpy.ndarray): Ground truth binary mask.
        y_pred (numpy.ndarray): Predicted binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice score.
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)



def test_func():
    """
    Test function to check if the script is loaded correctly.
    """
    print("Test function executed successfully.")



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

def resample_to_spacing(img_sitk,new_spacing=None,interpolation: Literal['linear', 'nearest'] = 'nearest', default_pixel_value: int = -2048):
    """ Resample a SimpleITK image to a new spacing.
    Args:        img_sitk (SimpleITK.Image): Input image to be resampled.
        new_spacing (tuple): New spacing in (x, y, z) order as a list.
        interpolation (str): Interpolation method ('linear' or 'nearest').
        default_pixel_value (int): Default pixel value for out-of-bounds regions.
    Returns:        SimpleITK.Image: Resampled image.
    """

    if interpolation not in ('linear', 'nearest'):
        raise ValueError("Interpolation must be 'linear' or 'nearest'.")

    interpolator = (
        sitk.sitkLinear if interpolation == 'linear'
        else sitk.sitkNearestNeighbor
    )

    original_size = img_sitk.GetSize()
    original_spacing = img_sitk.GetSpacing()

    new_size = [
        max(1, int(round(osz * ospc / nspc)))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)

    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetOutputDirection(img_sitk.GetDirection())

    # VERY important for CT
    resampler.SetDefaultPixelValue(-2048)

    return resampler.Execute(img_sitk)



def pad_to_shape(img_sitk, target_size=(128, 128, 128), pad_value=0):
    current_size = img_sitk.GetSize()  # (x, y, z)
    lower_pad = []
    upper_pad = []

    for i in range(3):
        total_pad = target_size[i] - current_size[i]
        if total_pad < 0:
            raise ValueError(f"Image dimension {i} is larger than target size.")
        lower = total_pad // 2
        upper = total_pad - lower
        lower_pad.append(lower)
        upper_pad.append(upper)

    padded = sitk.ConstantPad(img_sitk, lower_pad, upper_pad, pad_value)
    return padded



def crop_to_tighest_mask(mask_sitk: sitk.Image):
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(mask_sitk)

    # GetBoundingBox returns (x_min, y_min, z_min, size_x, size_y, size_z)
    bbox = filt.GetBoundingBox(1)
    index, size = list(bbox[:3]), list(bbox[3:])

    mask_cropped_sitk = sitk.RegionOfInterest(mask_sitk, size=size, index=index)

    return mask_cropped_sitk