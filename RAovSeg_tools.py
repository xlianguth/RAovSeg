import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import label, binary_closing

def ImgNorm(img, norm_type="minmax", percentile_low=None, percentile_high=None, custom_mn=None, custom_mx=None):
    '''
    Image Normalization. The data type of input img should be float but not int.
    '''
    #normalized the image from [min,max] to [0,1]
    if norm_type == "minmax":
        mn = np.min(img)
        mx = np.max(img)
        img_norm = ((img-mn)/(mx-mn))
    #normalized the image from [percentile_low,percentile_high] to [0,1]
    elif norm_type == "percentile_clip":
        mn = np.percentile(img,percentile_low)
        img[img<mn] = mn
        mx = np.percentile(img,percentile_high)   
        img[img>mx] = mx
        img_norm = ((img-mn)/(mx-mn)) 
    #normalized the image with custom range to [0,1]
    elif norm_type == "custom_clip":
        mn = custom_mn
        img[img<mn] = mn
        mx = custom_mx
        img[img>mx] = mx
        img_norm = ((img-mn)/(mx-mn))
    else:
        raise NameError ('No such normalization type')
    return(img_norm)

def ImgResample (image, out_spacing=(0.5, 0.5, 0.5), out_size=None, is_label=False, pad_value=0):
    """Resamples an image to given element spacing and output size."""

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    if out_size is None:
        out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    else:
        out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        #resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(sitk.Cast(image, sitk.sitkFloat32))

def preprocess_ (input, o1, o2):
    """
    Preprocess the input image.

    Args:
        input (np.ndarray): A NumPy array representing the image.
        o1 (float): A value between 0 and 1.
        o2 (float): A value between 0 and 1

    Returns:
        out (np.ndarray): A NumPy array representing the output image.
    """
    # Normalization
    mn = np.min(input)
    mx = np.max(input)
    input_norm = ((input-mn)/(mx-mn))
    # Using o1 and o2 for image enhancement
    out = input_norm.copy()
    out[input > o1] = 1 
    out[input < o1] = input[input < o1]
    out[input > o2] = input[input > o2]
    out[input > 0.5] = 1 - input[input > 0.5]
    return out

ep = 1e-5
def dsc_cal_np(ary1, ary2):
    """
    Calculate the Dice Similarity Coefficient (DSC) between two NumPy arrays.

    Args:
        ary1 (np.ndarray): A NumPy array representing the first label map.
        ary2 (np.ndarray): A NumPy array representing the second label map.

    Returns:
        float: The Dice Similarity Coefficient between the two arrays, rounded to four decimal places.
    """

    # Convert the arrays to boolean if they are not already
    ary1 = (ary1 == 1)
    ary2 = (ary2 == 1)

    # Calculate size1 and size2
    size1 = np.sum(ary1)
    size2 = np.sum(ary2)

    # Calculate the intersection (logical AND)
    intersection = np.logical_and(ary1, ary2)

    # Calculate the size of the intersection
    size_inter = np.sum(intersection)

    # Calculate the Dice Similarity Coefficient
    dsc_ = round((2 * size_inter / (size1 + size2 + ep)), 4)
    
    return dsc_


def dsc_cal_torch(ary1,ary2):
    """
    Compute the Dice Similarity Coefficient (DSC) between two PyTorch tensors.

    Args:
        ary1 (torch.Tensor): The first binary segmentation tensor.
        ary2 (torch.Tensor): The second binary segmentation tensor (typically the ground truth).

    Returns:
        float: Dice Similarity Coefficient between the two tensors, rounded to four decimal places.
    """

    # Calculate size1 and size2
    size1 = len(torch.where(ary1==1)[0])
    size2 = len(torch.where(ary2==1)[0])

    # Calculate the size of the intersection
    intersection = torch.logical_and(ary1, ary2)
    size_inter = len(torch.where(intersection==True)[0])

    # Calculate the Dice Similarity Coefficient
    dsc_ = round((2 * size_inter / (size1 + size2 + ep)),4)

    return dsc_

def postprocess_(binary_array, closing_iterations=10):
    """
    Post-processing of a binary segmentation map using morphological closing and connected components to retain the largest region.

    Args:
        binary_array (np.ndarray): Binary segmentation map.
        closing_iterations (int): Number of iterations for the morphological closing operation.

    Returns:
        np.ndarray: Binary segmentation map after post-processing.
    """
    # Closing operation to fill small gaps and holes
    closed_array = binary_closing(binary_array, iterations=closing_iterations)
    
    # Connected component analysis to label connected regions
    labeled_array, num_features = label(closed_array)
    
    # Find the size of each component
    component_sizes = [np.sum(labeled_array == label_idx) for label_idx in range(1, num_features + 1)]
    
    if len(component_sizes) > 0:
        # Find the label of the largest component
        largest_component_label = np.argmax(component_sizes) + 1
        
        # Create a mask to keep only the largest component
        largest_component_mask = labeled_array == largest_component_label
        
        # Apply the mask to the labeled array
        labeled_array = np.where(largest_component_mask, labeled_array, 0)
        num_features = 1
    
    labeled_array = (labeled_array > 0).astype(int)

    return labeled_array