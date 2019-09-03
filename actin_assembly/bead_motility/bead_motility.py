from os import walk
from os.path import join, split
import fnmatch
from skimage import img_as_uint, dtype_limits
from skimage.io import imread, imsave
from skimage.feature import canny
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import disk, dilation, erosion, remove_small_objects, reconstruction
from skimage.filters import threshold_otsu, gaussian
from numpy import sum, mean, std, array, copy, stack, append, zeros, all
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.exposure import rescale_intensity

MICRON_PER_PIXEL = 0.16
DIAMETER_THRESHOLD = 2.0

def find_tif_files(folder_name_str):
    tif_file_path_list = []
    search_str = '*ome.tif'
    for (dir_name_str, sub_dir_name_list, file_name_list) in walk(folder_name_str):
        for i_file_name in file_name_list:
            if fnmatch.fnmatch(i_file_name, search_str):
                tif_file_path_list.append(join(dir_name_str, i_file_name))
    return tif_file_path_list

def make_rgb_overlay(gray_mat, bw_mat, rgb_color_row):
    max_type = dtype_limits(gray_mat)[1]
    red_scale = rgb_color_row[0]
    green_scale = rgb_color_row[1]
    blue_scale = rgb_color_row[2]
    rgb_mat = stack((gray_mat, gray_mat, gray_mat), axis = -1)
    rgb_mat[bw_mat, 0] = red_scale * max_type
    rgb_mat[bw_mat, 1] = green_scale * max_type
    rgb_mat[bw_mat, 2] = blue_scale * max_type
    return rgb_mat

def measure_npf_density(file_path_str, save_images = False):
    # Load micromanager stack.
    mm_stack = imread(file_path_str)
    # Binarize npf-coated bead surface using Canny edge detector.
    npf_mat = mm_stack[0, :, :]
    npf_rescaled_mat = rescale_intensity(1.0 * npf_mat)
    bw_mat = binary_fill_holes(canny(npf_rescaled_mat, sigma = 0.2))
    bw_mat = remove_small_objects(bw_mat, min_size = sum(disk(6)))
    bw_mat ^= erosion(bw_mat, disk(3))
    # Measure npf intensity.
    label_mat, no_labels = label(bw_mat, return_num = True)
    mean_background_int = mean(npf_mat[~bw_mat])
    properties_list = regionprops(label_mat, npf_mat - mean_background_int)
    diameter_row = 0.5 * MICRON_PER_PIXEL * array([x.major_axis_length + x.minor_axis_length for x in properties_list])
    is_large_row = diameter_row > DIAMETER_THRESHOLD
    npf_intensity_row = array([x.mean_intensity for x in properties_list])
    # Save images.
    if save_images == True:
        npf_rgb_mat = make_rgb_overlay(npf_rescaled_mat, bw_mat, [0.0, 1.0, 0.0])
        imsave(file_path_str[:-4] + '_npf_segmentation.tif', img_as_uint(npf_rgb_mat))
    return npf_intensity_row
    
def measure_actin_density(file_path_str, save_images = False):
    # Load micromanager stack.
    mm_stack = imread(file_path_str)
    # Binarize npf-coated bead surface using Canny edge detector.
    npf_mat = mm_stack[0, :, :]
    npf_rescaled_mat = rescale_intensity(1.0 * npf_mat)
    bw_mat = binary_fill_holes(canny(npf_rescaled_mat, sigma = 0.3))
    bw_mat = remove_small_objects(bw_mat, min_size = sum(disk(6)))
    bw_mat ^= erosion(bw_mat, disk(3))
    # Measure npf intensity.
    label_mat, no_labels = label(bw_mat, return_num = True)
    mean_background_int = mean(npf_mat[~bw_mat])
    properties_list = regionprops(label_mat, npf_mat - mean_background_int)
    diameter_row = 0.5 * MICRON_PER_PIXEL * array([x.major_axis_length + x.minor_axis_length for x in properties_list])
    is_large_row = diameter_row > DIAMETER_THRESHOLD
    npf_intensity_row = array([x.mean_intensity for x in properties_list])
    # Save images.
    if save_images == True:
        npf_rgb_mat = make_rgb_overlay(npf_rescaled_mat, bw_mat, [0.0, 1.0, 0.0])
        imsave(file_path_str[:-4] + '_npf_segmentation.tif', img_as_uint(npf_rgb_mat))
    
    # Measure actin intensity.
    actin_intensity_row = zeros(npf_intensity_row.shape)
    if no_labels >= 1:
        actin_mat = gaussian(mm_stack[1, :, :], sigma = 0.5)
        actin_rescaled_mat = rescale_intensity(1.0 * mm_stack[1, :, :])
        mean_actin_back_au = mean(actin_mat)
        std_actin_back_au = std(actin_mat)
        bw_otsu_mat = actin_mat > max(threshold_otsu(actin_mat),
                                      mean_actin_back_au + 3 * std_actin_back_au)
        bw_otsu_mat = remove_small_objects(bw_otsu_mat, min_size = sum(disk(3)))
        bw_actin_mat = zeros(actin_mat.shape, dtype = bool)
        for i in range(1, no_labels + 1):
            i_bw_npf_mat = binary_fill_holes(label_mat == i)
            i_bw_dilated_mat = dilation(i_bw_npf_mat, disk(15))
            i_bw_search_mat = i_bw_dilated_mat ^ i_bw_npf_mat
            i_bw_actin_mat = i_bw_search_mat & bw_otsu_mat
            if all(i_bw_actin_mat == False):
                actin_intensity_row[i - 1] = mean_actin_back_au
            else:
                actin_intensity_row[i - 1] = mean(actin_mat[i_bw_actin_mat]) - mean_actin_back_au
                bw_actin_mat[i_bw_search_mat & bw_otsu_mat] = True
        if save_images == True:
            actin_rgb_mat = make_rgb_overlay(actin_rescaled_mat, bw_actin_mat, [1.0, 0.0, 0.0])
            imsave(file_path_str[:-4] + '_actin_segmentation.tif', img_as_uint(actin_rgb_mat))
    return npf_intensity_row, actin_intensity_row
        
def batch_measure_actin_density(folder_name_str, save_images = False):
    file_path_list = find_tif_files(folder_name_str)
    no_files = len(file_path_list)
    npf_intensity_row = array([])
    actin_intensity_row = array([])
    for i in range(no_files):
        i_intensity_row, i_actin_intensity_row = measure_actin_density(file_path_list[i], save_images)
        npf_intensity_row = append(npf_intensity_row, i_intensity_row)
        actin_intensity_row = append(actin_intensity_row, i_actin_intensity_row)
    return npf_intensity_row, actin_intensity_row

def batch_measure_npf_density(folder_name_str, save_images = False):
    file_path_list = find_tif_files(folder_name_str)
    no_files = len(file_path_list)
    npf_intensity_row = array([])
    for i in range(no_files):
        i_intensity_row = measure_npf_density(file_path_list[i], save_images)
        npf_intensity_row = append(npf_intensity_row, i_intensity_row)
    return npf_intensity_row