from os import walk
from os.path import join, split
import fnmatch
from skimage import img_as_uint, dtype_limits
from skimage.io import imread, imsave
from skimage.morphology import square, closing, opening, erosion
from skimage.filters import threshold_otsu, threshold_isodata
from numpy import mean, std, array, copy, stack, append
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

def tophat_reconstruction(raw_mat, elem_mat):
    eroded_mat = erosion(raw_mat, elem_mat)
    opened_mat = reconstruction(eroded_mat, raw_mat)
    tophat_mat = raw_mat - opened_mat
    return tophat_mat

def binarize_npf(file_path_str):
    micro_manager_mat = imread(file_path_str)
    closed_mat = closing(micro_manager_mat, square(3))
    opened_mat = opening(closed_mat, square(3))
    cut_off = mean(opened_mat) + 3 * std(opened_mat)
    bw_mat = opened_mat > max((threshold_isodata(opened_mat), cut_off))
    bw_mat ^= erosion(bw_mat, square(7))
    label_mat = label(bw_mat)
    mean_background_int = mean(micro_manager_mat[~bw_mat])
    properties_list = regionprops(label_mat, micro_manager_mat - mean_background_int)
    diameter_row = 0.5 * MICRON_PER_PIXEL * array([x.major_axis_length + x.minor_axis_length for x in properties_list])
    is_large_row = diameter_row > DIAMETER_THRESHOLD
    intensity_row = array([x.mean_intensity for x in properties_list])
    
    # Save images.
    rescaled_mat = rescale_intensity(micro_manager_mat)
    zeros_mat = copy(rescaled_mat)
    zeros_mat[bw_mat] = dtype_limits(micro_manager_mat)[0]
    ones_mat = copy(rescaled_mat)
    ones_mat[bw_mat] = dtype_limits(micro_manager_mat)[1]
    rgb_mat = stack((zeros_mat, ones_mat, zeros_mat), axis = -1)
    imsave(file_path_str[:-4] + '_segmentation.tif', img_as_uint(rgb_mat))
    return intensity_row[is_large_row]
    
def measure_npf_density(folder_name_str):
    file_path_list = find_tif_files(folder_name_str)
    no_files = len(file_path_list)
    npf_intensity_row = array([])
    for i in range(no_files):
        i_intensity_row = binarize_npf(file_path_list[i])
        npf_intensity_row = append(npf_intensity_row, i_intensity_row)
    return npf_intensity_row