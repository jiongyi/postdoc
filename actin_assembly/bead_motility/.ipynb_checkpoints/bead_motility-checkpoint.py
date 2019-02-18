from os import walk
from os.path import join, split
import fnmatch
from skimage import img_as_uint
from skimage.io import imread, imsave
from skimage.morphology import erosion, reconstruction, disk
from skimage.filters import threshold_otsu
from numpy import mean, array, copy, stack, append
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
    mmstack = imread(file_path_str)
    lambda_470_mat = mmstack[0, :, :]
    tophat_mat = tophat_reconstruction(lambda_470_mat, disk(3))
    bw_mat = tophat_mat > threshold_otsu(tophat_mat)
    label_mat = label(bw_mat)
    mean_background_int = mean(lambda_470_mat[~bw_mat])
    properties_list = regionprops(label_mat, lambda_470_mat - mean_background_int)
    diameter_row = 0.5 * MICRON_PER_PIXEL * array([x.major_axis_length + x.minor_axis_length for x in properties_list])
    is_large_row = diameter_row > DIAMETER_THRESHOLD
    intensity_row = array([x.mean_intensity for x in properties_list])
    return intensity_row[is_large_row]
    
    # Save images.
    rescaled_mat = rescale_intensity(lambda_470_mat, out_range = (0.0, 1.0))
    perim_mat = find_boundaries(bw_mat)
    zeros_mat = copy(rescaled_mat)
    zeros_mat[perim_mat] = 0.0
    ones_mat = copy(rescaled_mat)
    ones_mat[perim_mat] = 1.0
    rgb_mat = stack((zeros_mat, ones_mat, zeros_mat), axis = -1)
    imsave(file_path_str[:-4] + '_segmentation.tif', img_as_uint(rgb_mat))
    
def measure_npf_density(folder_name_str):
    file_path_list = find_tif_files(folder_name_str)
    no_files = len(file_path_list)
    npf_intensity_row = array([])
    for i in range(no_files):
        i_intensity_row = binarize_npf(file_path_list[i])
        npf_intensity_row = append(npf_intensity_row, i_intensity_row)
    return npf_intensity_row