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

def tophat_reconstruction(raw_mat, elem_mat):
    eroded_mat = erosion(raw_mat, elem_mat)
    opened_mat = reconstruction(eroded_mat, raw_mat)
    tophat_mat = raw_mat - opened_mat
    return tophat_mat

def binarize_npf(file_path_str, save_image = False):
    mm_stack = imread(file_path_str)
    npf_mat = mm_stack[0, :, :]
    rescaled_mat = rescale_intensity(npf_mat)
    bw_mat = binary_fill_holes(canny(rescaled_mat, sigma = 0.3))
    bw_mat = remove_small_objects(bw_mat, min_size = sum(disk(6)))
    bw_mat ^= erosion(bw_mat, disk(3))
    label_mat, no_labels = label(bw_mat, return_num = True)
    mean_background_int = mean(npf_mat[~bw_mat])
    properties_list = regionprops(label_mat, npf_mat - mean_background_int)
    diameter_row = 0.5 * MICRON_PER_PIXEL * array([x.major_axis_length + x.minor_axis_length for x in properties_list])
    is_large_row = diameter_row > DIAMETER_THRESHOLD
    npf_intensity_row = array([x.mean_intensity for x in properties_list])
    actin_intensity_row = zeros(npf_intensity_row.shape)
    
    if no_labels >= 1:
        actin_mat = gaussian(mm_stack[1, :, :], sigma = 0.5)
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
                actin_intensity_row[i - 1] = mean(actin_mat[i_bw_actin_mat])
                bw_actin_mat[i_bw_search_mat & bw_otsu_mat] = True
        #imsave(file_path_str[:-4] + '_reconstructed.tif', img_as_uint(bw_actin_mat))
        
    # Save images.
    if save_image == True:
        zeros_mat = copy(rescaled_mat)
        zeros_mat[bw_mat] = dtype_limits(npf_mat)[0]
        ones_mat = copy(rescaled_mat)
        ones_mat[bw_mat] = dtype_limits(npf_mat)[1]
        rgb_mat = stack((zeros_mat, ones_mat, zeros_mat), axis = -1)
        imsave(file_path_str[:-4] + '_segmentation.tif', img_as_uint(rgb_mat))
    return npf_intensity_row, actin_intensity_row
        
def measure_npf_density(folder_name_str):
    file_path_list = find_tif_files(folder_name_str)
    no_files = len(file_path_list)
    npf_intensity_row = array([])
    actin_intensity_row = array([])
    for i in range(no_files):
        i_intensity_row, i_actin_intensity_row = binarize_npf(file_path_list[i])
        npf_intensity_row = append(npf_intensity_row, i_intensity_row)
        actin_intensity_row = append(actin_intensity_row, i_actin_intensity_row)
    return npf_intensity_row, actin_intensity_row