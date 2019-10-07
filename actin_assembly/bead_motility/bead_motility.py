from os import walk
from os.path import join, split
import fnmatch
from skimage import img_as_float, img_as_uint, dtype_limits
from skimage.io import imread, imsave
from skimage.feature import canny
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import disk, dilation, erosion, remove_small_objects, reconstruction, medial_axis, thin
from skimage.filters import threshold_otsu, gaussian
from numpy import sum, mean, std, array, empty, vstack, zeros, stack
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries, clear_border

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

def read_and_split(file_path_str):
    mm_stack = imread(file_path_str)
    npf_im = img_as_float(mm_stack[:, :, 0])
    actin_im = img_as_float(mm_stack[:, :, 2])
    return npf_im, actin_im

def binarize_npf(npf_im):
    npf_rescaled_im = npf_im / max(npf_im.flatten())
    bw_im = binary_fill_holes(canny(npf_rescaled_im, sigma = 0.1))
    bw_im = remove_small_objects(bw_im, min_size = sum(disk(6)))
    npf_bw_im = clear_border(bw_im ^ erosion(bw_im, disk(3)))
    return npf_bw_im

def measure_npf_fluor(npf_im, npf_bw_im):
    label_im, no_labels = label(npf_bw_im, return_num = True)
    mean_background_fluor = mean(npf_bw_im[~npf_bw_im])
    properties_list = regionprops(label_im, npf_im - mean_background_fluor)
    npf_fluor_row = array([x.mean_intensity for x in properties_list]) * (2**16 - 1)
    return npf_fluor_row

def make_rgb_overlay(gray_im, bw_im, rgb_color_row):
    gray_im = gray_im / max(gray_im.flatten())
    max_type = dtype_limits(gray_im)[1]
    red_scale = rgb_color_row[0]
    green_scale = rgb_color_row[1]
    blue_scale = rgb_color_row[2]
    rgb_im = stack((gray_im, gray_im, gray_im), axis = -1)
    rgb_im[bw_im, 0] = red_scale * max_type
    rgb_im[bw_im, 1] = green_scale * max_type
    rgb_im[bw_im, 2] = blue_scale * max_type
    return rgb_im

def save_segmentation(file_path_str, actin_im, npf_bw_im, tail_axis_bw_im):
    rgb_im = make_rgb_overlay(actin_im, npf_bw_im | tail_axis_bw_im, [0.0, 1.0, 0.0])
    imsave(file_path_str[:-4] + '_tail_segmentation.tif', img_as_uint(rgb_im))

def binarize_actin(actin_im):
    actin_blurred_im = gaussian(actin_im, sigma = 2)
    threshold1 = threshold_otsu(actin_blurred_im)
    actin_bw1_im = actin_blurred_im > threshold1
    back_mean_fluor = mean(actin_blurred_im[~actin_bw1_im].flatten())
    back_std_fluor = std(actin_blurred_im[~actin_bw1_im].flatten())
    threshold2 = back_mean_fluor + 3 * back_std_fluor
    threshold3 = max([threshold1, threshold2])
    actin_bw2_im = clear_border(actin_blurred_im > threshold3)
    return actin_bw2_im

def measure_comet_props(npf_im, npf_bw_im, actin_im, actin_bw_im):
    npf_label_im, no_beads = label(npf_bw_im, return_num = True)
    npf_back_mean = mean(npf_im[~npf_bw_im].flatten())
    actin_back_mean = mean(actin_im[~actin_bw_im].flatten())
    comet_props_mat = empty((1, 3))
    tail_axis_bw_im = zeros(actin_im.shape, dtype = bool)
    for i in range(1, no_beads + 1):
        i_bead_bw_im = npf_label_im == i
        i_npf_fluor = mean(npf_im[i_bead_bw_im].flatten()) - npf_back_mean
        i_npf_fluor *= (2**16 - 1)
        i_bead_actin_bw_im = i_bead_bw_im & actin_bw_im
        i_tail_bw_im = reconstruction(i_bead_actin_bw_im, actin_bw_im)
        i_tail_axis_bw_im = medial_axis(i_tail_bw_im) & ~binary_fill_holes(i_bead_bw_im)
        tail_axis_bw_im[i_tail_axis_bw_im] = True
        i_tail_fluor = mean(actin_im[i_tail_axis_bw_im].flatten()) - actin_back_mean
        i_tail_fluor *= (2**16 - 1)
        i_tail_length = sum(i_tail_axis_bw_im.flatten()) * MICRON_PER_PIXEL
        comet_props_mat = vstack((comet_props_mat, [[i_npf_fluor, i_tail_fluor, i_tail_length]]))
    return comet_props_mat[1:], tail_axis_bw_im

def batch_analysis(folder_path_str):
    file_path_list = find_tif_files(folder_path_str)
    no_files = len(file_path_list)
    comet_props_mat = empty((1, 3))
    for i in range(no_files):
        i_npf_im, i_actin_im = read_and_split(file_path_list[i])
        i_npf_bw_im = binarize_npf(i_npf_im)
        i_actin_bw_im = binarize_actin(i_actin_im)
        i_actin_label_im, i_no_labels = label(i_actin_bw_im, return_num = True)
        if i_no_labels > 0:
            i_comet_props_mat, i_tail_axis_bw_im = measure_comet_props(i_npf_im, i_npf_bw_im, i_actin_im, i_actin_bw_im)
            comet_props_mat = vstack((comet_props_mat, i_comet_props_mat))
            save_segmentation(file_path_list[i], i_actin_im, i_npf_bw_im, i_tail_axis_bw_im)
    return comet_props_mat[1:]
