from os import walk
from os.path import join, split
from fnmatch import fnmatch
from numpy import zeros, argmax, arange, stack, sum, mean, std, copy, array, max, min, append
from skimage import img_as_uint, img_as_float, dtype_limits
from skimage.io import imread, imsave
from skimage.filters import gaussian, sobel, threshold_otsu, threshold_isodata
from skimage.morphology import disk, erosion, reconstruction, remove_small_objects
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops, label

# Define constants.
NO_PIXELS_2_UM2 = 0.10185185185**2

def find_tif_files(folder_name, search_str):
    tif_file_path_list = []
    for (dir_path, dir_name_list, file_name_list) in walk(folder_name):
        for i_file_name in file_name_list:
            if fnmatch(i_file_name, search_str):
                tif_file_path_list.append(join(dir_path, i_file_name))
    return tif_file_path_list

def extend_depth_field(z_stack_mat):
    no_slices, no_rows, no_columns = z_stack_mat.shape
    sobel_mat = zeros((no_slices, no_rows, no_columns))
    for i in range(no_slices):
        sobel_mat[i, :, :] = sobel(z_stack_mat[i, :, :])
    idx_max_mat = argmax(sobel_mat, axis = 0)
    z_stack_mat = z_stack_mat.reshape((no_slices, -1))
    z_stack_mat = z_stack_mat.transpose()
    extended_mat = z_stack_mat[arange(len(z_stack_mat)), idx_max_mat.ravel()]
    extended_mat = extended_mat.reshape((no_rows, no_columns))
    return extended_mat

def tophat_reconstruction(raw_mat, px_radius):
    eroded_mat = erosion(raw_mat, disk(px_radius))
    opened_mat = reconstruction(eroded_mat, raw_mat)
    tophat_mat = raw_mat - opened_mat
    tophat_mat[tophat_mat < 0] = 0.0
    return tophat_mat

def save_segmentation(root_file_name, gaussian_lambda_stack_mat, bw_lambda_stack_mat):
    red_mat = rescale_intensity(gaussian_lambda_stack_mat[:, :, 2])
    green_mat = rescale_intensity(gaussian_lambda_stack_mat[:, :, 1])
    blue_mat = rescale_intensity(gaussian_lambda_stack_mat[:, :, 0])
    rgb_stack_mat = stack((red_mat, green_mat, blue_mat), axis = -1)

    bw_red_mat = find_boundaries(bw_lambda_stack_mat[:, :, 2])
    bw_green_mat = find_boundaries(bw_lambda_stack_mat[:, :, 1])
    bw_blue_mat = find_boundaries(bw_lambda_stack_mat[:, :, 0])

    (min_dtype, max_dtype) = dtype_limits(red_mat)
    overlay_red_mat = copy(red_mat) / max_dtype
    overlay_red_mat[bw_red_mat] = 1.0
    overlay_green_mat = copy(green_mat) / max_dtype
    overlay_green_mat[bw_green_mat] = 1.0
    overlay_blue_mat = copy(blue_mat) / max_dtype
    overlay_blue_mat[bw_blue_mat] = 1.0

    overlay_red_stack_mat = stack((overlay_red_mat, bw_red_mat, bw_red_mat), axis = -1)
    overlay_green_stack_mat = stack((bw_green_mat, overlay_green_mat, bw_green_mat), axis = -1)
    overlay_blue_stack_mat = stack((bw_blue_mat, bw_blue_mat, overlay_blue_mat), axis = -1)

    imsave(root_file_name[:-4] + '_bw_561.tif', img_as_uint(overlay_red_stack_mat))
    imsave(root_file_name[:-4] + '_bw_488.tif', img_as_uint(overlay_green_stack_mat))
    imsave(root_file_name[:-4] + '_bw_405.tif', img_as_uint(overlay_blue_stack_mat))
    imsave(root_file_name[:-4] + '_extended.tif', img_as_uint(rgb_stack_mat))

def segment_stack(stack_file_path):
    mm_stack = imread(stack_file_path)
    no_channels, no_slices, no_rows, no_columns = mm_stack.shape
    # Extend depth of field.
    extended_lambda1_mat = extend_depth_field(mm_stack[0, :, :, :])
    extended_lambda2_mat = extend_depth_field(mm_stack[1, :, :, :])
    extended_lambda3_mat = extend_depth_field(mm_stack[2, :, :, :])
    # Apply gaussian blurr.
    gaussian_lambda1_mat = gaussian(extended_lambda1_mat, sigma = 2)
    gaussian_lambda2_mat = gaussian(extended_lambda2_mat, sigma = 2)
    gaussian_lambda3_mat = gaussian(extended_lambda3_mat, sigma = 2)
    return gaussian_lambda_stack_mat, bw_lambda_stack_mat

def batch_segment_dapi(folder_name):
    extended_mat_file_name_list = find_tif_files(folder_name, '*.ome_extended.tif')
    no_files = len(extended_mat_file_name_list)
    dapi_intensity_row = array([])
    for i in range(no_files):
        i_extended_mat = imread(extended_mat_file_name_list[i])
        dapi_intensity_row = append(dapi_intensity_row, i_extended_mat[:, :, 0].flatten())
    dapi_threshold = threshold_otsu(dapi_intensity_row)
    for i in range(no_files):
        i_extended_mat = imread(extended_mat_file_name_list[i])
        i_bw_lambda1_mat = i_extended_mat[:, :, 0] >= dapi_threshold
        imsave(extended_mat_file_name_list[i][:-4] + '_bw_lambda1.tif', img_as_uint(i_bw_lambda1_mat))
        
def batch_segment_ph2ax(folder_name):
    extended_mat_file_name_list = find_tif_files(folder_name, '*.ome_extended.tif')
    no_files = len(extended_mat_file_name_list)
    ph2ax_intensity_row = array([])
    for i in range(no_files):
        i_extended_mat = imread(extended_mat_file_name_list[i])
        i_bw_lambda1_mat = imread(extended_mat_file_name_list[i][:-4] + '_bw_lambda1.tif') > 0
        ph2ax_intensity_row = append(ph2ax_intensity_row, i_extended_mat[:, :, 1][i_bw_lambda1_mat].flatten())
    ph2ax_threshold = threshold_otsu(ph2ax_intensity_row)
    print(ph2ax_threshold)
    for i in range(no_files):
        i_extended_mat = imread(extended_mat_file_name_list[i])
        i_bw_lambda2_mat = i_extended_mat[:, :, 1] >= ph2ax_threshold
        imsave(extended_mat_file_name_list[i][:-4] + '_bw_lambda2.tif', img_as_uint(i_bw_lambda2_mat))
        
def batch_segment_dna_rp(folder_name):
    extended_mat_file_name_list = find_tif_files(folder_name, '*.ome_extended.tif')
    no_files = len(extended_mat_file_name_list)
    dna_rp_intensity_row = array([])
    for i in range(no_files):
        i_extended_mat = imread(extended_mat_file_name_list[i])
        i_bw_lambda1_mat = imread(extended_mat_file_name_list[i][:-4] + '_bw_lambda1.tif') > 0
        dna_rp_intensity_row = append(dna_rp_intensity_row, i_extended_mat[:, :, 2][i_bw_lambda1_mat].flatten())
    dna_rp_threshold = threshold_otsu(dna_rp_intensity_row)
    print(dna_rp_threshold)
    for i in range(no_files):
        i_extended_mat = imread(extended_mat_file_name_list[i])
        i_bw_lambda3_mat = i_extended_mat[:, :, 2] >= dna_rp_threshold
        imsave(extended_mat_file_name_list[i][:-4] + '_bw_lambda3.tif', img_as_uint(i_bw_lambda3_mat))
        
def batch_get_ph2ax_regionprops(folder_name):
    extended_mat_file_name_list = find_tif_files(folder_name, '*.ome_extended.tif')
    no_files = len(extended_mat_file_name_list)
    ph2ax_flux_row = array([])
    for i in range(no_files):
        i_extended_mat = imread(extended_mat_file_name_list[i])
        i_bw_lambda1_mat = imread(extended_mat_file_name_list[i][:-4] + '_bw_lambda1.tif') > 0
        i_bw_lambda2_mat = imread(extended_mat_file_name_list[i][:-4] + '_bw_lambda2.tif') > 0
        label_lambda1_mat, no_labels = label(i_bw_lambda1_mat, return_num = True)
        for j in range(1, no_labels + 1):
            j_bw_lambda1_mat = i_bw_lambda1_mat == j
            j_ph2ax_regionprops_list = regionprops(label(i_bw_lambda2_mat & j_bw_lambda1_mat), i_extended_mat[:, :, 1])
            j_ph2ax_area_row = array([x.area for x in j_ph2ax_regionprops_list])
            j_ph2ax_intensity_row = array([x.mean_intensity for x in j_ph2ax_regionprops_list])
            j_ph2ax_flux = sum(j_ph2ax_area_row * j_ph2ax_intensity_row) / sum(i_extended_mat[:, :, 0][j_bw_lambda1_mat])
            ph2ax_flux_row = append(ph2ax_flux_row, j_ph2ax_flux)
    return ph2ax_flux_row

def batch_get_dna_rp_regionprops(folder_name):
    extended_mat_file_name_list = find_tif_files(folder_name, '*.ome_extended.tif')
    no_files = len(extended_mat_file_name_list)
    dna_rp_flux_row = array([])
    for i in range(no_files):
        i_extended_mat = imread(extended_mat_file_name_list[i])
        i_bw_lambda1_mat = imread(extended_mat_file_name_list[i][:-4] + '_bw_lambda1.tif') > 0
        i_bw_lambda3_mat = imread(extended_mat_file_name_list[i][:-4] + '_bw_lambda3.tif') > 0
        label_lambda1_mat, no_labels = label(i_bw_lambda1_mat, return_num = True)
        for j in range(1, no_labels + 1):
            j_bw_lambda1_mat = i_bw_lambda1_mat == j
            j_dna_rp_regionprops_list = regionprops(label(i_bw_lambda3_mat & j_bw_lambda1_mat), i_extended_mat[:, :, 2])
            j_dna_rp_area_row = array([x.area for x in j_dna_rp_regionprops_list])
            j_dna_rp_intensity_row = array([x.mean_intensity for x in j_dna_rp_regionprops_list])
            j_dna_rp_flux = sum(j_dna_rp_area_row * j_dna_rp_intensity_row) / sum(i_extended_mat[:, :, 0][j_bw_lambda1_mat])
            dna_rp_flux_row = append(dna_rp_flux_row, j_dna_rp_flux)
    return dna_rp_flux_row
    
def batch_standardize_intensity(folder_name):
    mm_stack_file_name_list = find_tif_files(folder_name, '*.ome.tif')
    no_files = len(mm_stack_file_name_list)
    min_intensity_mat = zeros((no_files, 4))
    max_intensity_mat = zeros((no_files, 4))
    mean_intensity_mat = zeros((no_files, 4))
    std_intensity_mat = zeros((no_files, 4))
    for i in range(no_files):
        i_mm_stack = imread(mm_stack_file_name_list[i])
        min_intensity_mat[i, 0] = min(i_mm_stack[0, :, :, :])
        min_intensity_mat[i, 1] = min(i_mm_stack[1, :, :, :])
        min_intensity_mat[i, 2] = min(i_mm_stack[2, :, :, :])
        min_intensity_mat[i, 3] = min(i_mm_stack[3, :, :, :])
        
        max_intensity_mat[i, 0] = max(i_mm_stack[0, :, :, :])
        max_intensity_mat[i, 1] = max(i_mm_stack[1, :, :, :])
        max_intensity_mat[i, 2] = max(i_mm_stack[2, :, :, :])
        max_intensity_mat[i, 3] = max(i_mm_stack[3, :, :, :])
    for i in range(no_files):
        i_mm_stack = imread(mm_stack_file_name_list[i])
        # Rescale.
        rescaled_lambda1_stack = i_mm_stack[0, :, :] - min(min_intensity_mat[:, 0])
        rescaled_lambda2_stack = i_mm_stack[1, :, :] - min(min_intensity_mat[:, 1])
        rescaled_lambda3_stack = i_mm_stack[2, :, :] - min(min_intensity_mat[:, 2])
        rescaled_lambda4_stack = i_mm_stack[3, :, :] - min(min_intensity_mat[:, 3])
        rescaled_lambda1_stack /= max(max_intensity_mat[:, 0])
        rescaled_lambda2_stack /= max(max_intensity_mat[:, 1])
        rescaled_lambda3_stack /= max(max_intensity_mat[:, 2])
        rescaled_lambda4_stack /= max(max_intensity_mat[:, 3])
        # Extend depth of field.
        extended_lambda1_mat = extend_depth_field(rescaled_lambda1_stack)
        extended_lambda2_mat = extend_depth_field(rescaled_lambda2_stack)
        extended_lambda3_mat = extend_depth_field(rescaled_lambda3_stack)
        extended_lambda4_mat = extend_depth_field(rescaled_lambda4_stack)
        # Apply gaussian blurr.
        gaussian_lambda1_mat = gaussian(extended_lambda1_mat, sigma = 2)
        gaussian_lambda2_mat = gaussian(extended_lambda2_mat, sigma = 2)
        gaussian_lambda3_mat = gaussian(extended_lambda3_mat, sigma = 2)
        gaussian_lambda4_mat = gaussian(extended_lambda4_mat, sigma = 2)
        
        # Subtract background.
        tophat_lambda2_mat = tophat_reconstruction(gaussian_lambda2_mat, 5)
        tophat_lambda3_mat = tophat_reconstruction(gaussian_lambda3_mat, 5)
        # Save images.
        lambda_stack = stack((gaussian_lambda1_mat, tophat_lambda2_mat, tophat_lambda3_mat, gaussian_lambda4_mat), axis = -1)
        imsave(mm_stack_file_name_list[i][:-4] + '_extended.tif', img_as_uint(lambda_stack))
        