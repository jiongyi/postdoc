from os import walk
from os.path import join, split
import fnmatch
from numpy import zeros, argmax, arange, max, stack, mean, std
from skimage.filters import sobel, gaussian
from skimage import img_as_float, img_as_uint
from skimage.io import imread, imsave
from skimage.filters import threshold_isodata
from skimage.feature import canny
from skimage.morphology import disk, erosion, reconstruction

def find_tif_files(folder_name_str):
    tif_file_paths_str = []
    search_str = '*ome.tif'
    for (dir_path_str, dir_names_str, file_names_str) in walk(folder_name_str):
        for i_file_names_str in file_names_str:
            if fnmatch.fnmatch(i_file_names_str, search_str):
                tif_file_paths_str.append(join(dir_path_str, i_file_names_str))
    return tif_file_paths_str

def extended_depth_field(stack_xyz):
    no_slices, no_rows, no_cols = stack_xyz.shape
    sobel_xyz = zeros((no_slices, no_rows, no_cols))
    for i in range(no_slices):
        sobel_xyz[i, :, :] = sobel(stack_xyz[i, :, :])
    idx_max_xy = argmax(sobel_xyz, axis = 0)
    stack_xyz = stack_xyz.reshape((no_slices, -1))
    stack_xyz = stack_xyz.transpose()
    ext_depth_field_xy = stack_xyz[arange(len(stack_xyz)), idx_max_xy.ravel()]
    ext_depth_field_xy = ext_depth_field_xy.reshape((no_rows, no_cols))
    gaussian_xy = gaussian(ext_depth_field_xy)
    return gaussian_xy

def z_project(tif_file_path_str):
    wzxy = img_as_float(imread(tif_file_path_str))
    no_channels, no_slices, no_rows, no_cols = wzxy.shape
    z_project_wxy = zeros((no_channels, no_rows, no_cols))
    for i in range(no_channels):
        i_channel_xyz = wzxy[i, :, :]
        i_z_project_xy = max(i_channel_xyz, axis = 0)
        
        z_project_wxy[i] = i_z_project_xy
    return z_project_wxy

def open_reconstruction(raw_xy, struct_xy):
    eroded_xy = erosion(raw_xy, struct_xy)
    opened_xy = reconstruction(eroded_xy, raw_xy, selem = struct_xy)
    return opened_xy

def binarize_data(tif_file_path_str, wxy):
    no_channels, no_rows, no_cols = wxy.shape
    bw_wxy = zeros((no_channels, no_rows, no_cols), dtype = bool)
    for i in range(no_channels):
        i_gaussian_xy = gaussian(wxy[i], sigma = 1)
        i_opened_xy = open_reconstruction(i_gaussian_xy, disk(4))
        bw_wxy[i] = canny(wxy[i] / max(wxy[i]), sigma = 2)
        i_rgb_xy = stack((wxy[i], wxy[i], wxy[i]))
        i_rgb_xy[:, bw_wxy[i]] = 0.0
        i_rgb_xy[i, bw_wxy[i]] = 1.0
        imsave(tif_file_path_str[:-4] + '_C' + str(i) + '_z_project.tif', img_as_uint(i_rgb_xy))
    return bw_wxy
    
def batch_analysis(folder_name_str):
    tif_file_paths_str = find_tif_files(folder_name_str)
    no_files = len(tif_file_paths_str)
    for i in range(no_files):
        i_z_project_wxy = z_project(tif_file_paths_str[i])
        i_bw_wxy = binarize_data(tif_file_paths_str[i], i_z_project_wxy)
        
        