from os import walk
from os.path import join
import fnmatch
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import closing, disk, erosion, opening, reconstruction, remove_small_objects, skeletonize
from skimage.filters import threshold_otsu
from numpy import array, mean, nan, stack, sum, zeros
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries, clear_border
from pandas import DataFrame, concat

MICRON_PER_PIXEL = 0.16
DIAMETER_THRESHOLD = 2.0

def find_mmstack_files(folder_name_str):
    mmstack_file_path_list = []
    search_str = '*_MMStack_Pos0.ome.tif'
    for (dir_name_str, sub_dir_name_list, file_name_list) in walk(folder_name_str):
        for i_file_name in file_name_list:
            if fnmatch.fnmatch(i_file_name, search_str):
                mmstack_file_path_list.append(join(dir_name_str, i_file_name))
    return mmstack_file_path_list

def load_stack(file_path_str):
    mm_stack = imread(file_path_str)
    npf_im = img_as_float(mm_stack[:, :, 0])
    pulse_im = img_as_float(mm_stack[:, :, 1])
    chase_im = img_as_float(mm_stack[:, :, 2])
    return npf_im, pulse_im, chase_im
    
def measure_tail_length(mm_stack):
    npf_im = img_as_float(mm_stack[:, :, 0])
    pulse_im = img_as_float(mm_stack[:, :, 1])
    chase_im = img_as_float(mm_stack[:, :, 2])
    snr_npf_im = compute_snr(npf_im)
    snr_pulse_im = compute_snr(pulse_im)
    snr_chase_im = compute_snr(chase_im)
    bw_chase_im = snr_chase_im > threshold_otsu(snr_chase_im)
    snr_composite_im = (snr_npf_im + snr_pulse_im + snr_chase_im) / 3
    snr_composite_im = opening(snr_composite_im, disk(6))
    bw_composite_im = snr_composite_im > threshold_otsu(snr_composite_im)
    skel_composite_im = skeletonize(bw_composite_im, method = 'lee')
    bw_tail_im = skel_composite_im & bw_chase_im
    tail_props_list = regionprops(label(bw_tail_im))
    if len(tail_props_list) > 0:
        tail_length = tail_props_list[0].area * MICRON_PER_PIXEL
        return tail_length
    else:
        return nan

def save_segmentation(file_path_str, gray_im, bw_stack):
    gray_im /= gray_im.max()
    rgb_stack = stack((gray_im, gray_im, gray_im), axis = -1)
    bw_stack[:, :, 1] = find_boundaries(bw_stack[:, :, 1])
    bw_stack[:, :, 2] = find_boundaries(bw_stack[:, :, 2])
    rgb_stack[bw_stack[:, :, 1], 0] = 1
    rgb_stack[bw_stack[:, :, 1], 1] = 0
    rgb_stack[bw_stack[:, :, 1], 2] = 0
    rgb_stack[bw_stack[:, :, 2], 0] = 0
    rgb_stack[bw_stack[:, :, 2], 1] = 1
    rgb_stack[bw_stack[:, :, 2], 2] = 0
    rgb_stack[bw_stack[:, :, 3], 0] = 0
    rgb_stack[bw_stack[:, :, 3], 1] = 0
    rgb_stack[bw_stack[:, :, 3], 2] = 1
    imsave(file_path_str[:-21] + '_segmentation.jpg', img_as_ubyte(rgb_stack))
    
def segment_npf(npf_im):
    npf_closed_im = closing(npf_im, disk(3))
    npf_opened_im = opening(npf_closed_im, disk(3))
    npf_bw_im = npf_opened_im > threshold_otsu(npf_opened_im)
    npf_bw_im = remove_small_objects(npf_bw_im, min_size = sum(disk(6)))
    npf_bw_im = clear_border(npf_bw_im ^ erosion(npf_bw_im, disk(3)))
    return npf_bw_im

def segment_actin(actin_im):
    actin_closed_im = closing(actin_im, disk(3))
    actin_opened_im = opening(actin_closed_im, disk(3))
    actin_bw_im = clear_border(actin_opened_im > (actin_opened_im.mean() + 3 * actin_opened_im.std()))
    return binary_fill_holes(actin_bw_im)
    
def measure_comet_tail_props(npf_im, pulse_im, chase_im):
    npf_bw_im = segment_npf(npf_im)
    pulse_bw_im = segment_actin(pulse_im)
    chase_bw_im = segment_actin(chase_im)
    npf_back_mean = mean(npf_im[~npf_bw_im])
    pulse_back_mean = mean(pulse_im[~pulse_bw_im])
    chase_back_mean = mean(chase_im[~chase_bw_im])
    composite_bw_im = binary_fill_holes(npf_bw_im | pulse_bw_im | chase_bw_im)
    npf_label_im, no_beads = label(npf_bw_im, return_num = True)
    npf_fluor_row = zeros(no_beads)
    actin_fluor_row = zeros(no_beads)
    tail_length_row = zeros(no_beads)
    tail_bw_im = zeros(npf_im.shape, dtype = bool)
    for i in range(no_beads):
        i_npf_bw_im = (npf_bw_im == (i + 1))
        npf_fluor_row[i] = mean(npf_im[i_npf_bw_im]) - npf_back_mean
        i_pulse_bw_im = reconstruction(i_npf_bw_im & pulse_bw_im, pulse_bw_im).astype(bool)
        i_chase_bw_im = reconstruction(i_npf_bw_im & chase_bw_im, chase_bw_im).astype(bool)
        i_composite_bw_im = reconstruction(i_npf_bw_im & composite_bw_im, composite_bw_im).astype(bool)
        i_composite_skel_im = skeletonize(erosion(i_composite_bw_im, disk(3)), method = 'lee')
        if i_pulse_bw_im.sum() > i_chase_bw_im.sum():
            i_tail_bw_im = i_composite_skel_im & i_chase_bw_im
            actin_fluor_row[i] = mean(chase_im[i_chase_bw_im]) - chase_back_mean
        else:
            i_tail_bw_im = i_composite_skel_im & i_pulse_bw_im
            actin_fluor_row[i] = mean(pulse_im[i_pulse_bw_im]) - pulse_back_mean
        i_tail_props_list = regionprops(label(i_tail_bw_im))
        if len(i_tail_props_list) == 0:
            tail_length_row[i] = nan
        else:
            tail_length_row[i] = i_tail_props_list[0].area * MICRON_PER_PIXEL
            tail_bw_im[i_tail_bw_im == True] = True
    comet_tail_props_df = DataFrame.from_dict({'npf_fluor': npf_fluor_row * (2**16 - 1),
                                               'actin_fluor': actin_fluor_row * (2**16 -1),
                                               'tail_length': tail_length_row})
    bw_stack = stack((npf_bw_im, pulse_bw_im, chase_bw_im, tail_bw_im), axis = -1)
    return comet_tail_props_df, bw_stack
    
def batch_analysis(folder_path_str):
    comet_tail_props_df = DataFrame({'npf_fluor': array([]), 'actin_fluor': array([]), 'tail_length': array([])})
    mmstack_file_path_list = find_mmstack_files(folder_path_str)
    no_files = len(mmstack_file_path_list)
    for i in range(no_files):
        i_npf_im, i_pulse_im, i_chase_im = load_stack(mmstack_file_path_list[i])
        i_comet_tail_props_df, i_bw_stack = measure_comet_tail_props(i_npf_im, i_pulse_im, i_chase_im)
        i_comet_tail_props_df['file_path'] = mmstack_file_path_list[i]
        comet_tail_props_df = concat([comet_tail_props_df, i_comet_tail_props_df])
        save_segmentation(mmstack_file_path_list[i], i_chase_im, i_bw_stack)
    comet_tail_props_df = comet_tail_props_df.reset_index()
    comet_tail_props_df.to_csv('comet_tail_properties.csv')
    return comet_tail_props_df