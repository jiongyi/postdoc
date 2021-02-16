from os import walk
from os.path import join, basename
import fnmatch
from skimage.io import imread, imsave
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import dilation, disk, erosion, skeletonize, reconstruction, remove_small_objects, opening, white_tophat, medial_axis
from skimage.graph import route_through_array
from skan import skeleton_to_csgraph
from skimage.filters import gaussian, hessian, threshold_isodata, threshold_otsu, threshold_multiotsu
from skimage.feature import canny
from skimage.exposure import equalize_adapthist
from skimage.util import invert, img_as_ubyte
from skimage.color import label2rgb
from numpy import mean, abs, append, array, convolve, delete, hstack, max, ones, unravel_index, zeros, gradient, percentile, stack
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, find_boundaries, morphological_chan_vese
from pandas import DataFrame, read_csv
from matplotlib.pyplot import subplots, savefig, ioff, close, tight_layout
from seaborn import catplot
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import rank_filter
from scipy.signal import savgol_filter

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
    return mm_stack[:, :, 0], mm_stack[:, :, 1], mm_stack[:, :, 2]


def open_close(raw_im, radius):
    eroded_im = erosion(raw_im, disk(radius))
    opened_im = reconstruction(eroded_im, raw_im)
    dilated_im = dilation(opened_im, disk(radius))
    closed_im = invert(reconstruction(invert(dilated_im), invert(opened_im)))
    return closed_im
    
    
def make_composite(c0_im, c1_im, c2_im):
    c0_norm_im = c0_im / threshold_isodata(c0_im)
    c1_norm_im = c1_im / threshold_isodata(c1_im)
    c2_norm_im = c2_im / threshold_isodata(c2_im)
    composite_im = mean(stack((c0_norm_im, c1_norm_im, c2_norm_im), axis = -1), axis = -1)
    return composite_im
    
    
def segment_composite(composite_im):
    closed_im = open_close(composite_im, 6)
    composite_bw_im = closed_im > max([threshold_isodata(closed_im), closed_im.mean() + 3 * closed_im.std()])
    return composite_bw_im


def segment_bead(raw_im):
    closed_im = open_close(raw_im, 6)
    bw1_im = closed_im > threshold_isodata(closed_im)
    bw2_im = binary_fill_holes(bw1_im)
    bw3_im = clear_border(bw2_im)
    bead_bw_im = remove_small_objects(bw3_im, min_size=disk(6).sum())
    return bead_bw_im
    
    
def skeletonize_alternate(bead_bw_im, comet_bw_im):
    bead_props_list = regionprops(label(bead_bw_im))
    bead_center_row, bead_center_col = bead_props_list[0].centroid
    bead_center_row = round(bead_center_row)
    bead_center_col = round(bead_center_col)
    bead_center_bw_im = ones(bead_bw_im.shape, dtype = bool)
    bead_center_bw_im[bead_center_row, bead_center_col] = False
    distance_im = distance_transform_edt(bead_center_bw_im)
    distance_im *= comet_bw_im
    max_distance = distance_im.max()
    end_bw_im = distance_im == max_distance
    end_row, end_col = unravel_index(distance_im.argmax(), distance_im.shape)
    distance_im = distance_transform_edt(comet_bw_im)
    cost_im = distance_im.max() - distance_im
    path_list, _ = route_through_array(cost_im, \
                                       (bead_center_row, bead_center_col), \
                                       (end_row, end_col), \
                                       fully_connected = False, \
                                       geometric = True)
    skel_bw_im = zeros(bead_bw_im.shape, dtype = bool)
    for i in range(len(path_list)):
        skel_bw_im[path_list[i][0], path_list[i][1]] = True
    return skel_bw_im

def find_longest_path(bead_bw_im, composite_bw_im):
    pre_skel_bw_im = skeletonize(composite_bw_im)
    skel_bw_im = pre_skel_bw_im & ~bead_bw_im
    if skel_bw_im.sum() > 2:
        _, _, degrees_mat = skeleton_to_csgraph(skel_bw_im)
        ends_bw_im = degrees_mat == 1
        no_ends = ends_bw_im.sum()
        if no_ends > 1:
            # Index skeleton end nearest to bead center.
            bead_props_list = regionprops(label(bead_bw_im))
            bead_center_row, bead_center_col = bead_props_list[0].centroid
            ends_row_row, ends_col_row = ends_bw_im.nonzero()
            ends2bead_distance_row = (ends_row_row - bead_center_row)**2 + \
                                     (ends_col_row - bead_center_col)**2
            index_nearest_end = ends2bead_distance_row.argmin()
            start_tup = (ends_row_row[index_nearest_end], \
                             ends_col_row[index_nearest_end])
            other_ends_row_row = delete(ends_row_row, index_nearest_end)
            other_ends_col_row = delete(ends_col_row, index_nearest_end)
            distance_mat = distance_transform_edt(composite_bw_im)
            cost_mat = distance_mat.max() - distance_mat
            if no_ends > 2:
                no_other_ends = other_ends_col_row.size
                cost_row = zeros(no_other_ends)
                path_list = []
                for i in range(no_other_ends):
                    i_end_tup = (other_ends_row_row[i], other_ends_col_row[i])
                    i_path_list, cost_row[i] = route_through_array(cost_mat, \
                                                                   start_tup, \
                                                                   i_end_tup, \
                                                                   fully_connected = False, \
                                                                   geometric = True)
                    path_list.append(i_path_list)
                index_highest_cost = cost_row.argmax()
                longest_path = path_list[index_highest_cost]                                        
            else:
                end_tup = (other_ends_row_row[0], other_ends_col_row[0])
                longest_path, _ = route_through_array(cost_mat, \
                                                      start_tup, \
                                                      end_tup, \
                                                      fully_connected = False, \
                                                      geometric = True)
        else:
            longest_path = ([])
    else:
        longest_path = ([])
    return longest_path
    
    
def find_step(c1_fluor_row, c2_fluor_row):
    c1_float_row = c1_fluor_row.astype(float)
    c1_float_row -= c1_float_row.min()
    c1_float_row /= c1_float_row.max()
    c2_float_row = c2_fluor_row.astype(float)
    c2_float_row -= c2_float_row.min()
    c2_float_row /= c2_float_row.max()
    no_points = c1_float_row.size
    kernel_row = hstack((ones(no_points), -1 * ones(no_points)))
    c1_convolved_row = convolve(c1_float_row - c1_float_row.mean(), kernel_row, mode = 'valid')
    c2_convolved_row = convolve(c2_float_row - c2_float_row.mean(), kernel_row, mode = 'valid')
    c1_max = abs(c1_convolved_row).max()
    c2_max = abs(c2_convolved_row).max()
    if c1_max > c2_max:
        if c1_max > 0.2:
            index_step = abs(c1_convolved_row).argmax()
        else:
            index_step = no_points
    elif c2_max > 0.2:
        index_step = abs(c2_convolved_row).argmax()
    else:
        index_step = no_points
    return index_step

def estimate_comet_tail_props(file_path_str, save_scan = False):
    # Load stack.
    c0_im, c1_im, c2_im = load_stack(file_path_str)
    # Segment channels.
    bead_bw_im = segment_bead(c0_im)
    actin1_bw_im = segment_actin(c1_im, bead_bw_im)
    actin2_bw_im = segment_actin(c2_im, bead_bw_im)
    # Process each tail.
    bead_label_im, no_beads = label(bead_bw_im, return_num = True)
    if no_beads > 0:
        # Extract mean background fluorescence values.
        bead_mean_back_fluor = c0_im[~bead_bw_im].mean()
        actin1_mean_back_fluor = c1_im[~actin1_bw_im].mean()
        actin2_mean_back_fluor = c2_im[~actin2_bw_im].mean()
        # Set up property arrays.
        npf_fluor_row = zeros(no_beads)
        chase_fluor_row = zeros(no_beads)
        chase_tail_length_row = zeros(no_beads)
        chase_channel_name_row = zeros(no_beads)
        pulse_fluor_row = zeros(no_beads)
        pulse_tail_length_row = zeros(no_beads)
        npf_chase_tail_bw_im = zeros(c0_im.shape, dtype=bool)
        composite_bw_im = actin1_bw_im | actin2_bw_im
        for i in range(no_beads):
            i_bead_bw_im = bead_label_im == (i + 1)
            # NPF fluorescence
            i_npf_bw_im = i_bead_bw_im & ~erosion(i_bead_bw_im, disk(3))
            npf_fluor_row[i] = c0_im[i_npf_bw_im].mean() - bead_mean_back_fluor
            # Comet tail properties
            i_composite_bw_im = reconstruction(dilation(i_bead_bw_im, disk(3)) & composite_bw_im, 
                                               composite_bw_im).astype(bool)
            if i_composite_bw_im.sum() > 0:
                i_composite_props_list = regionprops(label(i_composite_bw_im))
                i_axis_ratio = i_composite_props_list[0].major_axis_length / \
                               i_composite_props_list[0].minor_axis_length
                if i_axis_ratio > 2:
                    i_skeleton_bw_im = skeletonize_comet(i_bead_bw_im, i_composite_bw_im)
                    i_sorted_row_row, i_sorted_col_row = sort_pixel_coords(i_bead_bw_im, 
                                                                           i_skeleton_bw_im)
                    i_ds_row = (((i_sorted_row_row[1:] - i_sorted_row_row[:-1])**2 + \
                                 (i_sorted_col_row[1:] - i_sorted_col_row[:-1])**2)**0.5) * 0.16
                    i_distance_row = append([0], i_ds_row.cumsum())
                    c1_path_bw_im = actin1_bw_im & i_skeleton_bw_im
                    c2_path_bw_im = actin2_bw_im & i_skeleton_bw_im
                    if c1_path_bw_im.sum() < c2_path_bw_im.sum():
                        chase_channel_name_row[i] = 1
                        chase_fluor_row[i] = c1_im[c1_path_bw_im].mean() - actin1_mean_back_fluor
                        chase_tail_length_row[i] = i_distance_row[c1_path_bw_im.sum() - 1]
                        pulse_fluor_row[i] = c2_im[c1_path_bw_im].mean() - actin2_mean_back_fluor
                        pulse_tail_length_row[i] = i_distance_row[-1] - chase_tail_length_row[i]
                        i_chase_im = equalize_adapthist(c1_im)
                    else:
                        chase_channel_name_row[i] = 2
                        chase_fluor_row[i] = c2_im[c2_path_bw_im].mean() - actin2_mean_back_fluor
                        chase_tail_length_row[i] = i_distance_row[c2_path_bw_im.sum() - 1]
                        pulse_fluor_row[i] = c1_im[c2_path_bw_im].mean() - actin1_mean_back_fluor
                        pulse_tail_length_row[i] = i_distance_row[-1] - chase_tail_length_row[i]
                        i_chase_im = equalize_adapthist(c2_im)
                    if save_scan == True:
                        ioff()
                        fig_hand, axes_hand = subplots()
                        axes_hand.plot(i_distance_row, c1_im[i_sorted_row_row, i_sorted_col_row], 'red')
                        axes_hand.plot(i_distance_row, c2_im[i_sorted_row_row, i_sorted_col_row], 'blue')
                        axes_hand.plot([chase_tail_length_row[i], chase_tail_length_row[i]], \
                                       [0, axes_hand.get_ylim()[1]], 'black')
                        axes_hand.set_xlabel("Distance from bead ($\mu$m)", fontsize = 8)
                        axes_hand.set_ylabel("Fluorescence intensity", fontsize = 8)
                        savefig(file_path_str[:-21] + '_linescan.png', facecolor = 'white', dpi = 300)
                        close()
                    imsave(file_path_str[:-21] + '_skel_segment.jpg', \
                           img_as_ubyte(label2rgb(label(i_skeleton_bw_im), \
                                                  image = invert(i_chase_im), \
                                                  bg_label = 0, colors = ['yellow'])), check_contrast = False)
            else:
                print("No actin tail.")
            comet_tail_props_df = DataFrame.from_dict({'npf_fluor': npf_fluor_row,
                                                   'chase_fluor': chase_fluor_row,
                                                   'chase_tail_length': chase_tail_length_row,
                                                   'chase_channel_name': chase_channel_name_row, 
                                                   'pulse_fluor': pulse_fluor_row, 
                                                   'pulse_tail_length': pulse_tail_length_row, 
                                                   'file_path_str': basename(file_path_str)})
    else:
        print("No beads found")
        comet_tail_props_df = DataFrame.from_dict({'npf_fluor': array([]),
                                                   'chase_fluor': array([]),
                                                   'chase_tail_length': array([]),
                                                   'chase_channel_name': array([]),
                                                   'pulse_fluor': array([]), 
                                                   'pulse_tail_length': array([]),
                                                   'file_path_str': basename(file_path_str)})
    return comet_tail_props_df
    
def comet_tail_props(file_path_str, save_segmentation):
    c0_im, c1_im, c2_im = load_stack(file_path_str)
    bead_bw_im = segment_bead(c0_im)
    composite_im = make_composite(c0_im, c1_im, c2_im)
    composite_bw_im = segment_composite(composite_im)
    bead_label_im, no_beads = label(bead_bw_im, return_num = True)
    path_bw_im = zeros(bead_bw_im.shape, dtype = bool)
    if save_segmentation == True:
        imsave(file_path_str[:-21] + '_c0_segment.jpg', \
               img_as_ubyte(label2rgb(label(find_boundaries(bead_bw_im)), \
                            image = equalize_adapthist(invert(c0_im)), \
                            bg_label = 0, colors = ['green'])), \
                            check_contrast = False)
        imsave(file_path_str[:-21] + '_c1_segment.jpg', \
               img_as_ubyte(label2rgb(label(find_boundaries(composite_bw_im)), \
                            image = equalize_adapthist(invert(c1_im)), \
                            bg_label = 0, colors = ['red'])), \
                            check_contrast = False)
        imsave(file_path_str[:-21] + '_c2_segment.jpg', \
               img_as_ubyte(label2rgb(label(find_boundaries(composite_bw_im)), \
                            image = equalize_adapthist(invert(c2_im)), \
                            bg_label = 0, colors = ['magenta'])), \
                            check_contrast = False)
    if no_beads > 0:
        c0_mean_back_fluor = c0_im[~bead_bw_im].mean()
        c1_mean_back_fluor = c1_im[~composite_bw_im].mean()
        c2_mean_back_fluor = c2_im[~composite_bw_im].mean()
        # Set up property arrays.
        npf_fluor_row = zeros(no_beads)
        chase_fluor_row = zeros(no_beads)
        chase_tail_length_row = zeros(no_beads)
        chase_channel_name_row = zeros(no_beads)
        pulse_fluor_row = zeros(no_beads)
        pulse_tail_length_row = zeros(no_beads)
        for i in range(no_beads):
            i_bead_bw_im = bead_label_im == (i + 1)
            i_npf_bw_im = i_bead_bw_im & ~erosion(i_bead_bw_im, disk(3))
            npf_fluor_row[i] = c0_im[i_npf_bw_im].mean() - c0_mean_back_fluor
            i_composite_bw_im = reconstruction(i_bead_bw_im & composite_bw_im, \
                                               composite_bw_im).astype(bool)
            i_longest_path = find_longest_path(i_bead_bw_im, i_composite_bw_im)
            i_path_size = len(i_longest_path)
            if i_path_size > 0:
                i_path_row_row = array([i_longest_path[k][0] for k in range(i_path_size)])
                i_path_col_row = array([i_longest_path[k][1] for k in range(i_path_size)])
                i_ds_row = (gradient(i_path_row_row)**2 + gradient(i_path_col_row)**2)**0.5
                i_distance_row = i_ds_row.cumsum() * 0.16
                i_index_step = find_step(c1_im[i_path_row_row, i_path_col_row], \
                                         c2_im[i_path_row_row, i_path_col_row])
                chase_tail_length_row[i] = i_distance_row[i_index_step]
                pulse_tail_length_row[i] = i_distance_row[-1] - chase_tail_length_row[i]
                c1_first_fluor_mean = c1_im[i_path_row_row[:i_index_step], i_path_col_row[:i_index_step]].mean() - c1_mean_back_fluor
                c1_second_fluor_mean = c1_im[i_path_row_row[i_index_step:], i_path_col_row[i_index_step:]].mean() - c1_mean_back_fluor
                c2_first_fluor_mean = c2_im[i_path_row_row[:i_index_step], i_path_col_row[:i_index_step]].mean() - c2_mean_back_fluor
                c2_second_fluor_mean = c2_im[i_path_row_row[i_index_step:], i_path_col_row[i_index_step:]].mean() - c2_mean_back_fluor
                if c1_first_fluor_mean > c1_second_fluor_mean:
                    chase_channel_name_row[i] = 1
                    chase_fluor_row[i] = c1_first_fluor_mean
                    pulse_fluor_row[i] = c2_first_fluor_mean
                else:
                    chase_channel_name_row[i] = 2
                    chase_fluor_row[i] = c2_first_fluor_mean
                    pulse_fluor_row[i] = c2_second_fluor_mean
                ioff()
                fig_hand, axes_hand = subplots()
                axes_hand.plot(i_distance_row, c1_im[i_path_row_row, i_path_col_row], 'red')
                axes_hand.plot(i_distance_row, c2_im[i_path_row_row, i_path_col_row], 'blue')
                axes_hand.plot([i_distance_row[i_index_step], i_distance_row[i_index_step]], \
                                       [0, axes_hand.get_ylim()[1]], 'black')
                savefig(file_path_str[:-21] + '_linescan.png', facecolor = 'white', dpi = 300)
                close()
                for j in range(i_index_step):
                    path_bw_im[i_longest_path[j]] = True
                
        save1_im = invert(composite_im / composite_im.max())
        overlay_im = label2rgb(label(path_bw_im), \
                               image = save1_im, \
                               bg_label = 0, colors = ['yellow'])
        imsave(file_path_str[:-21] + '_path.jpg', img_as_ubyte(overlay_im), \
               check_contrast = False)
        comet_tail_props_df = DataFrame.from_dict({'npf_fluor': npf_fluor_row,
                                                   'chase_fluor': chase_fluor_row,
                                                   'chase_tail_length': chase_tail_length_row,
                                                   'chase_channel_name': chase_channel_name_row, 
                                                   'pulse_fluor': pulse_fluor_row, 
                                                   'pulse_tail_length': pulse_tail_length_row, 
                                                   'file_path_str': basename(file_path_str)})
    else:
        print("No beads found")
        comet_tail_props_df = DataFrame.from_dict({'npf_fluor': array([]),
                                                   'chase_fluor': array([]),
                                                   'chase_tail_length': array([]),
                                                   'chase_channel_name': array([]),
                                                   'pulse_fluor': array([]), 
                                                   'pulse_tail_length': array([]),
                                                   'file_path_str': basename(file_path_str)})
    return comet_tail_props_df
            
    

def batch_analysis(folder_path_str, save_segmentation = False):
    mmstack_file_path_list = find_mmstack_files(folder_path_str)
    no_files = len(mmstack_file_path_list)
    comet_tail_props_df = DataFrame.from_dict({'npf_fluor': array([]),
                                               'chase_fluor': array([]),
                                               'chase_tail_length': array([]),
                                               'chase_channel_name': array([]),
                                               'pulse_fluor': array([]), 
                                               'pulse_tail_length': array([]),
                                               'file_path_str': array([])})
    for i in range(no_files):
        i_comet_tail_props_df = comet_tail_props(mmstack_file_path_list[i], save_segmentation)
        comet_tail_props_df = comet_tail_props_df.append(i_comet_tail_props_df)
    comet_tail_props_df = comet_tail_props_df.sort_values(by = ['pulse_tail_length'])
    comet_tail_props_df.to_csv(folder_path_str + '/comet_tail_properties.csv')
    return comet_tail_props_df


def plot_mean_npf_fluor(df_file_path_row):
    no_files = df_file_path_row.size
    median_fluor_row = zeros(no_files)
    up_bound_row = zeros(no_files)
    low_bound_row = zeros(no_files)
    for i in range(no_files):
        i_df = read_csv(df_file_path_row[i])
        median_fluor_row[i] = i_df['npf_fluor'].mean()
        i_percentile_row = percentile(i_df['npf_fluor'], (25, 75))
        low_bound_row[i] = i_percentile_row[0]
        up_bound_row[i] = i_percentile_row[1]
    yerr_mat = stack((median_fluor_row - low_bound_row, up_bound_row - median_fluor_row), axis = -1).T
    fig_hand, axes_hand = subplots()
    fig_hand.set_facecolor('white')
    axes_hand.set_ylim([0, up_bound_row.max() * 1.1])
    axes_hand.errorbar(array([25, 75, 125, 175, 225]), median_fluor_row, yerr = yerr_mat, marker = '.', markersize = 16)
    axes_hand.set_xlabel('Capping protein (nM)', fontsize = 12)
    axes_hand.set_ylabel('Mean EGFP fluorescence', fontsize = 12)
    tight_layout()
    return fig_hand, axes_hand
    
def plot_median_chase_fluor(df_file_path_row):
    no_files = df_file_path_row.size
    c1_median_fluor_row = zeros(no_files)
    c1_up_bound_row = zeros(no_files)
    c1_low_bound_row = zeros(no_files)
    c2_median_fluor_row = zeros(no_files)
    c2_up_bound_row = zeros(no_files)
    c2_low_bound_row = zeros(no_files)
    for i in range(no_files):
        i_df = read_csv(df_file_path_row[i])
        i_is_comet_row = i_df['pulse_tail_length'] > 1
        i_is_c1_row = i_df['chase_channel_name'] == 1
        i_is_c2_row = i_df['chase_channel_name'] == 2
        i_c1_fluor_row = i_df['chase_fluor'][i_is_comet_row & i_is_c1_row]
        i_c2_fluor_row = i_df['chase_fluor'][i_is_comet_row & i_is_c2_row]
        i_c1_percentile_row = percentile(i_c1_fluor_row, (25, 50, 75))
        c1_median_fluor_row[i] = i_c1_percentile_row[1]
        c1_low_bound_row[i] = i_c1_percentile_row[0]
        c1_up_bound_row[i] = i_c1_percentile_row[2]
        i_c2_percentile_row = percentile(i_c2_fluor_row, (25, 50, 75))
        c2_low_bound_row[i] = i_c2_percentile_row[0]
        c2_median_fluor_row[i] = i_c2_percentile_row[1]
        c2_up_bound_row[i] = i_c2_percentile_row[2]
    c1_yerr_mat = stack((c1_median_fluor_row - c1_low_bound_row, c1_up_bound_row - c1_median_fluor_row), axis = -1).T
    c2_yerr_mat = stack((c2_median_fluor_row - c2_low_bound_row, c2_up_bound_row - c2_median_fluor_row), axis = -1).T
    fig_hand, axes_hand = subplots()
    fig_hand.set_facecolor('white')
    axes_hand.set_ylim([0, 1.1 * max([c1_up_bound_row.max(), c2_up_bound_row.max()])])
    axes_hand.errorbar(array([25, 75, 125, 175, 225]), c1_median_fluor_row, yerr = c1_yerr_mat, marker = '.', markersize = 16, label = 'Hylite-555')
    axes_hand.errorbar(array([25, 75, 125, 175, 225]), c2_median_fluor_row, yerr = c2_yerr_mat, marker = '.', markersize = 16, label = 'Alexa-647')
    axes_hand.tick_params(labelsize = 12)
    axes_hand.set_xlabel('Capping protein (nM)', fontsize = 12)
    axes_hand.set_ylabel('Median actin fluorescence', fontsize = 12)
    axes_hand.legend(fontsize = 12)
    tight_layout()
    return fig_hand, axes_hand
    
def plot_median_tail_length(df_file_path_row, chase_minutes = 1.0):
    no_files = df_file_path_row.size
    median_tail_length_row = zeros(no_files)
    up_bound_row = zeros(no_files)
    low_bound_row = zeros(no_files)
    for i in range(no_files):
        i_df = read_csv(df_file_path_row[i])
        i_is_comet_row = i_df['pulse_tail_length'] > 1
        #i_tail_length_row = (i_df['chase_tail_length'][i_is_comet_row] + i_df['pulse_tail_length'][i_is_comet_row]) / chase_minutes
        i_tail_length_row = i_df['chase_tail_length'][i_is_comet_row] / \
                            chase_minutes
        i_percentile_row = percentile(i_tail_length_row, (25, 50, 75))
        median_tail_length_row[i] = i_percentile_row[1]
        low_bound_row[i] = i_percentile_row[0]
        up_bound_row[i] = i_percentile_row[2]     
    yerr_mat = stack((median_tail_length_row - low_bound_row, up_bound_row - median_tail_length_row), axis = -1).T
    fig_hand, axes_hand = subplots()
    fig_hand.set_facecolor('white')
    axes_hand.set_ylim([0, 1.1 * up_bound_row.max()])
    axes_hand.errorbar(array([25, 75, 125, 175, 225]), median_tail_length_row, yerr = yerr_mat, marker = '.', markersize = 16)
    axes_hand.tick_params(labelsize = 12)
    axes_hand.set_xlabel('Capping protein (nM)', fontsize = 12)
    axes_hand.set_ylabel('Median growth rate ($\mu$m/min)', fontsize = 12)
    tight_layout()
    return fig_hand, axes_hand

def box_plot_tail_length(df_file_path_row, chase_minutes = 1.0):
    no_files = df_file_path_row.size
    tail_length_df = DataFrame.from_dict({'25': array([]),
                                      '75': array([]),
                                      '125': array([]),
                                      '175': array([]),
                                      '225': array([])})
    for i in range(no_files):
        i_df = read_csv(df_file_path_row[i])
        is_comet_tail_row = i_df['pulse_tail_length'] > 1.0
        i_tail_length_row = i_df['chase_tail_length'][is_comet_tail_row] / \
                            chase_minutes
        tail_length_df[tail_length_df.columns[i]] = i_tail_length_row
    tail_length_plot_obj = catplot(data = tail_length_df, kind = 'swarm')
    tail_length_plot_obj.set_axis_labels("Capping protein (nM)", "Network growth rate ($\mu$m/min)", fontsize = 16)
    tail_length_plot_obj.ax.tick_params(labelsize = 12)
    tight_layout()
    return tail_length_plot_obj
