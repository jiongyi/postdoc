from os import walk
from os.path import join, basename
import fnmatch
from skimage.io import imread, imsave
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import closing, disk, square, erosion, dilation, opening, reconstruction, remove_small_objects, \
    medial_axis, thin, skeletonize
from skimage.graph import route_through_array
from skan import skeleton_to_csgraph
from skimage.filters import gaussian, hessian, threshold_isodata, threshold_otsu
from skimage.exposure import equalize_adapthist
from skimage.util import invert
from skimage.color import label2rgb
from numpy import array, max, stack, zeros, delete, append, ones, abs, convolve, hstack, unravel_index
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, find_boundaries
from pandas import DataFrame
from matplotlib.pyplot import subplots, savefig

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


def segment_bead(raw_im):
    gaussian_im = gaussian(raw_im, sigma = 1)
    bw1_im = gaussian_im > threshold_otsu(gaussian_im)
    bw2_im = binary_fill_holes(bw1_im)
    bw3_im = clear_border(bw2_im)
    bead_bw_im = remove_small_objects(bw3_im, min_size=disk(6).sum())
    return bead_bw_im


def segment_actin(raw_im, bead_bw_im):
    equal_im = equalize_adapthist(raw_im)
    gauss_im = gaussian(equal_im, sigma = 6)
    hessian_im = hessian(gauss_im, sigmas = range(1, 10, 2), mode = 'constant')
    bw1_im = hessian_im > threshold_isodata(hessian_im)
    bw2_im = binary_fill_holes(bw1_im)
    bw3_im = clear_border(bw2_im)
    bw4_im = remove_small_objects(bw3_im, min_size = disk(6).sum())
    actin_bw_im = reconstruction(dilation(bead_bw_im, disk(3)) & bw4_im, bw4_im).astype(bool)
    return actin_bw_im


def segment_comet(c0_im, c1_im, c2_im):
    composite_im = c0_im / c0_im.max() + c1_im / c1_im.max() + c2_im / c2_im.max()
    composite_opened_im = opening(composite_im, disk(1))
    composite_closed_im = closing(composite_opened_im, disk(1))
    # composite_bw_im = composite_closed_im > threshold_otsu(composite_closed_im)
    composite_bw_im = hessian(composite_closed_im) > threshold_isodata(hessian(composite_closed_im))
    composite_bw_im = clear_border(composite_bw_im)
    composite_bw_im = binary_fill_holes(composite_bw_im)
    composite_bw_im = closing(composite_bw_im, disk(5))
    return composite_bw_im, composite_closed_im


def save_segmentation(save_file_path, gray_im, bw_im):
    norm_im = gray_im.astype(float) / gray_im.astype(float).max()
    rgb_stack = stack((norm_im, norm_im, norm_im), axis=-1)
    rgb_stack[bw_im, 1] = 1
    rgb_stack[bw_im, 0] = 0
    rgb_stack[bw_im, 2] = 0
    rgb_stack *= 255
    imsave(save_file_path, rgb_stack.astype('uint8'))


def estimate_comet_tail_props(file_path_str):
    c0_im, c1_im, c2_im = load_stack(file_path_str)
    bead_bw_im = segment_bead(c0_im)
    comet_bw_im, comet_im = segment_comet(c0_im, c1_im, c2_im)
    bead_label_im, no_beads = label(bead_bw_im, return_num=True)
    if no_beads > 0:
        c1_bw_im = gaussian(c1_im, sigma=1) > max(
            [threshold_otsu(c1_im), c1_im[~comet_bw_im].mean() + 3 * c1_im[~comet_bw_im].std()])
        c2_bw_im = gaussian(c2_im, sigma=1) > max(
            [threshold_otsu(c2_im), c2_im[~comet_bw_im].mean() + 3 * c2_im[~comet_bw_im].std()])
        skel_bw_im = skeletonize(comet_bw_im, method='lee')

        bead_mean_back_fluor = c0_im[~bead_bw_im].mean()
        c1_mean_back_fluor = c1_im[~c1_bw_im].mean()
        c2_mean_back_fluor = c2_im[~c2_bw_im].mean()

        npf_fluor_row = zeros(no_beads)
        chase_fluor_row = zeros(no_beads)
        chase_tail_length_row = zeros(no_beads)
        chase_channel_name_row = zeros(no_beads)
        comet_tail_axial_ratio_row = zeros(no_beads)
        npf_chase_tail_bw_im = zeros(c0_im.shape, dtype=bool)
        for i in range(no_beads):
            i_bead_bw_im = bead_label_im == (i + 1)
            i_npf_bw_im = i_bead_bw_im & ~erosion(i_bead_bw_im, disk(3))
            npf_chase_tail_bw_im[i_npf_bw_im] = True
            npf_fluor_row[i] = c0_im[i_npf_bw_im].mean() - bead_mean_back_fluor
            i_comet_bw_im = reconstruction(i_bead_bw_im & comet_bw_im, comet_bw_im).astype(bool)
            i_comet_regionprops_list = regionprops(label(i_comet_bw_im))
            comet_tail_axial_ratio_row[i] = i_comet_regionprops_list[0].major_axis_length / i_comet_regionprops_list[
                0].minor_axis_length
            i_c1_bw_im = c1_bw_im & i_comet_bw_im
            i_c2_bw_im = c2_bw_im & i_comet_bw_im
            if i_c1_bw_im.sum() > i_c2_bw_im.sum() + disk(6).sum():
                i_chase_bw_im = i_c2_bw_im & ~i_bead_bw_im
                chase_channel_name_row[i] = '2'
                chase_fluor_row[i] = c2_im[i_chase_bw_im].mean() - c2_mean_back_fluor
            elif i_c2_bw_im.sum() > i_c1_bw_im.sum() + disk(6).sum():
                i_chase_bw_im = i_c1_bw_im & ~i_bead_bw_im
                chase_channel_name_row[i] = '1'
                chase_fluor_row[i] = c1_im[i_chase_bw_im].mean() - c1_mean_back_fluor
            else:
                i_chase_bw_im = zeros(i_bead_bw_im.shape, dtype=bool)
                chase_channel_name_row[i] = '0'
                chase_fluor_row[i] = 0.0
            i_skel_bw_im = i_chase_bw_im & skel_bw_im
            npf_chase_tail_bw_im[i_skel_bw_im == True] = True
            chase_tail_length_row[i] = i_skel_bw_im.sum() * 0.16
        comet_tail_props_df = DataFrame.from_dict({'npf_fluor': npf_fluor_row,
                                                   'chase_fluor': chase_fluor_row,
                                                   'chase_tail_length': chase_tail_length_row,
                                                   'chase_channel_name': chase_channel_name_row,
                                                   'axial_ratio': comet_tail_axial_ratio_row,
                                                   'file_path_str': file_path_str})
    else:
        comet_tail_props_df = DataFrame.from_dict({'npf_fluor': array([]),
                                                   'chase_fluor': array([]),
                                                   'chase_tail_length': array([]),
                                                   'chase_channel_name': array([]),
                                                   'axial_ratio': array([]),
                                                   'file_path_str': file_path_str})
        npf_chase_tail_bw_im = zeros(c0_im.shape, dtype=bool)
    save_segmentation(file_path_str[:-21] + '_segment.jpg', comet_im, npf_chase_tail_bw_im)
    # save_segmentation(file_path_str[:-21] + '_segment.jpg', comet_im, comet_bw_im)
    return comet_tail_props_df


def skeletonize_comet(bead_bw_im, composite_bw_im):
    # Add bead binary to smooth out axis.
    medial_bw_im = medial_axis(bead_bw_im | composite_bw_im)
    # Remove the bead binary and ignore spurious mask(s).
    medial_bw_im = medial_bw_im & ~bead_bw_im
    medial_label_im, no_skels = label(medial_bw_im, return_num = True)
    if no_skels > 1:
        medial_props_list = regionprops(medial_label_im)
        skel_size_row = zeros(no_skels)
        for i in range(no_skels):
            skel_size_row[i] = medial_props_list[i].area
        index_largest_skel = skel_size_row.argmax()
        medial_bw_im = medial_label_im == (index_largest_skel + 1)
    _, _, degrees_mat = skeleton_to_csgraph(medial_bw_im)
    ends_bw_im = degrees_mat == 1
    no_ends = ends_bw_im.sum()
    if no_ends > 2:
        # Find longest path.
        cost_mat = zeros((no_ends, no_ends))
        ends_row_row, ends_col_row = ends_bw_im.nonzero()
        for i in range(no_ends):
            for j in range(no_ends):
                if i != j:
                    _, ij_cost = route_through_array(~medial_bw_im + 1, [ends_row_row[i], ends_col_row[i]], \
                                                     [ends_row_row[j], ends_col_row[j]], \
                                                     fully_connected = True, geometric = True)
                    cost_mat[i, j] = ij_cost
        costliest_row, costliest_col = unravel_index(cost_mat.argmax(), cost_mat.shape)
        costliest_path = route_through_array(~medial_bw_im + 1, \
                                             [ends_row_row[costliest_row], ends_col_row[costliest_row]], \
                                             [ends_row_row[costliest_col], ends_col_row[costliest_col]], \
                                             fully_connected = True)
        longest_path_bw_im = zeros(medial_bw_im.shape, dtype = bool)
        for i in range(len(costliest_path[0])):
            longest_path_bw_im[costliest_path[0][i][0], costliest_path[0][i][1]] = True
        #branches_bw_im = degrees_mat > 2
        #minus_branches_bw_im = medial_bw_im & ~dilation(branches_bw_im, square(7))
        #bw1_im = reconstruction(bead_bw_im & minus_branches_bw_im, minus_branches_bw_im).astype(bool)
        #bw2_im = bw1_im & ~bead_bw_im
        #longest_path_bw_im = remove_small_objects(bw2_im, min_size = 7, connectivity = 2)
    else:
        longest_path_bw_im = medial_bw_im
    return longest_path_bw_im

def sort_pixel_coords(bead_bw_im, path_bw_im):
    # Find endpoint nearest to bead.
    bead_props_list = regionprops(label(bead_bw_im))
    bead_center_row, bead_center_col = bead_props_list[0].centroid
    _, _, degrees_mat = skeleton_to_csgraph(path_bw_im)
    ends_bw_im = degrees_mat == 1
    ends_row_row, ends_col_row = ends_bw_im.nonzero()
    bead2ends_dist_row = (bead_center_row - ends_row_row)**2 + \
                          (bead_center_col - ends_col_row)**2
    index_nearest_end = bead2ends_dist_row.argmin()
    # Iterate to sort.
    no_pixels = path_bw_im.sum()
    current_row = ends_row_row[index_nearest_end]
    current_col = ends_col_row[index_nearest_end]
    path_row_row, path_col_row = path_bw_im.nonzero()
    path_row_row = delete(path_row_row, index_nearest_end)
    path_col_row = delete(path_col_row, index_nearest_end)
    sorted_row_row = zeros(no_pixels, dtype = int)
    sorted_col_row = zeros(no_pixels, dtype = int)
    sorted_row_row[0] = current_row
    sorted_col_row[0] = current_col
    for i in range(1, no_pixels):
        dist_row = (current_row - path_row_row)**2 + \
                   (current_col - path_col_row)**2
        index_nearest = dist_row.argmin()
        current_row = path_row_row[index_nearest]
        current_col = path_col_row[index_nearest]
        path_row_row = delete(path_row_row, index_nearest)
        path_col_row = delete(path_col_row, index_nearest)
        sorted_row_row[i] = current_row
        sorted_col_row[i] = current_col
    return sorted_row_row, sorted_col_row
    

def index_step(raw_row):
    raw_row = raw_row.astype(float)
    raw_row -= raw_row.mean()
    no_pixels = len(raw_row)
    step_row = hstack((ones(no_pixels), -1 * ones(no_pixels)))
    raw_step_row = convolve(raw_row, step_row, mode = 'valid')
    index_step = abs(raw_step_row).argmax()
    mu1 = raw_row[:index_step].mean()
    sigma1 = raw_row[:index_step].std()
    mu2 = raw_row[index_step:].mean()
    sigma2 = raw_row[index_step:].std()
    step_quality = abs(mu2 - mu1) / max([sigma1, sigma2]) / 3
    if raw_step_row[index_step] < 0:
        index_step *= -1
    return index_step, step_quality, raw_step_row[:-1]
    
    
def estimate_comet_tail_props_dev(file_path_str):
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
        tail_step_quality_row = zeros(no_beads)
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
                    print(file_path_str)
                    i_sorted_row_row, i_sorted_col_row = sort_pixel_coords(i_bead_bw_im, 
                                                                           i_skeleton_bw_im)
                    i_ds_row = (((i_sorted_row_row[1:] - i_sorted_row_row[:-1])**2 + \
                                 (i_sorted_col_row[1:] - i_sorted_col_row[:-1])**2)**0.5) * 0.16
                    i_position_row = append([0], i_ds_row.cumsum())
                    i_c1_scan_row = c1_im[i_sorted_row_row, i_sorted_col_row]
                    i_c2_scan_row = c2_im[i_sorted_row_row, i_sorted_col_row]
                    i_c1_step_index, i_c1_step_quality, i_c1_step_row = index_step(i_c1_scan_row)
                    i_c2_step_index, i_c2_step_quality, i_c2_step_row = index_step(i_c2_scan_row)
                    if (i_c1_step_index < 0) & (i_c2_step_index > 0):
                        i_skeleton_bw_im[i_sorted_row_row[-i_c1_step_index:], \
                                         i_sorted_col_row[-i_c1_step_index:]] = False
                        print(i_position_row[-i_c1_step_index])
                        print(i_c1_step_quality)
                        fig_hand, axes_hand = subplots()
                        axes_hand.plot(i_position_row, i_c1_scan_row)
                        axes_hand.plot(i_position_row, i_c1_step_row / 10)
                        savefig(file_path_str[:-21] + '_linescan.png')
                        i_pulse_im = equalize_adapthist(c1_im)
                        chase_fluor_row[i] = c1_im[i_skeleton_bw_im].mean() - actin1_mean_back_fluor
                        chase_tail_length_row[i] = i_position_row[-i_c1_step_index]
                        chase_channel_name_row[i] = 1
                        tail_step_quality_row[i] = i_c1_step_quality
                    elif (i_c1_step_index > 0) & (i_c2_step_index < 0):
                        i_skeleton_bw_im[i_sorted_row_row[-i_c2_step_index:], \
                                         i_sorted_col_row[-i_c2_step_index:]] = False
                        i_pulse_im = equalize_adapthist(c2_im)
                        fig_hand, axes_hand = subplots()
                        axes_hand.plot(i_position_row, i_c2_scan_row)
                        axes_hand.plot(i_position_row, i_c2_step_row / 10)
                        savefig(file_path_str[:-21] + '_linescan.png')
                        print(i_position_row[-i_c2_step_index])
                        print(i_c2_step_quality)
                        chase_fluor_row[i] = c2_im[i_skeleton_bw_im].mean() - actin2_mean_back_fluor
                        chase_tail_length_row[i] = i_position_row[-i_c2_step_index]
                        chase_channel_name_row[i] = 2
                        tail_step_quality_row[i] = i_c2_step_quality
                    elif (i_c1_step_quality > i_c2_step_quality):
                        tail_step_quality_row[i] = i_c1_step_quality
                        if i_c1_step_index < 0:
                            i_c1_step_index *= -1
                            chase_channel_name_row[i] = 1
                        else:
                            chase_channel_name_row[i] = 2
                        i_skeleton_bw_im[i_sorted_row_row[i_c1_step_index:], \
                                         i_sorted_col_row[i_c1_step_index:]] = False
                        chase_tail_length_row[i] = i_position_row[i_c1_step_index]
                        if chase_channel_name_row[i] == 1:
                            chase_fluor_row[i] = c1_im[i_skeleton_bw_im].mean() - actin1_mean_back_fluor
                        else:
                            chase_fluor_row[i] = c2_im[i_skeleton_bw_im].mean() - actin2_mean_back_fluor
                        print(i_position_row[i_c1_step_index])
                        print(i_c1_step_quality)
                        fig_hand, axes_hand = subplots()
                        axes_hand.plot(i_position_row, i_c1_scan_row)
                        axes_hand.plot(i_position_row, i_c1_step_row / 10)
                        savefig(file_path_str[:-21] + '_linescan.png')
                        i_pulse_im = equalize_adapthist(c1_im)
                    elif (i_c2_step_quality > i_c1_step_quality):
                        tail_step_quality_row[i] = i_c2_step_quality
                        if i_c2_step_index < 0:
                            i_c2_step_index *= -1
                            chase_channel_name_row[i] = 2
                        else:
                            chase_channel_name_row[i] = 1
                        i_skeleton_bw_im[i_sorted_row_row[i_c2_step_index:], \
                                         i_sorted_col_row[i_c2_step_index:]] = False
                        chase_tail_length_row[i] = i_position_row[i_c2_step_index]
                        if chase_channel_name_row[i] == 1:
                            chase_fluor_row[i] = c1_im[i_skeleton_bw_im].mean() - actin1_mean_back_fluor
                        else:
                            chase_fluor_row[i] = c2_im[i_skeleton_bw_im].mean() - actin2_mean_back_fluor
                        i_pulse_im = equalize_adapthist(c2_im)
                        fig_hand, axes_hand = subplots()
                        axes_hand.plot(i_position_row, i_c2_scan_row)
                        axes_hand.plot(i_position_row, i_c2_step_row / 10)
                        savefig(file_path_str[:-21] + '_linescan.png')
                        print(i_position_row[i_c2_step_index])
                        print(i_c2_step_quality)
                    imsave(file_path_str[:-21] + '_skel_segment.jpg', label2rgb(label(i_skeleton_bw_im), image = invert(i_pulse_im), bg_label = 0, colors = ['yellow']).astype('uint8'))
            else:
                print("No actin tail.")
            comet_tail_props_df = DataFrame.from_dict({'npf_fluor': npf_fluor_row,
                                                   'chase_fluor': chase_fluor_row,
                                                   'chase_tail_length': chase_tail_length_row,
                                                   'chase_channel_name': chase_channel_name_row,
                                                   'tail_step_quality': tail_step_quality_row,
                                                   'file_path_str': basename(file_path_str)})
    else:
        print("No beads found")
        comet_tail_props_df = DataFrame.from_dict({'npf_fluor': array([]),
                                                   'chase_fluor': array([]),
                                                   'chase_tail_length': array([]),
                                                   'chase_channel_name': array([]),
                                                   'tail_step_quality': array([]),
                                                   'file_path_str': basename(file_path_str)})
    return comet_tail_props_df
    

def batch_analysis_dev(folder_path_str, save_segmentation = False):
    mmstack_file_path_list = find_mmstack_files(folder_path_str)
    no_files = len(mmstack_file_path_list)
    comet_tail_props_df = DataFrame.from_dict({'npf_fluor': array([]),
                                               'chase_fluor': array([]),
                                               'chase_tail_length': array([]),
                                               'chase_channel_name': array([]),
                                               'tail_step_quality': array([]),
                                               'file_path_str': array([])})
    for i in range(no_files):
        # Load stack.
        i_c0_im, i_c1_im, i_c2_im = load_stack(mmstack_file_path_list[i])
        # Segment channels.
        i_bead_bw_im = segment_bead(i_c0_im)
        i_actin1_bw_im = segment_actin(i_c1_im, i_bead_bw_im)
        i_actin2_bw_im = segment_actin(i_c2_im, i_bead_bw_im)
        # Save segmentation results.
        if save_segmentation == True:
            i_bead_overlay_im = label2rgb(label(find_boundaries(i_bead_bw_im)), image = equalize_adapthist(invert(i_c0_im)), bg_label = 0, colors = ['green'])
            i_actin1_overlay_im = label2rgb(label(find_boundaries(i_actin1_bw_im)), image = equalize_adapthist(invert(i_c1_im)), bg_label = 0, colors = ['red'])
            i_actin2_overlay_im = label2rgb(label(find_boundaries(i_actin2_bw_im)), image = equalize_adapthist(invert(i_c2_im)), bg_label = 0, colors = ['magenta'])
            imsave(mmstack_file_path_list[i][:-21] + '_c0_segment.jpg', i_bead_overlay_im.astype('uint8'))
            imsave(mmstack_file_path_list[i][:-21] + '_c1_segment.jpg', i_actin1_overlay_im.astype('uint8'))
            imsave(mmstack_file_path_list[i][:-21] + '_c2_segment.jpg', i_actin2_overlay_im.astype('uint8'))
        # Analyze comet tails.
        i_comet_tail_props_df = estimate_comet_tail_props_dev(mmstack_file_path_list[i])
        comet_tail_props_df = comet_tail_props_df.append(i_comet_tail_props_df)
    comet_tail_props_df = comet_tail_props_df.reset_index()
    comet_tail_props_df.to_csv(folder_path_str + '/comet_tail_properties.csv')
    return comet_tail_props_df
        

def batch_analysis(folder_path_str):
    comet_tail_props_df = DataFrame.from_dict({'npf_fluor': array([]),
                                               'chase_fluor': array([]),
                                               'chase_tail_length': array([]),
                                               'chase_channel_name': array([]),
                                               'axial_ratio': array([]),
                                               'file_path_str': array([])})
    mmstack_file_path_list = find_mmstack_files(folder_path_str)
    no_files = len(mmstack_file_path_list)
    for i in range(no_files):
        i_comet_tail_props_df = estimate_comet_tail_props(mmstack_file_path_list[i])
        comet_tail_props_df = comet_tail_props_df.append(i_comet_tail_props_df)
    comet_tail_props_df = comet_tail_props_df.reset_index()
    comet_tail_props_df.to_csv(folder_path_str + '/comet_tail_properties.csv')
    return comet_tail_props_df
