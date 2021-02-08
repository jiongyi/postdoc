from os import walk
from os.path import join, basename
import fnmatch
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from scipy.ndimage.morphology import binary_fill_holes
from skimage.feature import canny
from skimage.morphology import closing, disk, erosion, dilation, opening, reconstruction, remove_small_objects, \
    skeletonize
from skimage.filters import threshold_triangle, threshold_isodata, threshold_otsu, gaussian, hessian, sato
from skimage.exposure import equalize_adapthist
from numpy import array, mean, nan, stack, sum, zeros, linspace, pi, tan, inf, sqrt, unique, max, abs
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries, clear_border
from skimage.transform import rotate
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
    return mm_stack[:, :, 0].astype(float), mm_stack[:, :, 1].astype(float), mm_stack[:, :, 2].astype(float)


def segment_bead(raw_im):
    closed_im = closing(raw_im, disk(1))
    opened_im = opening(closed_im, disk(1))
    bw_im = opened_im > threshold_otsu(opened_im)
    bead_bw_im = clear_border(remove_small_objects(bw_im, min_size=disk(6).sum()))
    return bead_bw_im


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
            i_bead_bw_props_list = regionprops(label(i_bead_bw_im))
            i_bead_center_row, i_bead_center_col = i_bead_bw_props_list[0].centroid
            npf_chase_tail_bw_im[i_npf_bw_im] = True
            npf_fluor_row[i] = c0_im[i_npf_bw_im].mean() - bead_mean_back_fluor
            i_comet_bw_im = reconstruction(i_bead_bw_im & comet_bw_im, comet_bw_im).astype(bool)
            i_comet_regionprops_list = regionprops(label(i_comet_bw_im))
            comet_tail_axial_ratio_row[i] = i_comet_regionprops_list[0].major_axis_length / i_comet_regionprops_list[
                0].minor_axis_length
            i_c1_threshold = threshold_otsu(c1_im[i_comet_bw_im])
            i_c1_back_bw_im = (c1_im < i_c1_threshold) & i_comet_bw_im
            i_c1_fore_bw_im = (c1_im >= i_c1_threshold) & i_comet_bw_im
            i_c1_back_mean = c1_im[i_c1_back_bw_im].mean()
            i_c1_fore_mean = c1_im[i_c1_fore_bw_im].mean()
            i_c1_fore_bw_props_list = regionprops(label(i_c1_fore_bw_im))
            i_c1_fore_center_row, i_c1_fore_center_col = i_c1_fore_bw_props_list[0].centroid
            i_c1_back_bw_props_list = regionprops(label(i_c1_back_bw_im))
            i_c1_back_center_row, i_c1_back_center_col = i_c1_back_bw_props_list[0].centroid
            i_c2_threshold = threshold_otsu(c2_im[i_comet_bw_im])
            i_c2_back_bw_im = (c2_im < i_c2_threshold) & i_comet_bw_im
            i_c2_fore_bw_im = (c2_im >= i_c2_threshold) & i_comet_bw_im
            i_c2_back_mean = c2_im[i_c2_back_bw_im].mean()
            i_c2_fore_mean = c2_im[i_c2_fore_bw_im].mean()
            i_c2_fore_bw_props_list = regionprops(label(i_c2_fore_bw_im))
            i_c2_fore_center_row, i_c2_fore_center_col = i_c2_fore_bw_props_list[0].centroid
            i_c2_back_bw_props_list = regionprops(label(i_c2_back_bw_im))
            i_c2_back_center_row, i_c2_back_center_col = i_c2_back_bw_props_list[0].centroid
            bead2c1_dist = (i_bead_center_row - i_c1_fore_center_row)**2 + \
                           (i_bead_center_col - i_c1_fore_center_col)**2
            bead2c2_dist = (i_bead_center_row - i_c2_fore_center_row)**2 + \
                           (i_bead_center_col - i_c2_fore_center_col)**2
            if bead2c1_dist < bead2c2_dist:
                i_chase_bw_im = i_c1_fore_bw_im & ~i_bead_bw_im
                chase_channel_name_row[i] = '1'
                chase_fluor_row[i] = c1_im[i_chase_bw_im].mean() - c1_mean_back_fluor
            else:
                i_chase_bw_im = i_c2_fore_bw_im & ~i_bead_bw_im
                chase_channel_name_row[i] = '2'
                chase_fluor_row[i] = c2_im[i_chase_bw_im].mean() - c2_mean_back_fluor
            i_skel_bw_im = i_chase_bw_im & skel_bw_im
            npf_chase_tail_bw_im[i_skel_bw_im == True] = True
            chase_tail_length_row[i] = i_skel_bw_im.sum() * 0.16
        comet_tail_props_df = DataFrame.from_dict({'npf_fluor': npf_fluor_row,
                                                   'chase_fluor': chase_fluor_row,
                                                   'chase_tail_length': chase_tail_length_row,
                                                   'chase_channel_name': chase_channel_name_row,
                                                   'axial_ratio': comet_tail_axial_ratio_row,
                                                   'file_path_str': basename(file_path_str)})
    else:
        comet_tail_props_df = DataFrame.from_dict({'npf_fluor': array([]),
                                                   'chase_fluor': array([]),
                                                   'chase_tail_length': array([]),
                                                   'chase_channel_name': array([]),
                                                   'axial_ratio': array([]),
                                                   'file_path_str': basename(file_path_str)})
        npf_chase_tail_bw_im = zeros(c0_im.shape, dtype=bool)
    save_segmentation(file_path_str[:-21] + '_segment.jpg', comet_im, npf_chase_tail_bw_im)
    # save_segmentation(file_path_str[:-21] + '_segment.jpg', comet_im, comet_bw_im)
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
