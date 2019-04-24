from os import walk
from os.path import join, split
from fnmatch import fnmatch
from numpy import zeros, argmax, arange, mean, std, array, sum, median, percentile
from skimage.filters import sobel, gaussian, threshold_otsu
from skimage.io import imread, imsave
from skimage import img_as_uint, img_as_bool
from skimage.morphology import remove_small_objects, erosion, reconstruction, disk
from skimage.measure import label, regionprops
from pandas import DataFrame, read_csv, concat
from seaborn import boxplot, swarmplot

# Define constants.
PIXEL_AREA_2_UM2 = 0.10185185185**2

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

def open_reconstruction(raw_mat, px_radius):
    eroded_mat = erosion(raw_mat, disk(px_radius))
    opened_mat = reconstruction(eroded_mat, raw_mat)
    return opened_mat

def segment_dna_rp(dna_rp_mat, bw_dapi_mat):
    opened_dna_rp_mat = open_reconstruction(dna_rp_mat, 5)
    opened_offset = min(opened_dna_rp_mat[opened_dna_rp_mat > 0])
    opened_dna_rp_mat[opened_dna_rp_mat <= 0.0] = opened_offset
    scaled_dna_rp_mat = dna_rp_mat / opened_dna_rp_mat
    bw_dna_rp_mat = scaled_dna_rp_mat > threshold_otsu(scaled_dna_rp_mat[bw_dapi_mat])
    bw_dna_rp_mat[~bw_dapi_mat] = False
    # Check that ph2ax intensity is high enough.
    au_threshold_dna_rp_mat = zeros(bw_dna_rp_mat.shape)
    label_dapi_mat, no_labels = label(bw_dapi_mat, return_num = True)
    for i in range(1, no_labels + 1):
        i_label_dapi_mat = bw_dapi_mat == i
        i_label_dna_rp_back_mat = i_label_dapi_mat & ~bw_dna_rp_mat
        i_mean_back_dna_rp_au = mean(dna_rp_mat[i_label_dna_rp_back_mat])
        i_std_back_dna_rp_au = std(dna_rp_mat[i_label_dna_rp_back_mat])
        au_threshold_dna_rp_mat[i_label_dapi_mat] = i_mean_back_dna_rp_au + 3 * i_std_back_dna_rp_au
    bw_dna_rp_mat[dna_rp_mat < au_threshold_dna_rp_mat] = False
    return bw_dna_rp_mat

def segment_ph2ax(ph2ax_mat, bw_dapi_mat):
    opened_ph2ax_mat = open_reconstruction(ph2ax_mat, 5)
    opened_offset = min(opened_ph2ax_mat[opened_ph2ax_mat > 0])
    opened_ph2ax_mat[opened_ph2ax_mat <= 0.0] = opened_offset
    scaled_ph2ax_mat = ph2ax_mat / opened_ph2ax_mat
    bw_ph2ax_mat = scaled_ph2ax_mat > threshold_otsu(scaled_ph2ax_mat[bw_dapi_mat])
    bw_ph2ax_mat[~bw_dapi_mat] = False
    # Check that ph2ax intensity is high enough.
    au_threshold_ph2ax_mat = zeros(bw_ph2ax_mat.shape)
    label_dapi_mat, no_labels = label(bw_dapi_mat, return_num = True)
    for i in range(1, no_labels + 1):
        i_label_dapi_mat = bw_dapi_mat == i
        i_label_ph2ax_back_mat = i_label_dapi_mat & ~bw_ph2ax_mat
        i_mean_back_ph2ax_au = mean(ph2ax_mat[i_label_ph2ax_back_mat])
        i_std_back_ph2ax_au = std(ph2ax_mat[i_label_ph2ax_back_mat])
        au_threshold_ph2ax_mat[i_label_dapi_mat] = i_mean_back_ph2ax_au + 3 * i_std_back_ph2ax_au
    bw_ph2ax_mat[ph2ax_mat < au_threshold_ph2ax_mat] = False
    return bw_ph2ax_mat

def segment_dapi(dapi_mat):
    gaussian_dapi_mat = gaussian(dapi_mat, sigma = 5)
    bw_dapi_mat = remove_small_objects(gaussian_dapi_mat > threshold_otsu(gaussian_dapi_mat), min_size = 96)
    return bw_dapi_mat

def batch_segment_dapi(folder_name, channel_no = 1):
    extended_file_name_list = find_tif_files(folder_name, '*_extended.tif')
    no_files = len(extended_file_name_list)
    for i in range(no_files):
        try:
            i_extended_mat = imread(extended_file_name_list[i])
            i_bw_dapi_mat = segment_dapi(i_extended_mat[:, :, channel_no - 1])
            imsave(extended_file_name_list[i][:-4] + '_bw_dapi.tif', img_as_uint(i_bw_dapi_mat))
        except:
            print("Failed to segment DAPI in" + extended_file_name_list[i])

def batch_segment_ph2ax(folder_name, channel_no = 2):
    extended_file_name_list = find_tif_files(folder_name, '*_extended.tif')
    no_files = len(extended_file_name_list)
    for i in range(no_files):
        #try:
        i_extended_mat = imread(extended_file_name_list[i])
        i_bw_dapi_mat = img_as_bool(imread(extended_file_name_list[i][:-4] + '_bw_dapi.tif'))
        i_bw_ph2ax_mat = segment_ph2ax(i_extended_mat[:, :, channel_no - 1], i_bw_dapi_mat)
        imsave(extended_file_name_list[i][:-4] + '_bw_ph2ax.tif', img_as_uint(i_bw_ph2ax_mat))
        #except:
        #    print("Failed to segment ph2ax in" + extended_file_name_list[i])

def batch_segment_dna_rp(folder_name, channel_no = 3):
    extended_file_name_list = find_tif_files(folder_name, '*_extended.tif')
    no_files = len(extended_file_name_list)
    for i in range(no_files):
        try:
            i_extended_mat = imread(extended_file_name_list[i])
            i_bw_dapi_mat = img_as_bool(imread(extended_file_name_list[i][:-4] + '_bw_dapi.tif'))
            i_bw_dna_rp_mat = segment_dna_rp(i_extended_mat[:, :, channel_no - 1], i_bw_dapi_mat)
            imsave(extended_file_name_list[i][:-4] + '_bw_dna_rp.tif', img_as_uint(i_bw_dna_rp_mat))
        except:
            print("Failed to segment ph2ax in" + extended_file_name_list[i])

def batch_get_regionprops(folder_name, dapi_channel_no = 0, ph2ax_channel_no = 1, dna_rp_channel_no = 2):
    extended_file_name_list = find_tif_files(folder_name, '*_extended.tif')
    no_files = len(extended_file_name_list)
    for i in range(no_files):
        #try:
        i_extended_mat = imread(extended_file_name_list[i])
        i_bw_dapi_mat = img_as_bool(imread(extended_file_name_list[i][:-4] + '_bw_dapi.tif'))
        i_bw_ph2ax_mat = img_as_bool(imread(extended_file_name_list[i][:-4] + '_bw_ph2ax.tif'))
        i_bw_dna_rp_mat = img_as_bool(imread(extended_file_name_list[i][:-4] + '_bw_dna_rp.tif'))
        i_dapi_regionprops_list = regionprops(label(i_bw_dapi_mat), i_extended_mat[:, :, dapi_channel_no] - mean(i_extended_mat[~i_bw_dapi_mat, dapi_channel_no]))
        i_dapi_region_area_row = array([x.area for x in i_dapi_regionprops_list]) * PIXEL_AREA_2_UM2
        i_dapi_region_mean_au_row = array([x.mean_intensity for x in i_dapi_regionprops_list])
        i_label_dapi_mat, i_no_labels = label(i_bw_dapi_mat, return_num = True)
        i_ph2ax_no_foci_row = zeros(i_no_labels)
        i_ph2ax_flux_row = zeros(i_no_labels)
        i_dna_rp_no_foci_row = zeros(i_no_labels)
        i_dna_rp_flux_row = zeros(i_no_labels)
        i_ph2ax_dna_rp_no_foci_row = zeros(i_no_labels)
        i_ph2ax_dna_rp_flux_fraction_row = zeros(i_no_labels)
        for j in range(1, i_no_labels + 1):
            j_bw_dapi_mat = i_label_dapi_mat == j
            j_ph2ax_regionprops_list = regionprops(label(i_bw_ph2ax_mat & j_bw_dapi_mat), i_extended_mat[:, :, ph2ax_channel_no] - mean(i_extended_mat[j_bw_dapi_mat & ~i_bw_ph2ax_mat, ph2ax_channel_no]))
            i_ph2ax_no_foci_row[j - 1] = len(j_ph2ax_regionprops_list)
            j_dna_rp_regionprops_list = regionprops(label(i_bw_dna_rp_mat & j_bw_dapi_mat), i_extended_mat[:, :, dna_rp_channel_no] - mean(i_extended_mat[j_bw_dapi_mat & ~i_bw_dna_rp_mat, dna_rp_channel_no]))
            i_dna_rp_no_foci_row[j - 1] = len(j_dna_rp_regionprops_list)
            j_ph2ax_dna_rp_regionprops_list = regionprops(label(i_bw_ph2ax_mat & i_bw_dna_rp_mat & j_bw_dapi_mat), i_extended_mat[:, :, dna_rp_channel_no] - mean(i_extended_mat[j_bw_dapi_mat & ~i_bw_dna_rp_mat, dna_rp_channel_no]))
            i_ph2ax_dna_rp_no_foci_row[j - 1] = len(j_ph2ax_dna_rp_regionprops_list)
            if i_ph2ax_no_foci_row[j - 1] == 0:
                i_ph2ax_flux_row[j - 1] = 0.0
            else:
                j_ph2ax_region_area_row = array([x.area for x in j_ph2ax_regionprops_list]) * PIXEL_AREA_2_UM2
                j_ph2ax_region_mean_au_row = array([x.mean_intensity for x in j_ph2ax_regionprops_list])
                i_ph2ax_flux_row[j - 1] = sum(j_ph2ax_region_area_row * j_ph2ax_region_mean_au_row)
            if i_dna_rp_no_foci_row[j - 1] == 0:
                i_dna_rp_flux_row[j - 1] = 0.0
            else:
                j_dna_rp_region_area_row = array([x.area for x in j_dna_rp_regionprops_list]) * PIXEL_AREA_2_UM2
                j_dna_rp_region_mean_au_row = array([x.mean_intensity for x in j_dna_rp_regionprops_list])
                i_dna_rp_flux_row[j - 1] = sum(j_dna_rp_region_area_row * j_dna_rp_region_mean_au_row)
            if i_ph2ax_dna_rp_no_foci_row[j - 1] == 0:
                i_ph2ax_dna_rp_flux_fraction_row[j - 1] = 0.0
            else:
                j_ph2ax_dna_rp_region_area_row = array([x.area for x in j_ph2ax_dna_rp_regionprops_list]) * PIXEL_AREA_2_UM2
                j_ph2ax_dna_rp_region_mean_au_row = array([x.mean_intensity for x in j_ph2ax_dna_rp_regionprops_list])
                i_ph2ax_dna_rp_flux_fraction_row[j - 1] = sum(j_ph2ax_dna_rp_region_area_row * j_ph2ax_dna_rp_region_mean_au_row) / i_dna_rp_flux_row[j - 1]
        i_dapi_df = DataFrame.from_dict(data = {'DAPI area': i_dapi_region_area_row,
        'Mean DAPI AU': i_dapi_region_mean_au_row,
        'No pH2AX foci': i_ph2ax_no_foci_row,
        'pH2AX flux': i_ph2ax_flux_row,
        'No DNA RP foci': i_dna_rp_no_foci_row,
        'DNA RP flux': i_dna_rp_flux_row,
        'No DAPI-DNA RP foci': i_ph2ax_dna_rp_no_foci_row,
        'Fraction DNA RP flux': i_ph2ax_dna_rp_flux_fraction_row})
        i_dapi_df.to_csv(extended_file_name_list[i][:-4] + '.csv')
        #except:
        #    print("Failed to get regionprops from" + extended_file_name_list[i])

def batch_aggregate_data(folder_name):
    csv_file_name_list = find_tif_files(folder_name, '*_extended.csv')
    no_files = len(csv_file_name_list)
    aggregated_df = read_csv(csv_file_name_list[0])
    for i in range(1, no_files):
        aggregated_df = concat([aggregated_df, read_csv(csv_file_name_list[i])], ignore_index = True)
    return aggregated_df

def batch_extend_depth_field(folder_name):
    mm_stack_file_name_list = find_tif_files(folder_name, '*.ome.tif')
    no_files = len(mm_stack_file_name_list)
    for i in range(no_files):
        try:
            i_mm_stack = imread(mm_stack_file_name_list[i])
            i_no_channels, i_no_slices, i_no_rows, i_no_columns = i_mm_stack.shape
            i_extended_mat = zeros((i_no_channels, i_no_rows, i_no_columns))
            for j in range(i_no_channels):
                i_extended_mat[j, :, :] = gaussian(extend_depth_field(i_mm_stack[j, :, :, :]), sigma = 2)
            imsave(mm_stack_file_name_list[i][:-4] + '_extended.tif', img_as_uint(i_extended_mat))
        except:
            print("Failed to analyze" + mm_stack_file_name_list[i])
            
def compare_ph2ax_flux_density(a_df, a_label, b_df, b_label):
    a_ph2ax_flux_density_row = a_df['pH2AX flux'] / a_df['DAPI area']
    b_ph2ax_flux_density_row = b_df['pH2AX flux'] / b_df['DAPI area']
    ph2ax_flux_density_df = DataFrame.from_dict(data = {a_label: a_ph2ax_flux_density_row, b_label: b_ph2ax_flux_density_row}, orient = 'index')
    boxplot_handle = boxplot(data = ph2ax_flux_density_df.transpose())
    boxplot_handle.tick_params(labelsize = 16)
    boxplot_handle.set_ylabel('pH2AX flux density', fontsize = 16)
    print("Mean A:" + str(mean(a_ph2ax_flux_density_row)))
    print("Stdev A:" + str(std(a_ph2ax_flux_density_row)))
    print("Mean B:" + str(mean(b_ph2ax_flux_density_row)))
    print("Stdev B:" + str(std(b_ph2ax_flux_density_row)))
    return boxplot_handle

def compare_dna_rp_flux_density(a_df, a_label, b_df, b_label):
    a_dna_rp_flux_density_row = a_df['DNA RP flux'] / a_df['DAPI area']
    b_dna_rp_flux_density_row = b_df['DNA RP flux'] / b_df['DAPI area']
    dna_rp_flux_density_df = DataFrame.from_dict(data = {a_label: a_dna_rp_flux_density_row, b_label: b_dna_rp_flux_density_row}, orient = 'index')
    boxplot_handle = boxplot(data = dna_rp_flux_density_df.transpose())
    boxplot_handle.tick_params(labelsize = 16)
    boxplot_handle.set_ylabel('DNA RP flux density', fontsize = 16)
    print("Mean A:" + str(mean(a_dna_rp_flux_density_row)))
    print("Stdev A:" + str(std(a_dna_rp_flux_density_row)))
    print("Mean B:" + str(mean(b_dna_rp_flux_density_row)))
    print("Stdev B:" + str(std(b_dna_rp_flux_density_row)))
    return boxplot_handle
    
def compare_repair_positive_flux(a_df, a_label, b_df, b_label):
    a_repair_positive_flux_density_row = a_df['Fraction DNA RP flux']
    b_repair_positive_flux_density_row = b_df['Fraction DNA RP flux']
    repair_positive_flux_density_df = DataFrame.from_dict(data = {a_label: a_repair_positive_flux_density_row, b_label: b_repair_positive_flux_density_row}, orient = 'index')
    boxplot_handle = boxplot(data = repair_positive_flux_density_df.transpose())
    boxplot_handle.tick_params(labelsize = 16)
    boxplot_handle.set_ylabel('Fractional repair flux density', fontsize = 16)
    return boxplot_handle

def compute_boxplot_statistics(df):
    median_ph2ax_flux = median(df['pH2AX flux'] / df['DAPI area'])
    ph2ax_quartile_row = percentile(df['pH2AX flux'] / df['DAPI area'], [25, 75])
    median_dna_rp_flux = median(df['DNA RP flux'] / df['DAPI area'])
    dna_rp_quartile_row = percentile(df['DNA RP flux'] / df['DAPI area'], [25, 75])
    return median_ph2ax_flux, ph2ax_quartile_row, median_dna_rp_flux, dna_rp_quartile_row

def compare_dna_rp_foci_density(a_df, a_label, b_df, b_label):
    a_dna_rp_flux_density_row = a_df['No DNA RP foci'] / a_df['DAPI area']
    b_dna_rp_flux_density_row = b_df['No DNA RP foci'] / b_df['DAPI area']
    dna_rp_flux_density_df = DataFrame.from_dict(data = {a_label: a_dna_rp_flux_density_row, b_label: b_dna_rp_flux_density_row}, orient = 'index')
    boxplot_handle = swarmplot(data = dna_rp_flux_density_df.transpose())
    boxplot_handle.tick_params(labelsize = 16)
    boxplot_handle.set_ylabel('DNA RP flux density', fontsize = 16)
    print("Mean A:" + str(mean(a_dna_rp_flux_density_row)))
    print("Stdev A:" + str(std(a_dna_rp_flux_density_row)))
    print("Mean B:" + str(mean(b_dna_rp_flux_density_row)))
    print("Stdev B:" + str(std(b_dna_rp_flux_density_row)))
    return boxplot_handle