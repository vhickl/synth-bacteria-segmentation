#%%
from skimage.measure import regionprops
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import napari
mpl.rcParams['figure.dpi'] = 300

def IoU(I1,I2): #intersect over union
    #check dims
    if I1.shape != I2.shape:
        return ValueError("Dimensions of two images do not match")
    U = 0
    I = 0
    for i in range(I1.shape[0]):
        for j in range(I1.shape[1]):
            if I1[i,j]>0 and I2[i,j]>0:
                U += 1
                I += 1
            elif I1[i,j]>0 or I2[i,j]>0:
                U +=1
    return I/U

def find_seg_candidates(Iground, Iseg):
#candidates are any predicted cells that cover at least 50% of ground truth cell (IoU can be <0.5)
    gN = np.max(Iground)
    props = regionprops(Iground)
    FN = 0
    FP = 0
    TP = 0
    IoU = np.zeros(gN)
    TP_ind_list = []
    for i in range(gN):
        rr = props[i]['coords'][:,0]
        cc = props[i]['coords'][:,1]
        cand_ind = sorted(list(set(Iseg[rr,cc])), key = list(Iseg[rr,cc]).count, reverse=True)
        if cand_ind[0] == 0:
            FN += 1
            continue
        intersection = np.count_nonzero(Iseg[rr,cc] == cand_ind[0])
        union = (props[i]['area'] + np.count_nonzero(Iseg == cand_ind[0]) - intersection)
        IoU[i] = intersection / union
        TP += 1
        TP_ind_list.append(cand_ind[0])

    distinct_cands = len(set(TP_ind_list))
    FP = np.max(Iseg) - distinct_cands
    counts = (gN, TP,FP,FN)
    RQ = TP / (TP + 0.5*(abs(FP)+FN))
    SQ = np.sum(IoU)/counts[1]
    PQ = RQ * SQ

    return IoU, PQ, RQ, SQ, counts, distinct_cands

def find_seg_candidates2(Iground, Iseg):
#candidates are any predicted cells for which IoU with a ground truth cell is >0.5
    gN = np.max(Iground)
    props = regionprops(Iground)
    FN = 0
    FP = 0
    TP = 0
    IoU = np.zeros(gN)
    TP_ind_list = []
    for i in range(gN):
        rr = props[i]['coords'][:,0]
        cc = props[i]['coords'][:,1]
        cand_ind = sorted(list(set(Iseg[rr,cc])), key = list(Iseg[rr,cc]).count, reverse=True)
        if cand_ind[0] == 0:
            FN += 1
            continue
        intersection = np.count_nonzero(Iseg[rr,cc] == cand_ind[0])
        union = (props[i]['area'] + np.count_nonzero(Iseg == cand_ind[0]) - intersection)
        if intersection / union >= 0.5:
            IoU[i] = intersection / union
            TP += 1
            TP_ind_list.append(cand_ind[0])
        else:
            FN += 1

    FP = np.max(Iseg) - TP
    counts = (gN, TP,FP,FN)
    RQ = TP / (TP + 0.5*(abs(FP)+FN))
    SQ = np.sum(IoU)/counts[1]
    PQ = RQ * SQ

    return IoU, PQ, RQ, SQ, counts

def remove_small_cells(I,thresh):
    Inew = np.copy(I)
    props = regionprops(I)
    rem_count = 0
    for x in props:
        if x['area'] < thresh:
            rr = x['coords'][:,0]   
            cc = x['coords'][:,1]    
            Inew[rr,cc] = 0
            rem_count += 1
    print(f"Removed {rem_count} cells of size less than {thresh} pixels from image.")
    return Inew

def remove_double_candidates(L1, L2, thresh=0.25): #takes 2 label arrays, removes labels from L1 that appear in L2
    L1new = np.copy(L1)
    N1 = np.max(L1)
    props = regionprops(L1)
    count = 0
    for i in range(N1):
        rr = props[i]['coords'][:,0]
        cc = props[i]['coords'][:,1]
        cand_ind = sorted(list(set(L2[rr,cc])), key = list(L2[rr,cc]).count, reverse=True)
        if cand_ind[0] == 0:
            continue
        intersection = np.count_nonzero(L2[rr,cc] == cand_ind[0])
        union = (props[i]['area'] + np.count_nonzero(L2 == cand_ind[0]) - intersection)
        if intersection / union >= thresh:
            L1new[rr,cc] = 0
            count += 1

    return L1new, count


#%% load in images
Ig = io.imread('synthetic bacteria images/real images for cycleGAN/testing images/flat7_avg4_edit_014_cp_manual_masks.png')
I1 = io.imread('synthetic bacteria images/real images for cycleGAN/testing images/flat7_avg4_edit_014_cp_omni35_masks.png')
I2 = io.imread('synthetic bacteria images/real images for cycleGAN/testing images/flat7_avg4_edit_014_cp_synth_masks.png')

#%% remove small cells

I1b = remove_small_cells(I1,50)
I2b = remove_small_cells(I2,50)

#%%
IoU_s2, PQ_s2, RQ_s2, SQ_s2, counts_s2 = find_seg_candidates2(Ig, I2)
IoU_bf2, PQ_bf2, RQ_bf2, SQ_bf2, counts_bf2 = find_seg_candidates2(Ig, I1)


#%%
fig, ax = plt.subplots(1,1)

ax.hist(IoU_bf2, bins=20, edgecolor = "black", label = "Bact_fluor_omni", alpha = 0.8)
ax.hist(IoU_s2, bins=20, edgecolor = "black", label = "Synthetic model", alpha = 0.8)
ax = plt.gca()
plt.xlabel('IoU')
plt.ylabel('cell count')
ax.legend(loc='upper center')
plt.show()

#%% mixed images

I = io.imread('K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/Mixed testing 05-04-24/Brightfield/40X_mixed/edit_crop_256/testing_images/crop_bg_edit02_035.tif')

Ig_circ = io.imread('K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/Mixed testing 05-04-24/Brightfield/40X_mixed/edit_crop_256/testing_images/crop_bg_edit02_035_cp_circle_man_masks.png')
I_syn_circ = io.imread('K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/Mixed testing 05-04-24/Brightfield/40X_mixed/edit_crop_256/testing_images/crop_bg_edit02_035_cp_circle_synth_masks.png')
Ig_rod = io.imread('K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/Mixed testing 05-04-24/Brightfield/40X_mixed/edit_crop_256/testing_images/crop_bg_edit02_035_cp_rod_man_masks.png')
I_syn_rod = io.imread('K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/Mixed testing 05-04-24/Brightfield/40X_mixed/edit_crop_256/testing_images/crop_bg_edit02_035_cp_rod_synth_masks.png')

I_syn_circ, double_count = remove_double_candidates(I_syn_circ, I_syn_rod)

IoU_c, PQ_c, RQ_c, SQ_c, counts_c = find_seg_candidates2(Ig_circ, I_syn_circ)
IoU_r, PQ_r, RQ_r, SQ_r, counts_r = find_seg_candidates2(Ig_rod, I_syn_rod)

I_syn_circ_bin = np.divide(I_syn_circ, I_syn_circ, out=np.zeros_like(I_syn_circ, dtype='float64'), where=I_syn_circ!=0)
I_syn_rod_bin = np.divide(I_syn_rod, I_syn_rod, out=np.zeros_like(I_syn_rod, dtype='float64'), where=I_syn_rod!=0)

'''
viewer3 = napari.view_image(I)
viewer3.add_labels(I_syn_circ_bin.astype('uint16'))
viewer3.add_labels(I_syn_rod_bin.astype('uint16'))
'''

fig, ax = plt.subplots(1,1) 

ax.hist(IoU_c, bins=20, edgecolor = "black", label = "circles", alpha = 0.8)
ax.hist(IoU_r, bins=20, edgecolor = "black", label = "rods", alpha = 0.8)
ax = plt.gca()
plt.xlabel('IoU')
plt.ylabel('cell count')
ax.legend(loc='upper center')
plt.show()

#%% batch processing functions
def search_for_files(search_str, path):
    for f in os.listdir(path):
        if search_str in f:
            yield f

def reorder_labels(I):
    I2 = np.copy(I)
    n=1
    for i in np.array(list(set(I.flatten())))[1:]: #iterates through all unique values in I except the first, which is assumed to be 0 (background)
        I2[I2 == i] = n
        n += 1
    return I2
    
def batch_comparison(input_folder):
    Imgs = list(search_for_files('.tif', input_folder))
    Imgs_G = list(search_for_files('manual', input_folder))
    Imgs_bf = list(search_for_files('omni', input_folder))
    Imgs_syn = list(search_for_files('synth', input_folder))

    IoU_arr = np.empty(shape = (len(Imgs_G),2), dtype=np.ndarray)
    stats = np.empty(shape = (len(Imgs_G),2,8))
            
    for i, (Ig_name, I1_name, I2_name) in enumerate(zip(Imgs_G, Imgs_bf, Imgs_syn)):
        Ig = io.imread(os.path.join(input_folder, Ig_name))
        I1 = io.imread(os.path.join(input_folder, I1_name))
        I2 = io.imread(os.path.join(input_folder, I2_name))

        Ig = reorder_labels(Ig)
        I1 = reorder_labels(I1)
        I2 = reorder_labels(I2)

        IoU_arr[i,0], stats[i,0,7], stats[i,0,6], stats[i,0,5], counts_bf = find_seg_candidates2(Ig, I1)
        IoU_arr[i,1], stats[i,1,7], stats[i,1,6], stats[i,1,5], counts_s = find_seg_candidates2(Ig, I2)

        stats[i,0,0:4] = counts_bf
        stats[i,1,0:4] = counts_s

        stats[i,0,4] = len(np.unique(I1)) - 1
        stats[i,1,4] = len(np.unique(I2)) - 1

    
    IoUs_bf = np.hstack(IoU_arr[:,0])
    #IoUs_bf.flatten()
    IoUs_syn = np.hstack(IoU_arr[:,1])
    #IoUs_syn.flatten()

    return IoU_arr, IoUs_bf, IoUs_syn, stats

#%%
input_folder = 'K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/synthetic bacteria images/real images for cycleGAN/testing images/cropped/'

IoU_arr, IoUs_bf, IoUs_syn, stats = batch_comparison(input_folder)

#%%
fig, ax = plt.subplots(1,1)

ax.hist(IoUs_bf, bins=20, edgecolor = "black", label = "Bact_fluor_omni", alpha = 0.8)
ax.hist(IoUs_syn, bins=20, edgecolor = "black", label = "Synthetic model", alpha = 0.8)
ax = plt.gca()
plt.xlabel('Intersection over union')
plt.ylabel('cell count')
ax.legend(loc='upper right')
#fig.savefig("IoU_hist.svg", format = 'svg', dpi=300)
plt.show()

#%%
num_imgs = len(stats)
norm_stats = np.zeros((num_imgs,2), dtype = np.ndarray)
for i in range(num_imgs):
    for j in range(len(stats[0])):
        norm_stats[i,j] = np.divide(stats[i,j,0:5], stats[i,j][0])

stat_names = ("Cells found", "TP", "FP", "FN", "PQ")
stat_means = {
    'Bact_fluor_omni': (np.mean([x[4] for x in norm_stats[:,0]]), np.mean([x[1] for x in norm_stats[:,0]]), np.mean([x[2] for x in norm_stats[:,0]]), np.mean([x[3] for x in norm_stats[:,0]]), np.mean(stats[:,0,7])),
    'Synthetic model': (np.mean([x[4] for x in norm_stats[:,1]]), np.mean([x[1] for x in norm_stats[:,1]]), np.mean([x[2] for x in norm_stats[:,1]]), np.mean([x[3] for x in norm_stats[:,1]]), np.mean(stats[:,1,7]))
}

stat_sterr = {
    'Bact_fluor_omni': (np.std([x[4] for x in norm_stats[:,0]])/math.sqrt(num_imgs), np.std([x[1] for x in norm_stats[:,0]])/math.sqrt(num_imgs), np.std([x[2] for x in norm_stats[:,0]])/math.sqrt(num_imgs), np.std([x[3] for x in norm_stats[:,0]])/math.sqrt(num_imgs), np.std(stats[:,0,7])/math.sqrt(num_imgs)),
    'Synthetic model': (np.std([x[4] for x in norm_stats[:,1]])/math.sqrt(num_imgs), np.std([x[1] for x in norm_stats[:,1]])/math.sqrt(num_imgs), np.std([x[2] for x in norm_stats[:,1]])/math.sqrt(num_imgs), np.std([x[3] for x in norm_stats[:,1]])/math.sqrt(num_imgs), np.std(stats[:,1,7])/math.sqrt(num_imgs))
}

x = np.arange(len(stat_names))  # the label locations
width = 0.3  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in stat_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, align='edge', yerr = [2.18*x for x in stat_sterr.get(attribute)[:]], capsize = 3)
    ax.bar_label(rects, padding=3, fmt='\n%.2f')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Proportion of cells')
#ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, stat_names)
ax.legend(loc='upper right', ncols=3)
ax.set_ylim(0, 1.15)
#fig.savefig("seg_stats.svg", format = 'svg', dpi=300)

plt.show()

#%% MIXED batch processing functions
    
def batch_comparison(input_folder):
    Imgs = sorted(list(search_for_files('.tif', input_folder)))
    Imgs_G_c = sorted(list(search_for_files('circle_man', input_folder)))
    Imgs_G_r = sorted(list(search_for_files('rod_man', input_folder)))
    Imgs_syn_c = sorted(list(search_for_files('circle_synth', input_folder)))
    Imgs_syn_r = sorted(list(search_for_files('rod_synth', input_folder)))

    IoU_arr = np.empty(shape = (len(Imgs_G_c),2), dtype=np.ndarray)
    stats = np.empty(shape = (len(Imgs_G_c),2,8))
            
    for i, (Ig_c_name, Ig_r_name, I_c_name, I_r_name) in enumerate(zip(Imgs_G_c, Imgs_G_r, Imgs_syn_c, Imgs_syn_r)):
        Ig_c = io.imread(os.path.join(input_folder, Ig_c_name))
        Ig_r = io.imread(os.path.join(input_folder, Ig_r_name))
        Ic = io.imread(os.path.join(input_folder, I_c_name))
        Ir = io.imread(os.path.join(input_folder, I_r_name))

        #inputs Ic,Ir for brightfield
        Ir, double_count = remove_double_candidates(Ir, Ic)
        print(f'removed {double_count} cells from label for image {i}')

        Ig_c = reorder_labels(Ig_c)
        Ig_r = reorder_labels(Ig_r)
        Ic = reorder_labels(Ic)
        Ir = reorder_labels(Ir)

        IoU_arr[i,0], stats[i,0,7], stats[i,0,6], stats[i,0,5], counts_c = find_seg_candidates2(Ig_c, Ic)
        IoU_arr[i,1], stats[i,1,7], stats[i,1,6], stats[i,1,5], counts_r = find_seg_candidates2(Ig_r, Ir)

        stats[i,0,0:4] = counts_c
        stats[i,1,0:4] = counts_r

        stats[i,0,4] = len(np.unique(Ic)) - 1
        stats[i,1,4] = len(np.unique(Ir)) - 1

    
    IoUs_c = np.hstack(IoU_arr[:,0])
    #IoUs_bf.flatten()
    IoUs_r = np.hstack(IoU_arr[:,1])
    #IoUs_syn.flatten()

    return IoU_arr, IoUs_c, IoUs_r, stats

#%%
#input_folder = 'K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/Mixed testing 05-04-24/Brightfield/40X_mixed/edit_crop_256/testing_images/'
input_folder = 'K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/07.03.24/Pa Sa mix flat/mixed_flat_tiffs/cropped/testing images'

IoU_arr, IoUs_c, IoUs_r, stats = batch_comparison(input_folder)

#%%
fig, ax = plt.subplots(1,1)

ax.hist(IoUs_c, bins=20, edgecolor = "black", label = "S. aureus", alpha = 0.8)
ax.hist(IoUs_r, bins=20, edgecolor = "black", label = "P. aeruginosa", alpha = 0.8)
ax = plt.gca()
plt.xlabel('Intersection over union')
plt.ylabel('cell count')
ax.legend(loc='upper right')
#fig.savefig("IoU_hist.svg", format = 'svg', dpi=300)
plt.show()

#%%
num_imgs = len(stats)
norm_stats = np.zeros((num_imgs,2), dtype = np.ndarray)
for i in range(num_imgs):
    for j in range(len(stats[0])):
        norm_stats[i,j] = np.divide(stats[i,j,0:5], stats[i,j][0])

stat_names = ("Cells found", "TP", "FP", "FN", "PQ")
stat_means = {
    'S. aureus': (np.mean([x[4] for x in norm_stats[:,0]]), np.mean([x[1] for x in norm_stats[:,0]]), np.mean([x[2] for x in norm_stats[:,0]]), np.mean([x[3] for x in norm_stats[:,0]]), np.mean(stats[:,0,7])),
    'P. aeruginosa': (np.mean([x[4] for x in norm_stats[:,1]]), np.mean([x[1] for x in norm_stats[:,1]]), np.mean([x[2] for x in norm_stats[:,1]]), np.mean([x[3] for x in norm_stats[:,1]]), np.mean(stats[:,1,7]))
}

stat_sterr = {
    'S. aureus': (np.std([x[4] for x in norm_stats[:,0]])/math.sqrt(num_imgs), np.std([x[1] for x in norm_stats[:,0]])/math.sqrt(num_imgs), np.std([x[2] for x in norm_stats[:,0]])/math.sqrt(num_imgs), np.std([x[3] for x in norm_stats[:,0]])/math.sqrt(num_imgs), np.std(stats[:,0,7])/math.sqrt(num_imgs)),
    'P. aeruginosa': (np.std([x[4] for x in norm_stats[:,1]])/math.sqrt(num_imgs), np.std([x[1] for x in norm_stats[:,1]])/math.sqrt(num_imgs), np.std([x[2] for x in norm_stats[:,1]])/math.sqrt(num_imgs), np.std([x[3] for x in norm_stats[:,1]])/math.sqrt(num_imgs), np.std(stats[:,1,7])/math.sqrt(num_imgs))
}

x = np.arange(len(stat_names))  # the label locations
width = 0.3  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in stat_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, align='edge', yerr = [2.18*x for x in stat_sterr.get(attribute)[:]], capsize = 3)
    ax.bar_label(rects, padding=3, fmt='\n%.2f')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Proportion of cells')
#ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, stat_names)
ax.legend(loc='upper right', ncols=3)
ax.set_ylim(0, 1.19)
plt.savefig("K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/Multispecies 2D segmentation paper figures/confocal_mix_seg.eps", format = 'eps', dpi=300)
#fig.savefig("seg_stats.svg", format = 'svg', dpi=300)

plt.show()