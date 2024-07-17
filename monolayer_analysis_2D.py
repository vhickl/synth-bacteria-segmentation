#%%

import math
import numpy as np
import skimage
from skimage import io
import os
import glob
    #from aicsimageio import AICSImage  
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import scipy
plt.rcParams["image.cmap"] = "gray"
import napari
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
import pandas as pd
import scipy.sparse.linalg as linalg
import time
from skimage.measure import regionprops
from collections import Counter
from scipy.optimize import curve_fit

def bwareafilt(mask, n=1, area_range=(0, np.inf)):
    """Extract objects from binary image by size """
    # For openCV > 3.0 this can be changed to: areas_num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    labels = measure.label(mask.astype('uint8'), background=0)
    area_idx = np.arange(1, np.max(labels) + 1)
    areas = np.array([np.sum(labels == i) for i in area_idx])
    inside_range_idx = np.logical_and(areas >= area_range[0], areas <= area_range[1])
    area_idx = area_idx[inside_range_idx]
    areas = areas[inside_range_idx]
    keep_idx = area_idx[np.argsort(areas)[::-1][0:n]]
    kept_areas = areas[np.argsort(areas)[::-1][0:n]]
    if np.size(kept_areas) == 0:
        kept_areas = np.array([0])
    if n == 1:
        kept_areas = kept_areas[0]
    kept_mask = np.isin(labels, keep_idx)
    return kept_mask, kept_areas

def cell_filt_measure(mask, n=None, area_range=(0, np.inf), margin = 5):
    if n == None:
        n = np.max(mask)

    props = regionprops(mask)
    centroids = np.array([x['centroid'] for x in props])
    areas = np.array([x['area'] for x in props])
    orientations = np.array([x['orientation'] for x in props])

    area_idx = np.arange(1, np.max(mask) + 1)
    inside_range_idx = np.logical_and(np.logical_and(np.logical_and(np.logical_and(centroids[:,0] > margin, centroids[:,1] > margin), centroids[:,0] < cell_image.shape[0]-margin), centroids[:,1] < cell_image.shape[1]-margin), np.logical_and(areas >= area_range[0], areas <= area_range[1]))
    area_idx = area_idx[inside_range_idx]
    areas = areas[inside_range_idx]
    kept_centroids = centroids[inside_range_idx]
    kept_orientations = orientations[inside_range_idx]
    keep_idx = area_idx[np.argsort(areas)[::-1][0:n]]
    kept_areas = areas[np.argsort(areas)[::-1][0:n]]
    if np.size(kept_areas) == 0:
        kept_areas = np.array([0])
    if n == 1:
        kept_areas = kept_areas[0]
    kept_mask = np.isin(mask, keep_idx)
    new_mask = kept_mask*mask
    return new_mask, kept_areas, kept_centroids, kept_orientations

from cellpose_omni import models, core
from cellpose_omni.models import MODEL_NAMES 
import cellpose_omni

import ncolor
from omnipose.utils import rescale, crop_bbox
import fastremap    
from scipy.spatial import KDTree

#%%
file = 'K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/synthetic bacteria images/real images for cycleGAN/dilute4_minIP.tif'

#cell_image = io.imread('K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/Pa knockout on thin PDMS flat,ridges 22-06-23/edit_PDMS10_flat2.tif')
cell_image = io.imread(file)
#cell_image = io.imread('E coli GFPuv on ridges 9-8-23/brightfield/ridges10_5_crops/crop2_edit_rotate_ridges10_5_brightfield3.tif')
#binary_images = io.imread('ridges25_5_zstack_tseries_deconvolution_strong_tiffs/minIP/iso/phans_erode5_iso_edit_3Dmedian2_minIP.tif')
#test_image = io.imread('T:/users/hivi/Omnipose_training_data/rounded_rects_512_1/test/20x6_clustered4432_003.tif')

#del images[-1] #remove last z-slice (removes spurious objects at top edge)
spacing = np.array([0.066, 0.066]) #axes are [y, x]   

#viewer2 = napari.view_image(cell_image, contrast_limits=[0, 255],scale=spacing)

#model_name = 'bact_fluor_omni'  # 'bact_phase_omni'

#model = models.CellposeModel(gpu=True, model_type=model_name)
model = models.CellposeModel(gpu=False, nchan=1, nclasses=3, pretrained_model='K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/synthetic bacteria images/trained models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_train_2024_02_05_10_23_07.897430_epoch_1999', dim = 2)

chans= [0,0]
verbose = 0 # turn on if you want to see more output 
#use_gpu = use_GPU #defined above
#transparency = True # transparency in flow output
#rescale=None # give this a number if you need to upscale or downscale   your images
#omni = True # we can turn off Omnipose mask reconstruction, not advised 
# default is .4, but only needed if there are spurious masks to clean up; slows down output
#niter = None # None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation 
#resample = True #whether or not to run dynamics on rescaled grid or original grid 
#augment = False # average the outputs from flipped (augmented) images; slower, usually not needed 
#tile = False # break up image into smaller parts then stitch together
#affinity_seg = 0 #new feature, stay tuned...

#%% load mask

mask = io.imread('K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/synthetic bacteria images/real images for cycleGAN/flat6_avg4_edit_masks_synth.tif')

#%% 2D

tic = time.time() 
mask, flow, style = model.eval(cell_image, mask_threshold=0.0, 
    flow_threshold=0.0, diameter=0.0, invert=False, cluster=True, net_avg=False, do_3D=False, omni=True)

net_time = time.time() - tic
print('total segmentation time: {}s'.format(net_time))
plt.imshow(mask)

cmap = mpl.colormaps.get_cmap('viridis')

pic = cmap(rescale(ncolor.label(mask)))

pic[:,:,-1] = mask>0 # alpha 

plt.imshow(pic)

viewer = napari.view_image(cell_image)
viewer.add_labels(mask)
#%% save
cellpose_omni.io.imsave(file[:-4]+'_masks_synth.tif',mask)

#%% 3D
import torch

test3Dimage = io.imread("T:/users/hivi/Omnipose_training_data/3D/zstack1_t1_deconvolution3_edit.tif")
#test3Dimage = io.imread("T:/users/hivi/Omnipose_training_data/3D/training_set_test2/test/N350.tif")
#spacing = np.array([1, 1, 1]) 

#model = models.CellposeModel(gpu=True, nchan=1, nclasses=3, pretrained_model='T:/users/hivi/Omnipose_training_data/3D/training_set_test2/train/models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_3_train_2024_01_17_09_24_29.266476_epoch_192', dim = 3)
model = models.CellposeModel(gpu=True, nchan=1, nclasses=3, pretrained_model='T:/users/hivi/Omnipose_training_data/3D/training_set_test2/train/models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_3_train_2024_01_30_13_54_11.311315_epoch_499', dim = 3)

torch.cuda.empty_cache()
tic = time.time() 
mask, flow, _ = model.eval(test3Dimage, mask_threshold=0, channels = None,
    flow_threshold=0.0, diameter=0.0, invert=False, cluster=True, net_avg=False, do_3D=False, omni=True, interp = False, resample = True, transparency=True, compute_masks=1, verbose=1, flow_factor = 5, batch_size=8, hysteresis = True, bsize = 160)

net_time = time.time() - tic
print('total segmentation time: {}s'.format(net_time))

viewer = napari.view_image(test3Dimage)
viewer.add_labels(mask)


#%%
props = regionprops(mask)
centroids = np.array([x['centroid'] for x in props])
orientations = np.array([x['orientation'] for x in props])
areas = np.array([x['area'] for x in props])

# filter out object <=d pixels from boundaries and less than a in area

d = 10 #distance from edges in pixels
a = 100 #MAKE SURE INEQUALITY IS SET CORRECTLY BELOW
filt_centroids = centroids[np.logical_and(np.logical_and(np.logical_and(np.logical_and(centroids[:,0] > d, centroids[:,1] > d), centroids[:,0] < cell_image.shape[0]-d), centroids[:,1] < cell_image.shape[1]-d), areas > a)]
filt_orientations = orientations[np.logical_and(np.logical_and(np.logical_and(np.logical_and(centroids[:,0] > d, centroids[:,1] > d), centroids[:,0] < cell_image.shape[0]-d), centroids[:,1] < cell_image.shape[1]-d), areas > a)]
filt_areas = areas[np.logical_and(np.logical_and(np.logical_and(np.logical_and(centroids[:,0] > d, centroids[:,1] > d), centroids[:,0] < cell_image.shape[0]-d), centroids[:,1] < cell_image.shape[1]-d), areas > a)]

#reduce angles
filt_angles = abs(filt_orientations)
plt.hist(filt_angles, bins=25)
ax = plt.gca()
plt.xlabel('angle [rad]')
plt.ylabel('count')
plt.show()

#np.savetxt('filt_data.csv', data, delimiter=',')

#%% load mask without filtered cells

filt_mask = io.imread('K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/synthetic bacteria images/real images for cycleGAN/flat6_avg4_edit_filt_masks_synth.tif')

#%% make new mask without filtered cells
filt_mask = np.zeros(shape=mask.shape)
bad_inds = np.where(np.logical_not(np.logical_and(np.logical_and(np.logical_and(np.logical_and(centroids[:,0] > d, centroids[:,1] > d), centroids[:,0] < cell_image.shape[0]-d), centroids[:,1] < cell_image.shape[1]-d), areas > a)))

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i,j] == 0:
            filt_mask[i,j] = 0
        elif np.any(bad_inds[0] == mask[i,j]-1):
            filt_mask[i,j] = 0
        else:
            filt_mask[i,j] = mask[i,j] - np.where(np.sort(np.append(bad_inds[0], mask[i,j])) == mask[i,j])[0][0] #subtract number of cell being filtered out that have labels less than current cell
    #print(f'row {i} done')

filt_mask = filt_mask.astype(int)
plt.imshow(filt_mask)

#%% save
cellpose_omni.io.imsave(file[:-4]+'_filt_masks_synth.tif',filt_mask)

#%% dumb code to color objects by orientation
mask_orient = np.zeros(shape=mask.shape)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i,j] == 0:
            mask_orient[i,j] = 0
        else:
            mask_orient[i,j] = orientations[mask[i,j]-1] + math.pi/2

cmap = mpl.colormaps.get_cmap('hot')
pic = cmap(rescale(mask_orient))
pic[:,:,-1] = mask_orient>0 # alpha 
plt.imshow(pic)

#%% calculate angle differences as a function of distance

pix = spacing[0] #pixel size in um
from scipy.spatial import KDTree

filt_centroids_um = pix*filt_centroids
kd_tree = KDTree(filt_centroids_um)

pairs = kd_tree.query_pairs(r=20)   

n = len(pairs)
near_rel_angles2 = np.empty([n,2])
m=0
for (i,j) in pairs:
   near_rel_angles2[m,0] = math.sqrt((filt_centroids_um[i,0]-filt_centroids_um[j,0])**2 + (filt_centroids_um[i,1]-filt_centroids_um[j,1])**2)
   near_rel_angles2[m,1] = math.pi/2 - abs(abs(filt_orientations[i] - filt_orientations[j]) - math.pi/2)
   m += 1

x = np.array([x[0] for x in near_rel_angles2])
y = np.array([x[1] for x in near_rel_angles2])

nbins=30
N, _ = np.histogram(x, bins=nbins)
sy, _ = np.histogram(x, bins=nbins, weights=y)
sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
mean = sy / N
std = np.sqrt(sy2/N - mean*mean)
ste = std/np.sqrt(N)

plt.plot(x, y, 'bo', markersize=2)
plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=ste, fmt='r-')
plt.show()

# %% alternate code for loop 
#DOESN'T WORK, ALL_DIST IS NOT CORRECTLY CALCULATING DIFFERENCES BETWEEN LOCATIONS

n = filt_orientations.shape[0]
N = filt_orientations.shape[0]**2

x = filt_centroids[:,0]
y = filt_centroids[:,1]
XX, YY = np.meshgrid(x,y)
all_dist = pix*np.sqrt(XX**2 + YY**2)
valid = all_dist < 20
x_near, y_near = XX[valid] , YY[valid]
dist_near = all_dist[valid].flatten()

UU, VV = np.meshgrid(filt_orientations, filt_orientations)
all_orient = math.pi/2 - abs(abs(UU - VV) - math.pi/2)
orient_near = all_orient[valid].flatten()

x2 = dist_near
y2 = orient_near
nbins = 40

n, _ = np.histogram(x2, bins=nbins)
sy, _ = np.histogram(x2, bins=nbins, weights=y2)
sy2, _ = np.histogram(x2, bins=nbins, weights=y2*y2)
mean = sy / n
std = np.sqrt(sy2/n - mean*mean)
ste = std/np.sqrt(n)

plt.plot(x2, y2, 'bo', markersize=2)
plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=ste, fmt='r-')
plt.show()

#%% find and plot order parameter
near_distances = near_rel_angles2[:,0]

near_single_op = np.empty([n,1]) #"order parameter" but for individual cell pairs, rather than whole region
near_single_op = np.array([2*math.cos(filt_orientations[i] - filt_orientations[j])**2 - 1 for (i,j) in pairs])

x = near_distances
y = near_single_op

nbins=30
N, _ = np.histogram(x, bins=nbins)
sy, _ = np.histogram(x, bins=nbins, weights=y)
sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
mean = sy / N
std = np.sqrt(sy2/N - mean*mean)
ste = std/np.sqrt(N)

plt.plot(x, y, 'bo', markersize=2)
plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=ste, fmt='r-')
plt.show()

plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=ste, fmt='ro', capsize = 3, ms=4, ecolor = 'k', mec='k', mew=0.5 )
plt.yscale('log')
#plt.xscale('log')
plt.xlabel('distance [um]')
plt.ylabel('order parameter')
plt.show()

#%% calculate orientation vectors and properly find order parameter or spatial correlation function

norm_vecs = np.array([[np.sin(x),np.cos(x)] for x in filt_orientations])
cmap = mpl.colormaps.get_cmap('hot')
"""
plt.quiver(filt_centroids_um[:,1], filt_centroids_um[:,0], norm_vecs[:,0], norm_vecs[:,1], [filt_orientations], pivot='middle', headaxislength = 0, headlength=0, headwidth=1, cmap='hot')
plt.colorbar
plt.savefig('flat_quiver.eps', format='eps', dpi=600)
plt.show()
"""


#%% local cell density
pairs_10 = kd_tree.query_pairs(r=10)   
local_counts = np.zeros(len(filt_centroids_um), dtype=int)

flat_pairs = np.array(list(pairs_10)).flatten()
ind_counts = Counter(flat_pairs)

max_coord = len(cell_image)*spacing[0]
for i, x in enumerate(filt_centroids_um):
    if x[0]<10 or x[0]>max_coord-10 or x[1]<10 or x[1]>max_coord-10: #check if cell is within 20 um of the edge
        continue

    local_counts[i] = int(ind_counts.get(i))

nz_counts = local_counts[local_counts >= 1]

#%% individual cell order parameters
cell_ops = np.zeros(len(filt_orientations))
neighbor_lists = kd_tree.query_ball_tree(kd_tree, 10)

for i, neighbors in enumerate(neighbor_lists):
    for j in neighbors:
        if i != j:
            #cell_ops[i] += (2*math.cos(filt_orientations[i] - filt_orientations[j])**2 - 1)/len(neighbors)
            cell_ops[i] += (2*math.cos(filt_orientations[i] - filt_orientations[j])**2)/len(neighbors) #add 1 to each order parameter to make plotting work in next step, needs to be added back in for calculations later

#%% recolor by density
mask_density = np.zeros(shape=mask.shape)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if filt_mask[i,j] == 0:
            mask_density[i,j] = 0
        else:
            mask_density[i,j] = local_counts[filt_mask[i,j]-1]

cmap = mpl.colormaps.get_cmap('hot')
pic = cmap(rescale(mask_density))
#pic[:,:,-1] = mask_density>0 # alpha
plt.imshow(pic)            

            
#%% recolor by order parameter
mask_op = np.zeros(shape=mask.shape)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if filt_mask[i,j] != 0:
            mask_op[i,j] = cell_ops[filt_mask[i,j]-1]

cmap = mpl.colormaps.get_cmap('viridis')
pic = cmap(rescale(mask_op))
pic[:,:,-1] = mask_op>0 # alpha 
fig = plt.figure()
plt.imshow(pic)

#%% filter out cells near edges for cell order parameters, plot, and plot against local counts

cell_ops2 = np.zeros(len(local_counts))
for i,x in enumerate(cell_ops):
    if local_counts[i] != 0:
        cell_ops2[i] = x

mask_op2 = np.zeros(shape=mask.shape)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if filt_mask[i,j] != 0:
            mask_op2[i,j] = cell_ops2[filt_mask[i,j]-1]

cmap = mpl.colormaps.get_cmap('hot')
pic = cmap(rescale(mask_op2))
pic[:,:,-1] = mask_op2>0 # alpha 
fig = plt.figure()
plt.imshow(pic)

nz_cell_ops = cell_ops2[cell_ops2 > 0]
fig = plt.figure()
plt.scatter(nz_counts, nz_cell_ops - np.ones(shape=nz_cell_ops.shape))
ax = plt.gca()
#ax.set_xlim([xmin, xmax])
ax.set_ylim([-1, 1])

#%% Batch processing function for large images of dense monolayers

def search_for_files(search_str, path):
    for f in os.listdir(path):
        if search_str in f:
            yield f

def batch_statistics(input_folder):
    Imgs = list(search_for_files('.tif', input_folder))

    op_data = np.empty(shape = (len(Imgs),4), dtype=np.ndarray)
    nz_counts = np.empty(len(Imgs), dtype=np.ndarray)
    cell_ops = np.empty(len(Imgs), dtype=np.ndarray)
    cell_ops2 = np.empty(len(Imgs), dtype=np.ndarray)
    nz_cell_ops = np.empty(len(cell_ops2), dtype=np.ndarray)
    asp_ratios = np.empty(len(Imgs), dtype=np.ndarray)
    major_axes = np.empty(len(Imgs), dtype=np.ndarray)
    minor_axes = np.empty(len(Imgs), dtype=np.ndarray)
    cell_areas = np.empty(len(Imgs), dtype=np.ndarray)

    model = models.CellposeModel(gpu=False, nchan=1, nclasses=3, pretrained_model='K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/synthetic bacteria images/trained models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_train_2024_02_05_10_23_07.897430_epoch_1999', dim = 2)
   
    for ind,Img in enumerate(Imgs):
        I = io.imread(os.path.join(input_folder,Img))
        if Img.startswith("PDMS") or Img.startswith("dilute"):
            pix = 0.066
        else:
            pix = 0.085

        mask_list = list(search_for_files('mask', input_folder+'masks/'))
        if Img[:-4]+'_masks_synth.tif' in mask_list:
            mask = io.imread(os.path.join(input_folder+'masks/', Img[:-4]+'_masks_synth.tif'))
            print(f'mask found, skipped segmentation for image {ind}')
        else:    
            tic = time.time() 
            mask, flow, style = model.eval(I, mask_threshold=0.0, 
                flow_threshold=0.0, diameter=0.0, invert=False, cluster=True, net_avg=False, do_3D=False, omni=True)

            net_time = time.time() - tic
            print('total segmentation time: {}s'.format(net_time))
            cellpose_omni.io.imsave(os.path.join(input_folder+'masks/', Img[:-4]+'_masks_synth.tif'),mask)
        
        props = regionprops(mask)
        centroids = np.array([x['centroid'] for x in props])
        orientations = np.array([x['orientation'] for x in props])
        areas = np.array([x['area'] for x in props])
        ax_major = np.array([x['axis_major_length'] for x in props])
        ax_minor = np.array([x['axis_minor_length'] for x in props])

        # filter out object <=d pixels from boundaries and less than a in area

        d = 10 #distance from edges in pixels
        a = 100 #MAKE SURE INEQUALITY IS SET CORRECTLY BELOW
        filt_centroids = centroids[np.logical_and(np.logical_and(np.logical_and(np.logical_and(centroids[:,0] > d, centroids[:,1] > d), centroids[:,0] < I.shape[0]-d), centroids[:,1] < I.shape[1]-d), areas > a)]
        filt_orientations = orientations[np.logical_and(np.logical_and(np.logical_and(np.logical_and(centroids[:,0] > d, centroids[:,1] > d), centroids[:,0] < I.shape[0]-d), centroids[:,1] < I.shape[1]-d), areas > a)]
        filt_areas = areas[np.logical_and(np.logical_and(np.logical_and(np.logical_and(centroids[:,0] > d, centroids[:,1] > d), centroids[:,0] < I.shape[0]-d), centroids[:,1] < I.shape[1]-d), areas > a)]
        filt_ax_major = ax_major[np.logical_and(np.logical_and(np.logical_and(np.logical_and(centroids[:,0] > d, centroids[:,1] > d), centroids[:,0] < I.shape[0]-d), centroids[:,1] < I.shape[1]-d), areas > a)]
        filt_ax_minor = ax_minor[np.logical_and(np.logical_and(np.logical_and(np.logical_and(centroids[:,0] > d, centroids[:,1] > d), centroids[:,0] < I.shape[0]-d), centroids[:,1] < I.shape[1]-d), areas > a)]
        filt_asp_ratio = np.divide(filt_ax_major,filt_ax_minor)

        filt_centroids_um = pix*filt_centroids
        kd_tree = KDTree(filt_centroids_um)

        pairs = kd_tree.query_pairs(r=20)   

        n = len(pairs)
        near_rel_angles2 = np.empty([n,2])
        m=0
        for (i,j) in pairs:
            near_rel_angles2[m,0] = math.sqrt((filt_centroids_um[i,0]-filt_centroids_um[j,0])**2 + (filt_centroids_um[i,1]-filt_centroids_um[j,1])**2)
            near_rel_angles2[m,1] = math.pi/2 - abs(abs(filt_orientations[i] - filt_orientations[j]) - math.pi/2)
            m += 1

        near_distances = near_rel_angles2[:,0]

        near_single_op = np.empty([n,1]) #"order parameter" but for individual cell pairs, rather than whole region
        near_single_op = np.array([2*math.cos(filt_orientations[i] - filt_orientations[j])**2 - 1 for (i,j) in pairs])

        x = near_distances
        y = near_single_op

        nbins=30
        N, _ = np.histogram(x, bins=nbins)
        sy, _ = np.histogram(x, bins=nbins, weights=y)
        sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
        mean = sy / N
        std = np.sqrt(sy2/N - mean*mean)
        ste = std/np.sqrt(N)

        op_data[ind] = [(_[1:] + _[:-1])/2, mean, ste, np.mean(filt_asp_ratio)]

        pairs_10 = kd_tree.query_pairs(r=10)   
        local_counts = np.zeros(len(filt_centroids_um), dtype=int)

        flat_pairs = np.array(list(pairs_10)).flatten()
        ind_counts = Counter(flat_pairs)

        max_coord = len(I)*pix
        for i, x in enumerate(filt_centroids_um):
            if x[0]<10 or x[0]>max_coord-10 or x[1]<10 or x[1]>max_coord-10: #check if cell is within 10 um of the edge
                continue

            local_counts[i] = int(ind_counts.get(i) or 0)

        nz_counts[ind] = local_counts[local_counts >= 1]

        cell_ops_tmp = np.zeros(len(filt_orientations))
        neighbor_lists = kd_tree.query_ball_tree(kd_tree, 5)

        for i, neighbors in enumerate(neighbor_lists):
            for j in neighbors:
                if i != j:
                    cell_ops_tmp[i] += (2*math.cos(filt_orientations[i] - filt_orientations[j])**2 - 1)/(len(neighbors)-1)
                    #cell_ops_tmp[i] += (2*math.cos(filt_orientations[i] - filt_orientations[j])**2)/len(neighbors) #add 1 to each order parameter to make plotting work in next step, needs to be added back in for calculations later

        cell_ops[ind] = cell_ops_tmp

        cell_ops2_tmp = np.zeros(len(local_counts))
        asp_ratios_tmp = np.zeros(len(local_counts))
        ax_major_tmp = np.zeros(len(local_counts))
        ax_minor_tmp = np.zeros(len(local_counts))
        for i,x in enumerate(cell_ops_tmp):
            if local_counts[i] != 0:
                cell_ops2_tmp[i] = x
                asp_ratios_tmp[i] = filt_asp_ratio[i]
                ax_major_tmp[i] = filt_ax_major[i]
                ax_minor_tmp[i] = filt_ax_minor[i]

        cell_ops2[ind] = cell_ops2_tmp
        nz_cell_ops[ind] = cell_ops2_tmp[local_counts >= 1]
        asp_ratios[ind] = asp_ratios_tmp
        major_axes[ind] = filt_ax_major*pix
        minor_axes[ind] = filt_ax_minor*pix
        cell_areas[ind] = filt_areas*pix**2

    return op_data, nz_counts, cell_ops2, nz_cell_ops, asp_ratios, major_axes, minor_axes, cell_areas

#%%
input_folder = 'K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/synthetic bacteria images/real images for cycleGAN/dense/'

op_data, nz_counts, cell_ops2, nz_cell_ops, asp_ratios, major_axes, minor_axes, cell_areas = batch_statistics(input_folder)

#%%plotting

img_names = list(search_for_files('.tif', input_folder))

for i,x in enumerate(op_data[:]):
    plt.errorbar(x[0], x[1], yerr=x[2], fmt='o', capsize = 3, ms=4, ecolor = 'k', mec='k', mew=0.5, mfc=plt.cm.hot((op_data[i,3]-np.min(op_data[:,3]))/(np.max(op_data[:,3])-np.min(op_data[:,3]))))

plt.yscale('log')
#plt.xscale('log')
plt.xlabel('distance [um]')
plt.ylabel('order parameter')
plt.ylim([0.0007, 1.3])
plt.legend(img_names[:])
plt.show()

#%% averaging and fitting
mean_length = np.mean(np.concatenate(major_axes).ravel())
mean_width = np.mean(np.concatenate(minor_axes).ravel())
dist_means = np.zeros(30)
op_means = np.zeros(30)
ste_means = np.zeros(30)
for i in range(30):
    dist_means[i] = np.mean([x[0][i] for x in op_data])/mean_length
    op_means[i] = np.mean([x[1][i] for x in op_data])
    ste_means[i] = np.sqrt(sum(y*y for y in [x[2][i] for x in op_data]))

def power_law(x, a, b):
    return a * np.power(x, b)  
def exp_law(x, a, b):
    return a * np.exp(-b * x)

popt, pcov = curve_fit(exp_law, dist_means[1:], op_means[1:], bounds=([0.5, 0], [5, 10]))
x_reg = np.linspace(1.5,12,1000)
plt.errorbar(dist_means, op_means, yerr=ste_means, fmt='o', capsize = 3, ms=4, ecolor = 'k', mec='k', mew=0.5)
plt.plot(dist_means[1:], exp_law(dist_means[1:], *popt), '-r', linewidth = 3)
#plt.plot(x_reg, power_law(x_reg, 1, -1.9))
plt.yscale('log')
#plt.xscale('log')
plt.xlabel('r_ij/a')
plt.ylabel('order parameter')
plt.ylim([0.0007, 1.3])
#plt.savefig("K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/Multispecies 2D segmentation paper figures/average_ops.eps", format = 'eps', dpi=300)
plt.show()


#%%
for i, (x,y) in enumerate(zip(nz_counts, nz_cell_ops)):
    print(np.mean(y))
    print(np.std(y)/math.sqrt(len(y)))
    plt.plot(x, y, 'o', ms=4, mec='k', mew=0.5, mfc=plt.cm.hot((op_data[i,3]-np.min(op_data[:,3]))/(np.max(op_data[:,3])-np.min(op_data[:,3]))))

ax = plt.gca()
ax.set_ylim(-1, 1)
#plt.legend(img_names[:])

#%% boxplots
x = np.array([np.mean(x) for x in nz_counts])
plt.plot(np.linspace(0.5, 13.5, 100), np.zeros(100), '--')
plt.boxplot([y for _,y in sorted(zip(x, nz_cell_ops))], labels=np.around(x/(4*math.pi*10**2), 3))
ax = plt.gca()
#ax.set_xlim(0, 200)
ax.set_ylim(-1, 1)
plt.xlabel('mean cell density (cells/um^2)')
plt.ylabel('single-cell order parameter')

#%% plot local density (counts), cell oder parameters, and aspect ratios
x = np.concatenate(nz_counts).ravel() + 1
y = np.concatenate(nz_cell_ops).ravel()
c0 = np.concatenate(asp_ratios).ravel()
c = c0[c0 > 0]

plt.scatter(c, y, c=plt.cm.hot(rescale(x)), s = 4, edgecolors = 'k', linewidths=0.2)
plt.xlabel('cell aspect ratio')
plt.ylabel('single-cell order parameter')

#%% binning and plotting
mean_area = np.mean(np.concatenate(cell_areas).ravel())
n_bins = 12
edges = np.linspace(np.min(x),np.max(x), n_bins + 1)
y_bins = np.empty(n_bins, dtype = np.ndarray)

n=0
for i, edge in enumerate(edges[1:]):
    y_bins[i] = y[n:np.where(np.sort(np.append(x, edge)) == edge)[0][0]]
    n = np.where(np.sort(np.append(x, edge)) == edge)[0][0]

X = ((edges[1:]+edges[:-1])/(2*4*np.pi*10**2))*mean_area

plt.errorbar(X, [np.mean(y_bin) for y_bin in y_bins], yerr=[np.divide(np.std(y_bin),np.sqrt(len(y_bin))) for y_bin in y_bins], fmt='o', capsize = 3, ms=8, ecolor = 'k', mec='k', mew=0.5)
plt.xlabel('packing fraction')
plt.ylabel('order parameter')
#plt.savefig("K:/404-internal-Bac3Dmonolayers-VincentHickl-2023/Multispecies 2D segmentation paper figures/ops_density.eps", format = 'eps', dpi=300)
plt.show()

edges = np.linspace(np.min(c),np.max(c), n_bins + 1)
y_bins = np.empty(n_bins, dtype = np.ndarray)

n=0
for i, edge in enumerate(edges[1:]):
    y_bins[i] = y[n:np.where(np.sort(np.append(c, edge)) == edge)[0][0]]
    n = np.where(np.sort(np.append(c, edge)) == edge)[0][0]

X = (edges[1:]+edges[:-1])/2

plt.errorbar(X, [np.mean(y_bin) for y_bin in y_bins], yerr=[np.divide(np.std(y_bin),np.sqrt(len(y_bin))) for y_bin in y_bins], fmt='o', capsize = 3, ms=8, ecolor = 'k', mec='k', mew=0.5)
plt.xlabel('cell aspect ratio')
plt.ylabel('order parameter')
plt.show()

#%% Batch processing for phase contast images of E coli on ridges
import pims

frames = pims.open('E coli GFPuv on ridges 9-8-23/brightfield/ridges10_5_crops/crop*_edit_rotate_ridges10_5_brightfield3.tif')
N = len(frames)
pix = 0.1 #um per pix

orientations = np.array([])
centroids = np.empty([0,2])
areas = np.array([])
rel_angles = np.empty([0,2])

model_name = 'bact_phase_omni'  # 'bact_fluor_omni'

model = models.CellposeModel(gpu=False, model_type=model_name)

for x in frames:
    tic = time.time() 
    mask, flow, style = model.eval(x, channels=[0, 0], mask_threshold=-3.5, 
        flow_threshold=0.0, diameter=0.0, invert=False, cluster=True, net_avg=False, do_3D=False, omni=True)

    net_time = time.time() - tic
    print('total segmentation time: {}s'.format(net_time))

    cmap = mpl.colormaps.get_cmap('viridis')
    pic = cmap(rescale(ncolor.label(mask)))
    pic[:,:,-1] = mask>0 # alpha 
    plt.imshow(pic)
    plt.show()

    new_mask , kept_areas, kept_centroids, kept_orientations = cell_filt_measure(mask, area_range=(50,1000), margin = 5)
   
    """ #doesn't work because new_mask still has old values (cells labeled with values greater than number of kept cells)
        #but kept_orientation only has indices up to new number of cells
    new_mask_orient = np.zeros(shape=new_mask.shape)

    for i in range(new_mask.shape[0]):
        for j in range(new_mask.shape[1]):
            if new_mask[i,j] == 0:
                new_mask_orient[i,j] = 0
            else:
                new_mask_orient[i,j] = abs(kept_orientations[new_mask[i,j]-1])
   
    cmap = mpl.colormaps.get_cmap('hot')
    pic = cmap(rescale(new_mask_orient))
    pic[:,:,-1] = new_mask_orient>0 # alpha 
    plt.imshow(pic)
    plt.show()
    """
    pic = cmap(rescale(ncolor.label(new_mask)))
    pic[:,:,-1] = new_mask>0 # alpha 
    plt.imshow(pic)
    plt.show()

    props = regionprops(new_mask)
    centroids = np.append(centroids , kept_centroids , axis = 0)
    orientations = np.append(orientations , kept_orientations)
    areas = np.append(areas , kept_areas)

    temp = np.empty([0,2])
    for u, x in zip(kept_orientations, kept_centroids):
        for v, y in zip(kept_orientations, kept_centroids):
            temp = np.append(temp, [[pix*math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2), math.pi/2 - abs(abs(u - v) - math.pi/2)]], axis = 0)

    rel_angles = np.append(rel_angles, temp, axis=0)

filt_angles = abs(orientations)
plt.hist(filt_angles, bins=15)
ax = plt.gca()
plt.xlabel('angle [rad]')
plt.ylabel('count')
#plt.savefig('ridges10_5_Ecoli_angles_hist.eps', format='eps')
plt.show()

#%% plotting code that doesn't work

#from cellpose_omni import plot
#import omnipose
#bdi = flow[-1]

#f = 10
#szX = mask.shape[-1]/mpl.rcParams['figure.dpi']*f
#szY = mask.shape[-2]/mpl.rcParams['figure.dpi']*f
#fig = plt.figure(figsize=(szY,szX*4))
#fig.patch.set_facecolor([0]*4)

#plot.show_segmentation(fig, omnipose.utils.normalize99(cell_image), 

#                        mask, flow, bdi, channels=chans, omni=True)

#plt.tight_layout()
#plt.show()
