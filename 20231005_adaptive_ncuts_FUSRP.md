---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from SuperpixelCutsPy import *
# Configs for Notebooks
os.chdir('c:\\Users\\apopo\\Desktop\\Research')
plt.rcParams["figure.figsize"] = [9,7]
np.set_printoptions(suppress=True)
```

Dataset Importing & Preprocessing

```python
dataset_name = 'fields_data_2022'
h5_import = h5py.File("C:/Users/apopo/Desktop/Research/Data/fields_data_2022.h5",'r+').get('Cube/resultarray/inputdata')
hyperspectral_cube = np.array(h5_import)
hyperspectral_cube = np.moveaxis(np.array(hyperspectral_cube), [0], [2])
hyperspectral_cube = np.moveaxis(np.array(hyperspectral_cube), [0], [1])
hyperspectral_cube = hyperspectral_cube[5:205, 5:205, :].copy()
nx,ny,nb = hyperspectral_cube.shape
del h5_import
```

```python
preprocessing_pipeline = Preprocesser.Preprocesser(data = hyperspectral_cube)
#preprocessing_pipeline.gaussian_blur(blur_param = 0)
preprocessing_pipeline.singular_value_decomposition(n_svd = 5)
preprocessing_pipeline.layer_normalization()
hyperspectral_cube = preprocessing_pipeline.data.copy()
original_hyperspectral_cube = preprocessing_pipeline.original_data.copy()
```

```python
plt.imshow(hyperspectral_cube[:,:,0]);
```

SLIC Superpixel Generation

```python
n_superpixels = 2500 #2500
slic_m_param = 2    #2
assignments, centers = superpixel.generate_SLIC_assignments(data = hyperspectral_cube,
                                                            n_superpixels = n_superpixels,
                                                            slic_m_param = slic_m_param)
superpixeled_cube, superpixel_library = superpixel.generate_SLIC_superpixels(data = hyperspectral_cube,
                                                                             assignments = assignments)
n_superpixels = len(np.unique(assignments))
```

```python
fig, ax = plt.subplots(1,2, dpi=200);
layer_preview = 20
ax[0].imshow(hyperspectral_cube[:,:,layer_preview]);
ax[1].imshow(superpixeled_cube[:,:,layer_preview])
ax[1].scatter(centers[:,1], centers[:,0], c='white', s=0.1);
ax[0].set_title(f'Original Image Layer {layer_preview}', fontsize = 8);
ax[1].set_title(f'Superpixeled Image n={len(np.unique(assignments))}', fontsize = 8);
```

Spatial-Spectral Pixel Clustering

```python
sigma_param = 0.01 # 0.1 -> 0.001           #0.01
spatial_limit = 35# 15 -> 25 in steps of 5 #15
ne = 6#number of endmembers

superpixel_cluster_labels, mean_cluster_spectra = normalized_cuts.single_ncuts(data=hyperspectral_cube,
                                                                                superpixel_library=superpixel_library,
                                                                                superpixel_centers=centers,
                                                                                superpixel_assignments=assignments,
                                                                                n_endmembers=ne,
                                                                                spectral_sigma2_param=sigma_param,
                                                                                spatial_kappa_param=spatial_limit,
                                                                                spectral_metric='EUCLIDEAN')

labelled_img = normalized_cuts.assign_labels_onto_image(assignments, superpixel_cluster_labels)
```

```python
_, superpixel_original_library = superpixel.generate_SLIC_superpixels(data = original_hyperspectral_cube,
                                                                      assignments = assignments)

original_library = segmentation_evaluation.calc_mean_label_signatures(superpixel_original_library, superpixel_cluster_labels)
```

```python
fig, ax = plt.subplots(1,3, figsize=(19,5), dpi=200);
layer_preview = 20
#n_layers = 60
cmap = plt.get_cmap('Spectral', ne)
colors = cmap(list(np.unique(assignments)))

ax[0].imshow(hyperspectral_cube[:,:,layer_preview]);
ax[0].set_title(f'Original Image \n Layer {layer_preview}', fontsize = 5);

im = ax[1].imshow(labelled_img, cmap = cmap, vmin = 0);
ax[1].scatter(centers[:,1], centers[:,0], c='black', s=0.5);
ax[1].set_title(f'Spatial-Spectral Pixel Clustering Results \n n_superpixels = {n_superpixels}, σ = {sigma_param}, k = {spatial_limit}' , fontsize = 5);

for i in range(ne):
    ax[2].plot(original_library[:,i], color=colors[i])
ax[2].set_title(f'Extracted Mean Superpixel Endmember Signatures \n n_superpixels = {n_superpixels}, σ = {sigma_param}, k = {spatial_limit}', fontsize = 5);

fig.subplots_adjust(right=0.825)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax);
```

Split up a chunk further


```python
chunk_sigma_param = 0.01 # 0.1 -> 0.001        
chunk_spatial_limit = 35
label_to_subsegment = 3
subsegmented_labels, subsegmented_library = normalized_cuts.subsegment(data = hyperspectral_cube,
                                                superpixel_library = superpixel_library,
                                                superpixel_centers = centers,
                                                superpixel_assignments = assignments,
                                                segmented_labels = superpixel_cluster_labels,
                                                subsegment_label = 3,
                                                n_subsegments = 2,
                                                spectral_param = chunk_sigma_param,
                                                spatial_param = chunk_spatial_limit)

_, superpixel_original_library = superpixel.generate_SLIC_superpixels(data = original_hyperspectral_cube,
                                                                      assignments = assignments)

subsegmented_original_library = segmentation_evaluation.calc_mean_label_signatures(superpixel_original_library, subsegmented_labels)
```

```python
fig, ax = plt.subplots(1,3, figsize=(19,5), dpi=200);
layer_preview = 20
iter_preview = 0
n_layers = 60
n_e = len(np.unique(subsegmented_labels))
cmap = plt.get_cmap('Dark2', n_e)
colors = cmap(list(np.unique(subsegmented_labels)))

ax[0].imshow(hyperspectral_cube[:,:,layer_preview]);
ax[0].set_title(f'Original Image \n Layer {layer_preview}', fontsize = 5);

ax[1].imshow(hyperspectral_cube[:,:,layer_preview], alpha = 0.9);
im = ax[1].imshow(normalized_cuts.assign_labels_onto_image(assignments, subsegmented_labels), cmap = cmap, alpha= 0.7, vmin = 0);
ax[1].scatter(centers[:,1], centers[:,0], c='black', s=0.1);
ax[1].set_title(f'Subsegmented Image' , fontsize = 5);

for i in range(n_e):
    ax[2].plot(subsegmented_original_library[:,i], color=colors[i])
ax[2].set_title(f'Extracted Signatures' , fontsize = 5);

fig.subplots_adjust(right=0.825)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax);
```

```python

```
