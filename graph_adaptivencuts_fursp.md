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

Graph Constrained Unmixing w/ Normalized Cuts

```python
# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from SuperpixelCutsPy import *
# Configs for Notebooks
plt.rcParams["figure.figsize"] = [9,7]
np.set_printoptions(suppress=True)
```

```python
dataset_name = 'fields_data_2022'
h5_import = h5py.File("data/bhsi_2022.h5",'r+').get('Cube/resultarray/inputdata')
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
n_superpixels = 2500 #2500
slic_m_param = 1    #2
assignments, centers = superpixel.generate_SLIC_assignments(data = hyperspectral_cube,
                                                            n_superpixels = n_superpixels,
                                                            slic_m_param = slic_m_param)
superpixeled_cube, superpixel_library = superpixel.generate_SLIC_superpixels(data = hyperspectral_cube,
                                                                             assignments = assignments)
n_superpixels = len(np.unique(assignments))
```

```python
fig, ax = plt.subplots(1,2, dpi=100);
layer_preview = 20
ax[0].imshow(hyperspectral_cube[:,:,layer_preview]);
ax[1].imshow(superpixeled_cube[:,:,layer_preview])
ax[1].scatter(centers[:,1], centers[:,0], c='white', s=0.1);
ax[0].set_title(f'Original Image Layer {layer_preview}', fontsize = 8);
ax[1].set_title(f'Superpixeled Image n={len(np.unique(assignments))}', fontsize = 8);
```

```python
sigma_param = 0.01 # 0.1 -> 0.001           #0.01
spatial_limit = 25# 15 -> 25 in steps of 5 #15
spatial_beta_param = 0.025 #0.05
spatial_dmax_param = 8
ne = 6#number of endmembers

superpixel_cluster_labels, mean_cluster_spectra, int_results = normalized_cuts.graph_regularized_ncuts_admm(data=hyperspectral_cube,
                                                                                                            superpixel_library=superpixel_library,
                                                                                                            superpixel_centers=centers,
                                                                                                            superpixel_assignments=assignments,
                                                                                                            n_endmembers = ne,
                                                                                                            spectral_sigma2_param= sigma_param,
                                                                                                            spatial_kappa_param=spatial_limit,
                                                                                                            spatial_beta_param= spatial_beta_param,
                                                                                                            spatial_dmax_param = spatial_dmax_param,
                                                                                                            n_unmixing_iters = 100,
                                                                                                            spectral_metric='SAM')

labelled_img = normalized_cuts.assign_labels_onto_image(assignments, superpixel_cluster_labels)

_, superpixel_original_library = superpixel.generate_SLIC_superpixels(data = original_hyperspectral_cube,
                                                                      assignments = assignments)

#original_library = segmentation_evaluation.calc_mean_label_signatures(superpixel_original_library, superpixel_cluster_labels)
```

```python
plt.imshow(labelled_img);
```

```python
num_layers = min(int_results['abundance_results'].shape[2], 6)

fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 6))

for i in range(num_layers):
    axes[i].imshow(int_results['abundance_results'][:, :, i], cmap='viridis')
    axes[i].set_title(f'Initial Endmember {i+1}')
```

```python
#dict_keys(['loss', 'primal_residual', 'dual_residual', 'mean_abund_value', 'n_iters'])
view = 'mean_abund_value'
plt.plot(int_results['unmixing_history'][view]);
plt.title(view);
```

```python

```
