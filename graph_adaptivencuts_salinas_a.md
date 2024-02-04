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
import scipy as sp
# Configs for Notebooks
plt.rcParams["figure.figsize"] = [9,7]
np.set_printoptions(suppress=True)
```

```python
dataset_name = 'Salinas A'
hyperspectral_cube = sp.io.loadmat("data/SalinasA_corrected.mat")['salinasA_corrected'] # Load Dataset
ground_truth = sp.io.loadmat("data/SalinasA_gt.mat")['salinasA_gt']
hyperspectral_cube = hyperspectral_cube[0:55,0:55,:]

#hyperspectral_cube[12,14,:] = hyperspectral_cube[12,10,:]
ground_truth = ground_truth[0:55,0:55]
ground_truth = np.vectorize(lambda x: {0: 0, 1:1, 10:2, 11:3, 12:4, 13:5, 14:6}[x])(ground_truth)
nx, ny, nb = hyperspectral_cube.shape
print(hyperspectral_cube.shape)
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
n_superpixels = 250 #2500
slic_m_param = 2  #2
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
ax[1].imshow(superpixeled_cube[:,:,layer_preview]);
ax[1].scatter(centers[:,1], centers[:,0], c='white', s=0.3);
ax[0].set_title(f'Original Image Layer {layer_preview}', fontsize = 8);
ax[1].set_title(f'Superpixeled Image n={len(np.unique(assignments))}', fontsize = 8);
```

```python
sigma_param = 0.005 # 0.1 -> 0.001           #0.01
spatial_limit = 25# 15 -> 25 in steps of 5 #15
spatial_beta_param = 0.005
spatial_dmax_param = 10
ne = 5#number of endmembers

labelled_img, normalized_signatures, int_results = normalized_cuts.graph_regularized_ncuts_admm(data=hyperspectral_cube,
                                                                                                superpixel_library=superpixel_library,
                                                                                                superpixel_centers=centers,
                                                                                                superpixel_assignments=assignments,
                                                                                                n_endmembers = ne,
                                                                                                spectral_sigma2_param= sigma_param,
                                                                                                spatial_kappa_param=spatial_limit,
                                                                                                spatial_beta_param= spatial_beta_param,
                                                                                                spatial_dmax_param = spatial_dmax_param,
                                                                                                n_unmixing_iters = 200,
                                                                                                spectral_metric='SAM')

original_library  = segmentation_evaluation.calc_mean_label_signatures(utility.cube_to_matrix(original_hyperspectral_cube),
                                                                        labelled_img.reshape(-1))

#original_library = segmentation_evaluation.calc_mean_label_signatures(superpixel_original_library, superpixel_cluster_labels)
```

```python
fig, ax = plt.subplots(1,2, dpi=100);
ax[0].imshow(hyperspectral_cube[:,:,layer_preview]);
ax[1].imshow((labelled_img+1)*(ground_truth != 0));
```

```python
num_layers = min(int_results['abundance_results'].shape[2], 5)

fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))

for i in range(num_layers):
    axes[i].imshow(int_results['abundance_results'][:, :, i], cmap='viridis')
    axes[i].set_title(f'Initial Endmember {i+1}')

```

```python
#dict_keys(['loss', 'primal_residual', 'dual_residual', 'mean_abund_value', 'n_iters'])
view = 'mean_abund_value'
plt.axhline(y=1, color='r', linestyle='--');
plt.plot(int_results['unmixing_history'][view]);
plt.title(view);
```

```python

```
