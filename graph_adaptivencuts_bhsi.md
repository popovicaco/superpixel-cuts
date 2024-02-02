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

**Graph Laplacian Estimation**

<!-- #region -->
**Graph Constrained Constrained Unmixing**


For simplicity, we denote the postive, sum-to-one constraint set as $\Delta = \{A | A \geq 0, 1_n^T A = 1_n  \}$. We express the A subproblem as follows:
$$
\min_{A} \frac{1}{2}\|MA-X\|_F^2 + \frac{\beta}{2} \mathrm{Tr}(ALA^T) + i_{\Delta}(A)
$$
This has the following equivalent formulation:
$$
\begin{align*}
&\min_{U,V_1,V_2,V_2,V_4,V_5} \frac{1}{2}\|V_1-X\|_F^2 + \frac{\beta}{2} \mathrm{Tr}(V_2LV_2^T) + i_{\Delta}(V_2)\\


&\begin{align*}
 \text{subject to } V_1 &= MU\\
V_2 &= U\\
V_2 &= U\\
\end{align*}
\end{align*}
$$
which, in compact form, becomes
$$
\min_{U,V} g(V) \\\text{ subject to } GU + BV = 0
$$
where,
$$
g(V) = \frac{1}{2}\|V_1-X\|_F^2 + \frac{\beta}{2} \mathrm{Tr}(V_2LV_2^T) + i_{\Delta}(V_3)
$$

$$
V = \begin{bmatrix}
 V_1 &   &   \\ 
     &V_2&   \\ 
     &   &V_3\\ 
\end{bmatrix},
\quad
G = 
\begin{bmatrix}
M\\ 
I\\ 
I
\end{bmatrix},
\quad
B = 
\begin{bmatrix}
-I &  &  \\ 
   &-I&  \\ 
   &  &-I\\ 
\end{bmatrix}
$$

The augmented lagrangian $\mathcal{L}$ with parameter $\tau > 0$ of this problem is:
$$
\mathcal{L}(U,V,D) = g(V) + \frac{\tau}{2} \|GU + BV -D \|_F^2
$$
The corresponding ADMM updates are:
$$
\begin{align*}

U^{(k+1)} &= \argmin_U \frac{\tau}{2} \|GU + BV^{(k)} - D^{(k)} \|_F^2 \\
V^{(k+1)} &= \argmin_V g(V) + \frac{\tau}{2} \|GU^{(k+1)} + BV - D^{(k)} \|_F^2 \\
D^{(k+1)} &= D^{(k)} - GU^{(k+1)} - BV^{(k+1)}
\end{align*}

$$
By expanding V and D, we have
$$
\mathcal{L}(U,V_1,V_2,V_3,D_1,D_2,D_3) = \frac{1}{2}\|V_1-X\|_F^2 + \frac{\beta}{2} \mathrm{Tr}(V_2LV_2^T) + i_{\Delta}(V_3) + \frac{\tau}{2}\|MU-V_1-D_1\|_F^2 + \frac{\tau}{2}\|U-V_2-D_2\|_F^2 + \frac{\tau}{2}\|U-V_3-D_3\|_F^2
$$

The expanded but equivalent ADMM updates are:
$$
\begin{align*}
U^{(k+1)} &= \argmin_U \frac{\tau}{2}\|MU-V_1-D_1\|_F^2 + \frac{\tau}{2}\|U-V_2-D_2\|_F^2 + \frac{\tau}{2} \|U-V_3-D_3\|_F^2\\

V^{(k+1)}_1 &= \argmin_{V_1} \frac{1}{2}\|V_1-X\|_F^2 + \frac{\tau}{2} \|MU^{(k+1)} - V_1 - D^{(k)}_1 \|_F^2 \\
V^{(k+1)}_2 &= \argmin_{V_2} \frac{\beta}{2} \mathrm{Tr}(V_2LV_2^T) + \frac{\tau}{2} \|U^{(k+1)} - V_2 - D^{(k)}_2 \|_F^2 \\
V^{(k+1)}_3 &= \argmin_{V_3} i_{\Delta}(V_3) + \frac{\tau}{2} \|U^{(k+1)} - V_3 - D^{(k)}_3 \|_F^2 \\

D^{(k+1)}_1 &= D^{(k)}_1 - MU^{(k+1)} - V^{(k+1)}_1 \\
D^{(k+1)}_2 &= D^{(k)}_2 - U^{(k+1)} - V^{(k+1)}_2 \\
D^{(k+1)}_3 &= D^{(k)}_3 - U^{(k+1)} - V^{(k+1)}_3
\end{align*}

$$

<!-- #endregion -->

# Deriving Updates

**U Update**

As the problem for the U update is convex and differentiable, we derive the KKT condition and solve for U accordingly:

$$

\begin{align*}

0 &= \frac{\partial}{\partial U}\left[\frac{\tau}{2}\|MU-V_1-D_1\|_F^2 + \frac{\tau}{2}\|U-V_2-D_2\|_F^2 + \frac{\tau}{2} + \|U-V_3-D_3\|_F^2\right]
\\
0 &= \tau \left(M^T(MU-V_1-D_1) + (U-V_2-D_2) + (U-V_3-D_3)\right)
\\
M^TMU + 2U &= M^T(V_1+D_1) + (V_2+D_2) + (V_3+D_3)
\\
U &= (M^T M + 2I)^{-1}(M^T(V_1+D_1) + (V_2+D_2) + (V_3+D_3)) 
\end{align*}
$$
The update for U is:
$$
U^{(k+1)} = \left(M^T M + 2I\right)^{-1}\left(M^T(V_1^{(k)}+D_1^{(k)}) + (V_2^{(k)}+D_2^{(k)}) + (V_3^{(k)}+D_3^{(k)})\right)
$$

**V1 Update**

For the $V_1$ update, since it is convex and differentiable, we derive the KKT condition and solve for $V_1$ directly:
$$
\begin{align*}
0 &= \frac{\partial}{\partial V_1} \left[ \frac{1}{2}\|V_1-X\|_F^2 + \frac{\tau}{2} \|MU - V_1 - D_1 \|_F^2 \right] \\
0 &= (V_1 - X) + \tau(V_1 - (MU - D_1)) \\
V_1 &= \frac{1}{1+\tau} \left(X + (MU - D_1)\right)
\end{align*}
$$
For the $V_1$ update, we have:
$$
V_1^{(k+1)} = \frac{1}{1+\tau} \left(X + (MU^{(k)} - D_1^{(k)})\right)
$$

**V2 Update**

For the $V_2$ update, it is both convex and differentiable, so we derive the KKT condition and solve for $V_2$ directly. We also note that, we can estimate $L = S \Sigma S^T$
$$
\begin{align*}

0 &= \frac{\partial}{\partial V_2} \left[\frac{\beta}{2} \mathrm{Tr}(V_2LV_2^T) + \frac{\tau}{2} \|U - V_2 - D_2 \|_F^2   \right] \\
0 &= \frac{\beta}{2}\left(V_2 \left(S \Sigma S^T \right)^T +  V_3 \left(S \Sigma S^T \right) \right) + \tau\left( V_2 - (U - D_2)\right) \\
0 &= \beta V_2 S \Sigma S^T + \tau V_2 - \tau (U - D_2) \\
V_2\left(S \Sigma S^T + \frac{\tau}{\beta} I\right) &= \frac{\tau}{\beta} (U - D_2) \\
V_2 &= \frac{\tau}{\beta} (U - D_2)\left(S \Sigma S^T + \frac{\tau}{\beta} I\right)^{-1} \\
V_2 &= \frac{\tau}{\beta} (U - D_2)S\left(\Sigma + \frac{\tau}{\beta}I\right)^{-1}S^T
\end{align*}

$$
The update for $V_2$ is simple as $(\Sigma + \frac{\tau}{\beta}I)$ is a diagonal matrix, so the inversion is given by taking the reciprocal of the entries. 
$$
V_2^{(k+1)} = \frac{\tau}{\beta} (U^{(k+1)} - D_2^{(k)})S\left(\Sigma + \frac{\tau}{\beta}I\right)^{-1}S^T
$$

**V3 Update**

For the $V_3$ update, as $\Delta$ is an affine set, the update simply involes a least squares projection on $\Delta$:
$$
V_3^{(k+1)} = \textbf{proj}_{\Delta}(U^{(k+1)} - D_3^{(k)})
$$

**ADMM Updates**

The final ADMM updates are:
$$
\begin{align*}
U^{(k+1)} &= \left(M^T M + 2I\right)^{-1}\left(M^T(V_1^{(k)}+D_1^{(k)}) + (V_2^{(k)}+D_2^{(k)}) + (V_3^{(k)}+D_3^{(k)})\right)\\

V^{(k+1)}_1 &= \frac{1}{1+\tau} \left(X + (MU^{(k)} - D_1^{(k)})\right) \\
V^{(k+1)}_2 &= \frac{\tau}{\beta} (U^{(k+1)} - D_2^{(k)})S\left(\Sigma + \frac{\tau}{\beta}I\right)^{-1}S^T \\
V^{(k+1)}_3 &= \textbf{proj}_{\Delta}(U^{(k+1)} - D_3^{(k)}) \\

D^{(k+1)}_1 &= D^{(k)}_1 - MU^{(k+1)} - V^{(k+1)}_1 \\
D^{(k+1)}_2 &= D^{(k)}_2 - U^{(k+1)} - V^{(k+1)}_2 \\
D^{(k+1)}_3 &= D^{(k)}_3 - U^{(k+1)} - V^{(k+1)}_3
\end{align*}

$$




## Code

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
h5_import = h5py.File("data/bhsi_2023.h5",'r+').get('Cube/resultarray/inputdata')
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
slic_m_param = 2    #2
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
from SuperpixelCutsPy.normalized_cuts import *
from SuperpixelCutsPy.unmixing import graph_fclsu_admm
def create_cube(ASSIGNMENTS, ABUND):
    nx, ny = ASSIGNMENTS.shape
    ne, n_p = ABUND.shape
    cube = np.zeros((nx, ny, ne))

    for i in range(nx):
        for j in range(ny):
            for k in range(ne):
                cube[i, j, k] = ABUND[k, ASSIGNMENTS[i, j]]

    return cube

def graph_regularized_ncuts_admm(data,
                        superpixel_library,
                        superpixel_centers,
                        superpixel_assignments,
                        n_endmembers,
                        spectral_sigma2_param,
                        spatial_kappa_param,
                        spatial_dmax_param,
                        spatial_beta_param,
                        spectral_metric = 'SAM'):
    '''
    Description:
        Adaptive NCuts algorithm using Unmixing Information
        i) Do the initial clustering
        ii) Unmix using the extracted clusters as endmembers
        iii) Concatenate unmixing information to the end of the hyperspectral cube
        iv) Recreate the Superpixeled Cube with the new spectral information
        v) Do the spectral clustering on the new superpixeled cube
    ===========================================
    Parameters:
        data  
        superpixel_library
        superpixel_centers
        superpixel_assignments
        n_endmembers
        spectral_sigma2_param
        spectral_metric
        spatial_kappa_param
        spatial_dmax
        spatial_beta_param
    '''
    nx, ny, nb = data.shape
    hyperspectral_image = cube_to_matrix(data)
    distance_mtx = calc_spatial_distance_mtx(superpixel_centers, 1, 1)
    spatial_filter = (distance_mtx < spatial_kappa_param).astype(int) # gets cached
    history = []
    convergence_checks = []

    ## Initial Normalized Cuts Segmentation
    spectral_similarity_mtx = calc_spectral_similarity_mtx(superpixel_library, sigma2_param = spectral_sigma2_param, metric=spectral_metric)
    spatial_spectral_matrix = spatial_filter * spectral_similarity_mtx
    superpixel_cluster_labels = sklcluster.spectral_clustering(spatial_spectral_matrix, n_clusters=n_endmembers)
    history.append(superpixel_cluster_labels)

    ## Extract Mean Cluster Spectral Signatures
    mean_cluster_spectra = calc_mean_label_signatures(superpixel_library, superpixel_cluster_labels)
    print(f'Initial Clustering')

    ## Unmix Original HSI using Mean Cluster Spectral Signatures
    abund_mtx, history = graph_fclsu_admm(mean_cluster_spectra,
                                          superpixel_library,
                                            d_mtx = distance_mtx,
                                            d_max = spatial_dmax_param,
                                            beta = spatial_beta_param,
                                            mu = 1, #0.01
                                            n_iters=200,  ## In practice, ASC constraint is held well when n_iters is high
                                            eps_tol=0.01)
    
    abund_cube = create_cube(ASSIGNMENTS= superpixel_assignments, ABUND= abund_mtx)

    # Concatenate Unmixing Results to Original HSI, then Superpixel using Old Assignments
    abundance_plus_hyperspectral_cube = np.concatenate([data, abund_cube], axis = 2)
    _, abundance_plus_superpixel_library = generate_SLIC_superpixels(abundance_plus_hyperspectral_cube, superpixel_assignments)
    
    # Cluster Signatures + Abundances using Normalized Cuts Segmentation
    print(f'Spectral + Unmixing Clustering')
    spectral_similarity_mtx = calc_spectral_similarity_mtx(abundance_plus_superpixel_library, sigma2_param = spectral_sigma2_param, metric='SAM')
    spatial_spectral_matrix = spatial_filter * spectral_similarity_mtx
    superpixel_cluster_labels = sklcluster.spectral_clustering(spatial_spectral_matrix, n_clusters=n_endmembers, random_state = 5)

    ## Extract New Mean Cluster Spectral Signatures
    mean_cluster_spectra = calc_mean_label_signatures(superpixel_library, superpixel_cluster_labels)

    return superpixel_cluster_labels, mean_cluster_spectra
```

```python
sigma_param = 0.005 # 0.1 -> 0.001           #0.01
spatial_limit = 35# 15 -> 25 in steps of 5 #15

ne = 5#number of endmembers

superpixel_cluster_labels, mean_cluster_spectra = graph_regularized_ncuts_admm(data=hyperspectral_cube,
                                                                                superpixel_library=superpixel_library,
                                                                                superpixel_centers=centers,
                                                                                superpixel_assignments=assignments,
                                                                                n_endmembers=ne,
                                                                                spectral_sigma2_param=sigma_param,
                                                                                spatial_kappa_param=spatial_limit,
                                                                                spatial_beta_param= 0.1,
                                                                                spatial_dmax_param = 10,
                                                                                spectral_metric='EUCLIDEAN')

labelled_img = normalized_cuts.assign_labels_onto_image(assignments, superpixel_cluster_labels)

_, superpixel_original_library = superpixel.generate_SLIC_superpixels(data = original_hyperspectral_cube,
                                                                      assignments = assignments)

#original_library = segmentation_evaluation.calc_mean_label_signatures(superpixel_original_library, superpixel_cluster_labels)
```

```python
plt.imshow(labelled_img);
```

```python
sigma_param = 0.005 # 0.1 -> 0.001           #0.01
spatial_limit = 35# 15 -> 25 in steps of 5 #15

ne = 5#number of endmembers

superpixel_cluster_labels, mean_cluster_spectra = graph_regularized_ncuts_admm(data=hyperspectral_cube,
                                                                                superpixel_library=superpixel_library,
                                                                                superpixel_centers=centers,
                                                                                superpixel_assignments=assignments,
                                                                                n_endmembers=ne,
                                                                                spectral_sigma2_param=sigma_param,
                                                                                spatial_kappa_param=spatial_limit,
                                                                                spatial_beta_param= 0.1,
                                                                                spatial_dmax_param = 10,
                                                                                spectral_metric='SAM')

labelled_img = normalized_cuts.assign_labels_onto_image(assignments, superpixel_cluster_labels)

_, superpixel_original_library = superpixel.generate_SLIC_superpixels(data = original_hyperspectral_cube,
                                                                      assignments = assignments)

#original_library = segmentation_evaluation.calc_mean_label_signatures(superpixel_original_library, superpixel_cluster_labels)
```

```python
plt.imshow(labelled_img);
```

TO DO:
complete updates for both A and M.
