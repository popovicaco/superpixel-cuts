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
layer_normalized_spectra = np.load("data/layer_normalized_spectra_bhsi.npy")
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
def calculate_norm(A : np.ndarray,
                    p : int = 2,
                    q : int = 2) -> float:
    '''
        Calculates the L_{p,q} mixed matrix norm
    '''
    n_rows, n_cols = A.shape
    # sum by row, then column
    return ((np.abs(A)**p).sum(axis = 1)**(q/p)).sum()**(1/q)

def convex_projection(X : np.ndarray) -> np.ndarray:
    '''
        Projects the columns of a matrix X - (m,n) onto a convex set
        projection = 'probability_simplex' will project the columns
        of the matrix onto the set △(N) = { a ∈ R^N_+ | 1^T a = 1 }
        sourced from: https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    '''
    p, n = X.shape
    u = np.sort(X, axis=0)[::-1, ...]
    pi = np.cumsum(u, axis=0) - 1
    ind = (np.arange(p) + 1).reshape(-1, 1)
    mask = (u - pi / ind) > 0
    rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
    theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
    return np.maximum(X - theta, 0)

def primal_residual_norm(M : np.ndarray,
                         U : np.ndarray,
                         V1 : np.ndarray,
                         V2 : np.ndarray,
                         V3 : np.ndarray ) -> float:
    '''
        Calculates the Primal Residual Norm
                \|GU + BV\|_F 
    '''
    return np.sqrt(calculate_norm(M@U - V1)**2 + calculate_norm(U-V2)**2 + calculate_norm(U-V3)**2)

def dual_residual_norm(mu_param, M, D1, D2, D3, D1_prev, D2_prev, D3_prev):
    '''
        Calculates the Dual Residual Frobenius Norm
                \|mu*(G.T@V)@(D - D_prev)\|_F 
    '''
    return mu_param*np.sqrt(calculate_norm(M.T@(D1 - D1_prev))**2 + calculate_norm(D2 - D2_prev)**2 + calculate_norm(D3 - D3_prev)**2)  

def create_cube(ASSIGNMENTS, ABUND):
    nx, ny = ASSIGNMENTS.shape
    ne, n_p = ABUND.shape
    cube = np.zeros((nx, ny, ne))

    for i in range(nx):
        for j in range(ny):
            for k in range(ne):
                cube[i, j, k] = ABUND[k, ASSIGNMENTS[i, j]]

    return cube
```

```python
from scipy.sparse.csgraph import laplacian
def graph_fclsu_admm_2(M       : np.ndarray,
                     X       : np.ndarray,
                     centers : np.ndarray,
                     d_max   : float,
                     beta    : float,
                     mu      : float = 100,
                     n_iters : int = 200,
                     eps_tol : float = 0.001):
    '''
        ADMM Based Impementation of Graph Regularized Fully Constrained Linear Unmixing
        Parameters:
            M       - Known Endmember Spectra Matrix
            X       - Hyperspectral Image Matrix
            centers - Superpixel Centers
            D_max   - Maximum Spatial Distance to be considered "similar"
            beta    - Graph Regularization parameter
            mu      - ADMM convergence parameter
            eps_tol - convergence criteria
    '''
    history = {
        'loss':[],
        'primal_residual':[],
        'dual_residual':[],
        'n_iters':0
    }

    _, n_endmember = M.shape
    n_p = X.shape[1]

    ### Initializations ###
    U  = np.random.random((n_endmember, n_p))
    V1 = M@(U.copy()) #np.random.random((n_band, n_p)) ## V1 = MU 
    V2 = U.copy()
    V3 = U.copy()

    D1 = np.zeros_like(V1) # M@(U.copy()) ## V1 = MU
    D2 = np.zeros_like(V2) #U.copy()
    D3 = np.zeros_like(V3) #U.copy()
    
    ### Residuals for Tracking
    D1_prev = None
    D2_prev = None
    D2_prev = None

    distance_mtx = normalized_cuts.calc_spatial_distance_mtx(centers= centers, nx = 1, ny = 1)
    W = laplacian((distance_mtx <= d_max).astype(int))
    S, E, _ = np.linalg.svd(W) # w = S E S.T

    partial_V2_update = np.linalg.inv(np.diag(E) + (mu/beta)*np.eye(n_p))
    partial_U_update = np.linalg.inv(M.T @ M + 2*np.eye(n_endmember))

    for k in range(n_iters):
        if k%50 == 0:
            print(k)
        ## U Update
        U = partial_U_update@(M.T@(V1 + D1) + (V2 + D2) + (V3 + D3))
        ## V1 Update
        V1 = (1/(1+mu))*(X + mu*(M@U - D1))
        ## V2 Update (Soft Thresholding Operator)
        #V2 = soft_threshold(U - D2, lambda_param/mu_param)
        V2 = (mu/beta)*(U - D2)@S@partial_V2_update@S.T
        ## V3 Update - Projection onto Delta
        V3 = convex_projection(U - D3)
        ## D1,D2,D3 Update 
        D1_prev = D1.copy()
        D2_prev = D2.copy()
        D3_prev = D3.copy()

        D1 = D1 - M@U + V1
        D2 = D2 - U + V2
        D3 = D3 - U + V3

        loss = calculate_norm(M@U - X, p=2, q=2)**2 + np.trace(U@W@U.T)
        primal_res = primal_residual_norm(M,U,V1,V2,V3)
        dual_res = dual_residual_norm(mu, M, D1, D2, D3, D1_prev, D2_prev, D3_prev)

        ## REPORTING ##
        history['loss'].append(loss) 
        history['primal_residual'].append(primal_res)
        history['dual_residual'].append(dual_res)
        history['n_iters'] += 1

    return U, history

```

```python
abund_mtx, history = graph_fclsu_admm_2(layer_normalized_spectra,
                                     superpixel_library,
                                     centers = centers,
                                     d_max = 10,
                                     beta = 0.05,
                                     mu = 1, #0.01
                                     n_iters=75,  ## In practice, ASC constraint is held well when n_iters is high
                                     eps_tol=0.01)

abund_cube = create_cube(ASSIGNMENTS = assignments, ABUND= abund_mtx)
```

```python
abund_cube.sum(axis = 2).mean()
```

```python
num_layers = min(abund_cube.shape[2], 5)

fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))

for i in range(num_layers):
    axes[i].imshow(abund_cube[:, :, i], cmap='viridis')
    axes[i].set_title(f'Layer {i+1}')
```

```python

```

TO DO:
complete updates for both A and M.
