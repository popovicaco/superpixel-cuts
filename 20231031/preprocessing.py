import numpy as np

def cube_to_matrix(data):
    '''
    Description:
        Reshapes a 3D NumPy Matrix to a 2D NumPy Matrix
    ===========================================
    Parameters:
        data - (nx, ny, nz) NumPy Matrix
    ===========================================
    Returns:
        data - (nx * ny, nz) NumPy Matrix
    '''
    return data.reshape((data.shape[0]*data.shape[1],data.shape[2])).T

def matrix_to_cube(X, nx, ny, nb):
    '''
    Description:
        Reshapes a 2D NumPy Matrix to a 3D NumPy Matrix
    ===========================================
    Parameters:
        data - (nx * ny, nz) NumPy Matrix
    ===========================================
    Returns:
        data - (nx, ny, nz) NumPy Matrix
    '''
    return X.T.reshape((nx,ny,nb))

def layer_normalize(data):
    '''
    Description:
        Performs Layer Normalization on a 3D NumPy 
        Matrix across it's last dimension.
        For each layer  i of the matrix X:
           X_i = X - min(X_i) / max(X_i) - min(X_i)

    ===========================================
    Parameters:
        data - (nx, ny, nz) NumPy Matrix
    ===========================================
    Returns:
        normalized_data - (nx, ny, nz) NumPy Matrix
    '''
    nb = data.shape[2]
    normalized_data = np.zeros(data.shape)
    for i in range(nb):
        min_i = np.min(data[:,:,i])
        max_i = np.max(data[:,:,i])
        normalized_data[:,:,i] = (data[:,:,i] - min_i)/(max_i - min_i)
    return normalized_data

def svd_denoise(data, n_svd = 4, verbose = True):    
    '''
    Description:
        Performs SVD Denoising for a 3D NumPy Matrix
    ===========================================
    Parameters:
        data - (nx, ny, nz) NumPy Matrix
    ===========================================
    Returns:
        denoised_data - (nx, ny, nz) NumPy Matrix
    '''
    if n_svd == 0:
        return data
    nx,ny,nb = data.shape
    u,s,v = np.linalg.svd(cube_to_matrix(data), full_matrices=False)
    u_truncated = u[:, :n_svd]
    s_truncated = np.diag(s[:n_svd])
    v_truncated = v[:n_svd, :]
    denoised_mtx = u_truncated @ s_truncated @ v_truncated
    denoised_data = matrix_to_cube(denoised_mtx, nx, ny, nb)
    if verbose:
        loss = np.sqrt(((data - denoised_data)**2).sum())
        print(f'The reconstruction error is {loss}')
    return denoised_data