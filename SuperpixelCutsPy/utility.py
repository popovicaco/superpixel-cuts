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
