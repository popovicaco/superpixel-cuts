import numpy as np
import scipy as sp

class Preprocesser:
    '''
    Description:
        General Preproccesing Pipeline for 
        Hyperspectral Images
    ===========================================
    Parameters:
        data
    '''
    def __init__(self, data : np.ndarray):
        self.nx, self.ny, self.nb = data.shape
        self.data = data.copy()
        self.original_data = data.copy()

    def layer_normalization(self):
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

        for i in range(self.nb):
            min_i = np.min(self.data[:,:,i])
            max_i = np.max(self.data[:,:,i])
            self.data[:,:,i] = (self.data[:,:,i] - min_i)/(max_i - min_i)

    def gaussian_blur(self, blur_param : float = 1):
        '''
        Description:
            Adds Gaussian Blur to Each Layer within 
            the hyperspectral cube
        ===========================================
        Parameters:
            data - (nx, ny, nz) NumPy Matrix
        ===========================================
        Returns:
            contaminated_data
        '''
        for k in range(self.nb):
            self.data[:,:,k] = sp.ndimage.gaussian_filter(self.data[:,:,k], sigma = blur_param)
    
    def singular_value_decomposition(self, n_svd : int = 5):
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
        data_flat = self.data.reshape((self.nx*self.ny,self.nb)).T
        u, s, v = np.linalg.svd(data_flat, full_matrices=False)
        u_truncated = u[:, :n_svd]
        s_truncated = np.diag(s[:n_svd])
        v_truncated = v[:n_svd, :]
        data_flat_denoised = u_truncated @ s_truncated @ v_truncated
        self.data = data_flat_denoised.T.reshape((self.nx,self.ny,self.nb))

    def principal_component_analysis(self, n_pc : int = 5, verbose = True):
        '''
        Description:
            Performs PCA dimension reduction for a 3D NumPy Matrix
        ===========================================
        Parameters:
            data - (nx, ny, nz) NumPy Matrix
        ===========================================
        Returns:
            denoised_data - (nx, ny, nz) NumPy Matrix
        '''   
        data_flat = self.data.reshape((self.nx*self.ny,self.nb)).T

        # Standardize the Data
        data_mean = data_flat.mean(axis = 0)
        data_std = data_flat.std(axis = 0)
        data_flat = (data_flat - data_mean)/data_std

        #computes the covariance matrix and does eigendecomposition
        covariance_matrix = np.cov(data_flat)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sorting the eigenvalues in decreasing order
        indices = np.arange(0,len(eigenvalues),1)
        indices = [x for _,x in sorted(zip(eigenvalues,indices))]
        indices = indices[::-1]
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:,indices]

        #extracting the top (num_principal_components) vectors and their eigenvalues
        eigenvalues = eigenvalues[:n_pc]
        eigenvectors = eigenvectors[:,:n_pc]

        # projecting the bands into the principal components of shape (nx*ny, n_bands)
        data_flat_reduced = np.dot(eigenvectors.T, data_flat)

        #reshaping the reduced bands into a cube of shape (nx, ny, num_principal_components)
        self.data = data_flat_reduced.T.reshape((self.nx, self.ny, n_pc)).copy().real
        self.nb = n_pc

        if verbose:
            print(f'The first {n_pc} principal components explain {np.round(100*explained_variance_ratio[0:n_pc].sum(), 3)} of the variance')