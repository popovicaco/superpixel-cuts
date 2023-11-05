import numpy as np
import scipy as sp

def generate_SLIC_assignments(data, n_superpixels, slic_m_param, gradient_search_param = 3, max_iters = 10, verbose = True):
    '''
    Description:
        Simple Linear Iterative Clustering Algorithm for Hyperspectral Images.
        i) 
            Initialize Centroids uniformly and move them to lowest gradient position
            in a gradient_search_param x gradient_search_param neighborhood. 
        ii)
            For each cluster, calculate the similarity measure with respect to the the
            2S X 2S neighborhood where S = sqrt(n_pixels / n_superpixels). Assign pixels
            a label corresponding to the cluster they have the best similarity score to.
        iii)
            Update clusters by calculating mean centroid (x,y) and mean spectral signature.
        iv) 
            Repeat steps (i) and (ii) up to max_iters times.
    ===========================================
    Parameters:
        data                    - (nx, ny, nz) NumPy Matrix
        n_superpixels           - Number of Superpixels
        slic_m_param            - SLIC parameter m
        gradient_search_param   - Gradient Initialization Search param
        max_iters               - Maximum number of iterations
        verbose                 - Return Console Output
    ===========================================
    Returns:
        assignments             - (nx, ny) NumPy Matrix with Superpixel Assignments
        centers                 - (n_superpixel, 2) NumPy Matrix with Superpixel Centroids
    ===========================================
    References:
        [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk,
            SLIC Superpixels Compared to State-of-the-art Superpixel Methods. IEEE Transactions on Pattern 
            Analysis and Machine Intelligence, Volume 34, Issue 11, pp. 2274-2282, May 2012
    '''
    nx, ny, nb = data.shape
    step_size = int(np.sqrt(nx*ny/n_superpixels)) #S is our stepsize 

    #===================# STEP 1 #===================#
    # Calculate the number of points in each dimension
    num_points_x = int(np.ceil(nx / step_size))
    num_points_y = int(np.ceil(ny / step_size))
    # Create the coordinate arrays
    centers_x = np.arange(step_size//2, nx-1, step_size)
    centers_y = np.arange(step_size//2, ny-1, step_size)
    # Adjust the number of points to match the coordinate arrays
    n_points_x = min(num_points_x, len(centers_x))
    n_points_y = min(num_points_y, len(centers_y))
    # Create the (x,y) centroid pairs for the centroids
    centers_x = centers_x[:n_points_x]
    centers_y = centers_y[:n_points_y]
    # Adjust n_superpixels if need be
    if n_points_x*n_points_y != n_superpixels:
        if verbose:
            print(f'Adjusting n_superpixels: {n_points_x*n_points_y}')
        n_superpixels = n_points_x*n_points_y
    # Calculate gradient map G(x,y) = ||(x+1,y) - (x-1,y)||**2 +  ||(x,y+1) - (x,y-1)||**2
    x_kernel = np.array([[-1,0,1]]) ## This kernel is applied across axis = 0
    y_kernel = x_kernel.T           ## This kernel is applied across axis = 1
    x_diffs = np.stack([sp.ndimage.convolve(data[:,:,band], x_kernel) for band in range(nb)], axis = -1)
    y_diffs = np.stack([sp.ndimage.convolve(data[:,:,band], y_kernel) for band in range(nb)], axis = -1)
    image_gradient = (x_diffs**2).sum(axis=2) + (y_diffs**2).sum(axis=2)
    # Move initial centroids to lowest gradient position in gradient_search_param neighborhood
    centers = np.zeros((n_superpixels, 2), dtype=np.int16)
    search_bound = np.floor(gradient_search_param/2).astype(int)
    for i, ctr_x in enumerate(centers_x):
        for j, ctr_y in enumerate(centers_y):
            neighborhood_gradient = image_gradient[max(0, ctr_x - search_bound):min(nx-1, ctr_x + search_bound + 1), max(0, ctr_y - search_bound):min(ny-1, ctr_y + search_bound + 1)]
            min_neighborhood_idx = np.argmin(neighborhood_gradient)
            centers[i * len(centers_y) + j] = [ctr_x + min_neighborhood_idx//gradient_search_param -1, ctr_y + min_neighborhood_idx%gradient_search_param - 1]
    
    #===================# STEP 2 #===================#
    # Initialize assignments and distances
    assignments = np.zeros((nx, ny), dtype=np.int32) - 1
    distances = np.full((nx, ny), np.inf)
    # SLIC Iterations
    for _ in range(max_iters):
        # Update cluster assignments and distances
        for i, center in enumerate(centers):
            x_min = max(center[0] - 2*step_size, 0)
            x_max = min(center[0] + 2*step_size, nx)
            y_min = max(center[1] - 2*step_size, 0)
            y_max = min(center[1] + 2*step_size, ny)
            # Calculate spectral similarity measure
            spectral_diff = data[x_min:x_max, y_min:y_max, :] - data[center[0], center[1], :]
            spectral_distances = np.sqrt((spectral_diff**2).sum(axis=2))
            # Calculate spatial distances
            x_diff = np.indices((x_max - x_min, y_max - y_min))[0] + x_min - center[0]
            y_diff = np.indices((x_max - x_min, y_max - y_min))[1] + y_min - center[1]
            euclidean_distances = np.sqrt(x_diff ** 2 + y_diff**2)
            # Calculate SLIC distances
            slic_distances = np.sqrt(spectral_distances**2 + (slic_m_param**2)*(euclidean_distances/step_size)**2)
            # Assign labels
            mask = (slic_distances < distances[x_min:x_max, y_min:y_max])
            distances[x_min:x_max, y_min:y_max][mask] = slic_distances[mask]
            assignments[x_min:x_max, y_min:y_max][mask] = i
        
        #===================# STEP 3 #===================#
        # Update clusters by calculating mean centroid (x,y) and mean spectral signature.
        for i, center in enumerate(centers):
            mask = (assignments == i)
            coordinates = np.nonzero(mask)
            new_center = np.mean(np.column_stack(coordinates), axis=0).astype(int)
            centers[i] = new_center

    if verbose:
        print(f'Created {len(np.unique(assignments))} superpixels')
    return assignments, centers

def generate_SLIC_superpixels(data, assignments):
    '''
    Description:
        Generate Superpixels given label assignments from SLIC
    ===========================================
    Parameters:
        data                    - (nx, ny, nz) NumPy Matrix
        assignments             - (nx, ny) NumPy Matrix with Superpixel Assignments
    ===========================================
    Returns:
        superpixel_data         - (nx, ny, nz) NumPy Matrix
        superpixel_spectra      - (nz, n_superpixel) NumPy Matrix
    ===========================================
    References:
        [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk,
            SLIC Superpixels Compared to State-of-the-art Superpixel Methods. IEEE Transactions on Pattern 
            Analysis and Machine Intelligence, Volume 34, Issue 11, pp. 2274-2282, May 2012
    '''
    nx, ny, nb = data.shape
    superpixel_data = np.zeros_like(data)
    superpixel_spectra = []
    for label in list(np.unique(assignments)):
        idxs = np.argwhere((assignments==label))
        avg_label_spectra = data[idxs[:,0], idxs[:,1], :].mean(axis=0)
        superpixel_spectra.append(avg_label_spectra)
        superpixel_data[idxs[:,0], idxs[:,1]] = avg_label_spectra
    superpixel_spectra = np.stack(superpixel_spectra).T
    return superpixel_data, superpixel_spectra
