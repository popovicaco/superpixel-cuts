import numpy as np
import sklearn.cluster as sklcluster
from .unmixing import *
from .utility import *
from .superpixel import *
from .segmentation_evaluation import *

def example_function():
    '''
    Description:
        Put Description Here
    ===========================================
    Parameters:
        data  
        param1
    ===========================================
    Returns:
        return_data 
    ===========================================
    References:
        [1] Some Random Paper
    '''
    return 0


def find_adjacent_labels(label, assignments):
    '''
    Description:
        Gets Adjacent Labels for a Given Label
    ===========================================
    Parameters:
        label  
        assignments
    ===========================================
    Returns:
        neighbor_labels
    '''
    nx, ny = assignments.shape
    ## Get Indices of Pixels with Label
    labelled_pixels = np.where((assignments == label))

    ## Get Indices Above, Below, Left and Right
    pixels_above = (np.maximum(labelled_pixels[0] - 1, 0), labelled_pixels[1])
    pixels_above_left = (np.maximum(labelled_pixels[0] - 1, 0), np.maximum(labelled_pixels[1] - 1, 0))
    pixels_above_right = (np.maximum(labelled_pixels[0] - 1, 0), np.minimum(labelled_pixels[1] + 1, ny-1))
    pixels_below = (np.minimum(labelled_pixels[0] + 1, nx - 1), labelled_pixels[1])
    pixels_below_left = (np.minimum(labelled_pixels[0] + 1, nx - 1), np.maximum(labelled_pixels[1] - 1, 0))
    pixels_below_right = (np.minimum(labelled_pixels[0] + 1, nx - 1), np.minimum(labelled_pixels[1] + 1, ny-1))
    pixels_left = (labelled_pixels[0], np.maximum(labelled_pixels[1] - 1, 0))
    pixels_right = (labelled_pixels[0], np.minimum(labelled_pixels[1] + 1, ny-1))

    ## Get Labels from those Indices
    unique_labels_above = np.unique(assignments[pixels_above[0],pixels_above[1]])
    unique_labels_above_left = np.unique(assignments[pixels_above_left[0],pixels_above_left[1]])
    unique_labels_above_right = np.unique(assignments[pixels_above_right[0],pixels_above_right[1]])
    unique_labels_below = np.unique(assignments[pixels_below[0],pixels_below[1]])
    unique_labels_below_left = np.unique(assignments[pixels_below_left[0],pixels_below_left[1]])
    unique_labels_below_right = np.unique(assignments[pixels_below_right[0],pixels_below_right[1]])
    unique_labels_left = np.unique(assignments[pixels_left[0],pixels_left[1]])
    unique_labels_right = np.unique(assignments[pixels_right[0],pixels_right[1]])

    neighbor_labels = np.concatenate([unique_labels_above, unique_labels_above_left, unique_labels_above_right,
                                unique_labels_below, unique_labels_below_left, unique_labels_below_right,
                                unique_labels_left, unique_labels_right])
    neighbor_labels = list(set(neighbor_labels))
    if label in neighbor_labels:
        neighbor_labels.remove(label)
    return neighbor_labels

def calc_spatial_adjacency_mtx(assignments):
    '''
    Description:
        Create Spatial Adjacency Matrix (0/1 Entries)
    ===========================================
    Parameters:
        assignments      - (nx, ny) NumPy Matrix  
    ===========================================
    Returns:
        adjacency_matrix - (n_labels, n_labels) NumPy Adjacency Matrix
    '''
    unique_labels = list(np.unique(assignments))
    n_labels = len(unique_labels)
    adjacency_matrix = np.zeros((n_labels, n_labels))
    for label in unique_labels:
        adjacent_labels = find_adjacent_labels(label, assignments)
        adjacency_matrix[label,adjacent_labels + [label]] = 1
    return adjacency_matrix
#
def calc_spatial_distance_mtx(centers, nx, ny):
    '''
    Description:
        Create Spatial Euclidean Distance Matrix
    ===========================================
    Parameters:
        centers      - (n_superpixels, 2) NumPy Matrix  
    ===========================================
    Returns:
        distance_matrix - (n_labels, n_labels) Euclidean Distance Matrix
    '''
    dx = (centers[:,0, np.newaxis] - centers[:,0, np.newaxis].T)/nx
    dy = (centers[:,1, np.newaxis] - centers[:,1, np.newaxis].T)/ny
    distance_matrix = np.sqrt(dx**2 + dy**2)
    return distance_matrix

def calc_spectral_similarity_mtx(spectral_library, sigma2_param, metric = 'SAM'):
    '''
    Description:
        Calculates the Heat Kernel for NCuts
            np.exp((-(similarity_mtx**2)/sigma2_param))
    ===========================================
    Parameters:
        spectral_library  - (nb, n_superpixel) NumPy Matrix 
        sigma2_param      - Spectral Weighting Parameters
        metric            - Metric Used ('SAM', 'EUCLIDEAN', 'SID')
    ===========================================
    Returns:
        heat_kernel       - (n_superpixel, n_superpixel) NumPy Heat Kernel Matrix 
    '''
    '''Calculates the Heat Kernel/Similarity Matrix for NCuts'''
    nb, n_pixels = spectral_library.shape
    similarity_mtx = np.zeros((n_pixels, n_pixels))

    if metric == 'SAM':
        norms = np.linalg.norm(spectral_library, axis = 0)
        normed_spectral_library = spectral_library / norms
        similarity_mtx = np.arccos(np.clip(np.dot(normed_spectral_library.T, normed_spectral_library), 0, 1))

    elif metric == 'EUCLIDEAN':
        similarity_mtx = ((spectral_library.T[:, np.newaxis, :] - spectral_library.T[np.newaxis, :, :])**2).sum(axis=2)
    
    elif metric == 'SID':
        nb, n_superpixels = spectral_library.shape
        normed_spectral_library = spectral_library/spectral_library.sum(axis = 0) + np.spacing(1)
        for i in range(n_superpixels):
            for j in range(n_superpixels):
                sid_similarity_vec = normed_spectral_library[:,i] * np.log(normed_spectral_library[:,i] / normed_spectral_library[:,j]) + normed_spectral_library[:,j] * np.log(normed_spectral_library[:,j] / normed_spectral_library[:,i])
                similarity_mtx[i,j] = sid_similarity_vec.sum()
    
    heat_kernel = np.exp((-(similarity_mtx**2)/sigma2_param))
    return heat_kernel
    
def assign_labels_onto_image(assignments, labels):
    '''
    Description:
        Assign Labels onto an Image
        assign_labels_onto_image
    ===========================================
    Parameters:
        assignments 
        labels
    ===========================================
    Returns:
        labelled_img
    '''
    labelled_img = assignments.copy()
    for label in list(np.unique(labels)):
        labelled_img[np.isin(assignments, np.where(labels == label)[0])] = label
    return labelled_img

def calc_mean_label_signatures(spectral_library, labels):
    '''
    Description:
        Extract Mean Label Spectra Signature
    ===========================================
    Parameters:
        spectral_library  
        labels
    ===========================================
    Returns:
        mean_endmember_spectra
    '''
    nb, _ = spectral_library.shape
    unique_labels = np.unique(labels)
    ne = len(unique_labels)
    mean_endmember_spectra = np.zeros((nb,ne))
    for i in range(ne):
        mean_endmember_spectra[:,i] = spectral_library[:,np.where(labels == unique_labels[i])[0]].mean(axis=1) 
    return mean_endmember_spectra

def calc_mean_label_signatures_v2(spectral_library  : np.ndarray,
                                  labels            : np.ndarray,
                                  ignore_label      : int = -1):
    """
    Description:
        Extract Mean Label Spectra Signature
    ===========================================
    Args:
        spectral_library (np.ndarray): _description_
        labels (np.ndarray): _description_
    ===========================================
    Returns:
        np.ndarray: _description_
    """
    nb, _ = spectral_library.shape
    unique_labels = np.unique(labels)
    if ignore_label in unique_labels:
        unique_labels = unique_labels[1::].copy()
    ne = len(unique_labels)
    mean_endmember_spectra = np.zeros((nb,ne))
    for i in range(ne):
        mean_endmember_spectra[:,i] = spectral_library[:,np.where(labels == unique_labels[i])[0]].mean(axis=1) 
    return mean_endmember_spectra

def calc_std_label_signatures(spectral_library, labels):
    '''
    Description:
        Extract Standard Deviation of Label Spectra Signature across Spectral Axis
    ===========================================
    Parameters:
        spectral_library  
        labels
    ===========================================
    Returns:
        std_endmember_spectra
    '''
    nb, _ = spectral_library.shape
    unique_labels = np.unique(labels)
    ne = len(unique_labels)
    std_endmember_spectra = np.zeros((nb,ne))
    for i in range(ne):
        std_endmember_spectra[:,i] = spectral_library[:,np.where(labels == unique_labels[i])[0]].std(axis=-1)
    return std_endmember_spectra

def adaptive_ncuts(data, superpixel_library, superpixel_centers, superpixel_assignments, n_endmembers, spectral_sigma2_param, spatial_kappa_param, spectral_metric = 'SAM', n_iters = 10):
    '''
    Description:
        Adaptive NCuts algorithm using Unmixing Information
        i) Do the initial clustering
        ii) Unmix using the extracted clusters as endmembers
        iii) Concatenate unmixing information to the end of the hyperspectral cube
        iv) Recreate the Superpixeled Cube with the new spectral information
        v) Do the spectral clustering on the new superpixeled cube
        vi) Repeat (ii) - (vi) until n_iters
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
        n_iters
    ===========================================
    Returns:
        labels
        endmember_spectra
        history
    ===========================================
    References:
        [1] Jianbo Shi and J. Malik, "Normalized cuts and image segmentation," in IEEE 
            Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 8, 
            pp. 888-905, Aug. 2000, doi: 10.1109/34.868688.
    '''
    nx, ny, nb = data.shape
    hyperspectral_image = cube_to_matrix(data)
    spatial_filter = (calc_spatial_distance_mtx(superpixel_centers, 1, 1) < spatial_kappa_param).astype(int) # gets cached
    history = []
    convergence_checks = []

    ## Initial Normalized Cuts Segmentation
    spectral_similarity_mtx = calc_spectral_similarity_mtx(superpixel_library, sigma2_param = spectral_sigma2_param, metric=spectral_metric)
    spatial_spectral_matrix = spatial_filter * spectral_similarity_mtx
    superpixel_cluster_labels = sklcluster.spectral_clustering(spatial_spectral_matrix, n_clusters=n_endmembers)
    history.append(superpixel_cluster_labels)

    ## Extract Mean Cluster Spectral Signatures
    mean_cluster_spectra = calc_mean_label_signatures(superpixel_library, superpixel_cluster_labels)
    print(f'Initial Clustering | n_segments = {len(np.unique(history[0]))}')
    for i in range(n_iters):

        ## Unmix Original HSI using Mean Cluster Spectral Signatures
        abund_mtx = active_set_fcls(mean_cluster_spectra, hyperspectral_image)
        abund_cube = matrix_to_cube(abund_mtx, nx, ny, n_endmembers)

        # Concatenate Unmixing Results to Original HSI, then Superpixel using Old Assignments
        abundance_plus_hyperspectral_cube = np.concatenate([data, abund_cube], axis = 2)
        _, abundance_plus_superpixel_library = generate_SLIC_superpixels(abundance_plus_hyperspectral_cube, superpixel_assignments)

        # Cluster Signatures + Abundances using Normalized Cuts Segmentation
        spectral_similarity_mtx = calc_spectral_similarity_mtx(abundance_plus_superpixel_library, sigma2_param = spectral_sigma2_param, metric='SAM')
        spatial_spectral_matrix = spatial_filter * spectral_similarity_mtx
        superpixel_cluster_labels = sklcluster.spectral_clustering(spatial_spectral_matrix, n_clusters=n_endmembers, random_state = 5)

        history.append(superpixel_cluster_labels)

        ## Extract New Mean Cluster Spectral Signatures
        mean_cluster_spectra = calc_mean_label_signatures(superpixel_library, superpixel_cluster_labels)
        print(f'Feedback Iteration {i+1} | n_segments = {len(np.unique(history[i+1]))}')
        confusion_mtx = compare_segmentations(history[i], history[i+1]) 
        segmentation_similarity_accuracy = confusion_mtx.trace()/confusion_mtx.sum()
        convergence_checks.append(segmentation_similarity_accuracy)

        if segmentation_similarity_accuracy >= 1.0:
            print(f'Convergence, Terminating at Iteration {i+1}...')
            break
    
    return superpixel_cluster_labels, mean_cluster_spectra, history, convergence_checks

def single_ncuts(data,
                 superpixel_library,
                 superpixel_centers,
                 superpixel_assignments,
                 n_endmembers,
                 spectral_sigma2_param,
                 spatial_kappa_param,
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
    ===========================================
    Returns:
        labels
        endmember_spectra
    ===========================================
    References:
        [1] Jianbo Shi and J. Malik, "Normalized cuts and image segmentation," in IEEE 
            Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 8, 
            pp. 888-905, Aug. 2000, doi: 10.1109/34.868688.
    '''
    nx, ny, nb = data.shape
    hyperspectral_image = cube_to_matrix(data)
    spatial_filter = (calc_spatial_distance_mtx(superpixel_centers, 1, 1) < spatial_kappa_param).astype(int) # gets cached
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
    abund_mtx = active_set_fcls(mean_cluster_spectra, hyperspectral_image)
    abund_cube = matrix_to_cube(abund_mtx, nx, ny, n_endmembers)

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

def basic_hole_filling(superpixel_cluster_labels, assignments, superpixel_library, mean_cluster_spectra, verbose=True):
    labels_new = superpixel_cluster_labels.copy()
    adjacency_mtx = calc_spatial_adjacency_mtx(assignments)
    for i in range(adjacency_mtx.shape[0]):
        adjacency_labels = labels_new[np.where(adjacency_mtx[i,:] == 1)]
        original_label = labels_new[i]
        original_label_count = (adjacency_labels == original_label).sum()
        labels_unique, labels_counts = np.unique(adjacency_labels, return_counts=True)
        n_unique_labels = len(labels_unique)
        if n_unique_labels == 3: #IF THERE ARE THREE UNIQUE LABELS
            if labels_unique[np.argmax(labels_counts)] != original_label: # IF THE CURRENT LABEL IS NOT THE MAJORITY ONE
                superpixel = superpixel_library[:,i]
                normed_superpixel = superpixel / np.linalg.norm(superpixel, axis = 0)
                normed_endmember_spectra = mean_cluster_spectra / np.linalg.norm(mean_cluster_spectra, axis = 0)
                sam_scores = 1/np.arccos(np.clip(np.dot(normed_endmember_spectra.T, normed_superpixel), 0, 1))
                test = np.array([1 if num in labels_unique and num != original_label else 0 for num in list(np.unique(superpixel_cluster_labels)+1)])
                labels_new[i] = np.argmax(sam_scores*test) + 1
                if verbose:
                    print(f'superpixel {i} : curr_label {original_label} : new_label : {labels_new[i]} : {adjacency_labels} :  {labels_unique} : {labels_counts}')
    
    labelled_image_holefilled = assign_labels_onto_image(assignments, labels_new)
    return labelled_image_holefilled, labels_new


def superpixel_subsegment(data                   : np.ndarray,
                            superpixel_library     : np.ndarray,
                            superpixel_centers     : np.ndarray,
                            superpixel_assignments : np.ndarray,
                            n_endmembers           : int,
                            spectral_param         : float,
                            spatial_param          : float,
                            spectral_metric        : str,
                            ignore_label           : int = -1):
    """
     Description:
        Adaptive NCuts algorithm using Unmixing Information
        i) Do the initial clustering
        ii) Unmix using the extracted clusters as endmembers
        iii) Concatenate unmixing information to the end of the hyperspectral cube
        iv) Recreate the Superpixeled Cube with the new spectral information
        v) Do the spectral clustering on the new superpixeled cube

    Args:
        data (np.ndarray): _description_
        superpixel_library (np.ndarray): _description_
        superpixel_centers (np.ndarray): _description_
        superpixel_assignments (np.ndarray): _description_
        n_endmembers (int): _description_
        spectral_param (float): _description_
        spatial_param (float): _description_
        spectral_metric (str): _description_
    """
    nx, ny, _ = data.shape
    hyperspectral_image = cube_to_matrix(data)
    spatial_filter = (calc_spatial_distance_mtx(superpixel_centers, 1, 1 < spatial_param)).astype(int)
    
    spectral_similarity_mtx = calc_spectral_similarity_mtx(superpixel_library,
                                                           sigma2_param=spectral_param,
                                                           metric=spectral_metric)
    spatial_spectral_mtx = spatial_filter * spectral_similarity_mtx

    # Spectral Clustering
    cluster_labels = sklcluster.spectral_clustering(spatial_spectral_mtx,
                                                    n_clusters=n_endmembers,
                                                    random_state=20021225)
    
    # Extract Mean Signatures
    mean_cluster_spectra = calc_mean_label_signatures(superpixel_library, cluster_labels)

    ## Unmix Original HSI using Mean Cluster Spectral Signatures
    abund_mtx = active_set_fcls(mean_cluster_spectra, hyperspectral_image)
    abund_cube = matrix_to_cube(abund_mtx, nx, ny, n_endmembers)

    abundance_plus_hyperspectral_cube = np.concatenate([data, abund_cube], axis = 2)

    _, abundance_plus_superpixel_library = generate_SLIC_superpixels(abundance_plus_hyperspectral_cube,
                                                                     superpixel_assignments)
    
    if ignore_label in superpixel_assignments:
        abundance_plus_superpixel_library = abundance_plus_superpixel_library[:,1::].copy()

    spectral_similarity_mtx = calc_spectral_similarity_mtx(abundance_plus_superpixel_library,
                                                           sigma2_param = spectral_param,
                                                           metric='SAM')
    
    spatial_spectral_matrix = spatial_filter * spectral_similarity_mtx
    
    superpixel_cluster_labels = sklcluster.spectral_clustering(spatial_spectral_matrix,
                                                               n_clusters=n_endmembers,
                                                               random_state = 20021225)

    mean_cluster_spectra = calc_mean_label_signatures(superpixel_library, superpixel_cluster_labels)

    return superpixel_cluster_labels, mean_cluster_spectra

def subsegment(data : np.ndarray,
                superpixel_library : np.ndarray,
                superpixel_centers : np.ndarray,
                superpixel_assignments : np.ndarray,
                segmented_labels : np.ndarray,
                subsegment_label : int,
                n_subsegments : int,
                spectral_param: float,
                spatial_param : float,
                spectral_metric : str = "EUCLIDEAN"):
    
    num_endmembers = len(np.unique(segmented_labels))
    subsegmented_labels = segmented_labels.copy()
    chunk_assignments = np.vectorize(lambda x: x if x in list(np.where(subsegmented_labels == subsegment_label)[0]) else -1)(superpixel_assignments)
    chunk_superpixel_library = superpixel_library[:,(segmented_labels == subsegment_label)].copy()
    chunk_superpixel_centers = superpixel_centers[(segmented_labels == subsegment_label),:].copy()

    chunk_labels, chunk_spectra = superpixel_subsegment(data                    = data,
                                                        superpixel_library      = chunk_superpixel_library,
                                                        superpixel_centers      = chunk_superpixel_centers,
                                                        superpixel_assignments  = chunk_assignments,
                                                        n_endmembers            = n_subsegments,
                                                        spectral_param          = spectral_param,
                                                        spatial_param           = spatial_param,
                                                        spectral_metric         = spectral_metric)
    
    mapping_labels = dict(zip(np.unique(chunk_labels), np.unique(chunk_labels) + num_endmembers - 1))
    mapping_labels[0] = subsegment_label

    subsegmented_labels[(segmented_labels == subsegment_label)] = np.vectorize(lambda x: mapping_labels[x])(chunk_labels)
    mean_cluster_spectra = calc_mean_label_signatures(superpixel_library, subsegmented_labels)

    return subsegmented_labels, 

def single_ncuts_admm(data,
                 superpixel_library,
                 superpixel_centers,
                 superpixel_assignments,
                 n_endmembers,
                 spectral_sigma2_param,
                 spatial_kappa_param,
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
    ===========================================
    Returns:
        labels
        endmember_spectra
    ===========================================
    References:
        [1] Jianbo Shi and J. Malik, "Normalized cuts and image segmentation," in IEEE 
            Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 8, 
            pp. 888-905, Aug. 2000, doi: 10.1109/34.868688.
    '''
    nx, ny, nb = data.shape
    hyperspectral_image = cube_to_matrix(data)
    spatial_filter = (calc_spatial_distance_mtx(superpixel_centers, 1, 1) < spatial_kappa_param).astype(int) # gets cached
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
    abund_mtx, _ = fclsu_admm(mean_cluster_spectra, hyperspectral_image)
    abund_cube = matrix_to_cube(abund_mtx, nx, ny, n_endmembers)

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