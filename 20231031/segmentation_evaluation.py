# Imports
import numpy as np
from sklearn.metrics import confusion_matrix
from normalized_cuts import *
from preprocessing import *
from superpixel import *
from unmixing import *

def justify_segmentation_labels(ground_truth, predicted):
    '''
    Description:
        In cases where segmentation might not produce the same labels as the ground_truth, we have to determine the
        "true" labels outselves. We do this by comparing the masks. If two classes have the largest intersection, they are deemed the same.
    ===========================================
    Parameters:
        ground_truth            - 
        predictions             -
    ===========================================
    Returns:
        adjusted_predictions    - 
    '''
    gt_unique_labels = list(np.unique(ground_truth))
    pred_unique_labels = list(np.unique(predicted))
    segmentation_mapping = {}

    for gt_label in gt_unique_labels:
        pred_max_label = 0
        max_intersection = -1
        for pred_label in pred_unique_labels:
            curr_intersection = ((ground_truth == gt_label) * (predicted == pred_label)).sum()
            if curr_intersection > max_intersection and pred_label not in segmentation_mapping.values():
                pred_max_label = pred_label
                max_intersection = curr_intersection
        segmentation_mapping[gt_label] = pred_max_label
    segmentation_mapping_inverted = dict((v, k) for k, v in segmentation_mapping.items())
    segmentation_map = np.vectorize(lambda x : segmentation_mapping_inverted[x])
    return segmentation_map(predicted)

def calc_confusion_matrix(ground_truth, predicted):
    '''
    Description:
        Calculates confusion matrix given ground truth data and a prediction
    ===========================================
    Parameters:
        ground_truth
        prediction
    ===========================================
    Returns:
        confusion_mtx
    '''
    confusion_mtx = confusion_matrix(ground_truth.reshape(-1), predicted.reshape(-1))
    return confusion_mtx

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def compare_segmentations(prev_prediction,curr_prediction):
    '''
    Description:
        In cases where segmentation might not produce the same labels as the ground_truth, we have to determine the
        "true" labels outselves. We do this by comparing the masks. If two classes have the largest intersection, they are deemed the same.
    ===========================================
    Parameters:
        prev_prediction            - 
        curr_prediction             -
    ===========================================
    Returns:
        confusion_mtx    - 
    '''
    adjusted_curr_prediction = justify_segmentation_labels(prev_prediction,curr_prediction)
    confusion_mtx = calc_confusion_matrix(prev_prediction, adjusted_curr_prediction)
    return confusion_mtx

def add_noise_per_label(data, ground_truth, std_mtx, pct_std):
    '''
    Description:
        Adds Noise to each Endmember at each Layer
    ===========================================
    Parameters:
        data
        ground_truth
        std_mtx
        pct_std
    ===========================================
    Returns:
        contaminated_data
    '''
    nx,ny,nb = data.shape
    contaminated_data = data.copy()
    unique_labels, unique_labels_counts = np.unique(ground_truth, return_counts=True)
    for label in unique_labels:
        for k in range(nb):
            noise_by_label_by_layer = np.random.normal(0,pct_std*std_mtx[k, label], unique_labels_counts[np.where(unique_labels==label)] )
            contaminated_data[np.where(ground_truth == label)[0],np.where(ground_truth == label)[1],k] += noise_by_label_by_layer
    contaminated_data = np.clip(contaminated_data, 0, np.inf)
    return contaminated_data

def blur_cube(data, sigma = 1):
    '''
    Description:
        Adds Noise to each Endmember at each Layer
    ===========================================
    Parameters:
        data
    ===========================================
    Returns:
        contaminated_data
    '''
    nx,ny,nb = data.shape
    data = data.copy()
    for k in range(nb):
        data[:,:,k] = sp.ndimage.gaussian_filter(data[:,:,k], sigma = sigma)
    return data