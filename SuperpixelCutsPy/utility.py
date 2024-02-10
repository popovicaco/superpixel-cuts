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

def save_hcube(hyperspectral_cube : np.ndarray,
               n_layers : int = 50,
               output_img : str = 'output_img.png'):

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    step_size = hyperspectral_cube.shape[2]//n_layers
    plot_cube = np.flip(hyperspectral_cube[:,:,np.arange(0,hyperspectral_cube.shape[2],step_size)], axis=(0))
    # Create a sample 3D matrix
    nx,ny,nz = plot_cube.shape
    # Create figure and 3D axis
    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');

    # Create meshgrid for x, y, z
    x, y, z = np.meshgrid(range(nx), range(ny), range(nz))

    # Plot cube
    ax.scatter(x, z, y, c=plot_cube.flatten(), cmap='viridis', marker='s');

    # Set the aspect ratio to be equal for all axes
    ax.set_box_aspect([1, 1, 1]);
    ax.axis('off');# Show plot
    fig.patch.set_alpha(0);

    # Save the figure with the transparent background
    plt.savefig(output_img, bbox_inches='tight', transparent=True);
    plt.close()

def save_img(image : np.ndarray,
             output_img : str = 'output_img.png'):
    import matplotlib.pyplot as plt

    # Create figure and 3D axis
    fig = plt.figure();
    ax = fig.add_subplot();

    # Create meshgrid for x, y, z
    ax.imshow(image);

    # Set the aspect ratio to be equal for all axes
    ax.axis('off');# Show plot
    fig.patch.set_alpha(0);

    # Save the figure with the transparent background
    plt.savefig(output_img, bbox_inches='tight', transparent=True);
    plt.close();

