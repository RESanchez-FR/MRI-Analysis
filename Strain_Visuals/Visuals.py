import scipy.io 
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RectangleSelector
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata 
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import art3d


#get current working directory
path = os.getcwd()

#checking if the mat file loads
mat = scipy.io.loadmat(path + '\Strain_Visuals\data_BC.mat')

#acessing the varibales we need
L_Vector = mat['strain_data'] #shape 160 80 22 24 3 3
m_data = mat['m_data'] # shape 160 80 24 24
m_data = m_data[: , :, :-2, :] #to get into the correct shape

# Global variable to store rectangle coordinates
rect_coords = []

def on_select(eclick, erelease):
    """Callback function for rectangle selection."""
    global rect_coords
    rect_coords = [(int(eclick.xdata), int(eclick.ydata)), (int(erelease. xdata), int(erelease.ydata))]
    print(f"Selected coordinates: {rect_coords}")

    #close the window
    plt.close()

def draw_rectangle(m_data):
    """Display m_data and allow user to select a rectangle."""
    fig, ax = plt.subplots()
    ax.imshow(m_data, cmap='viridis')  # Display m_data as an image
    ax.set_title('Select Region of Interest (ROI)')

    # Create a RectangleSelector
    rectangle_selector = RectangleSelector(ax, on_select, useblit=True,
                                           button=[1],  # Left mouse button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)

    plt.show()

def extract_roi(data, m_data, rect_coords, slice_index, frame_index):
    """
    Extracts a region of interest (ROI) from the 6D data arrays based on selected rectangle coordinates.
    
    Parameters:
    - data: The original 6D data array.
    - m_data: The original mask or metadata array.
    - rect_coords: Coordinates of the selected rectangle.
    - slice_index: The index for the third dimension.
    - frame_index: The index for the fourth dimension.

    Returns:
    - roi_data: The sliced data array for the ROI.
    - roi_m_data: The sliced m_data array for the ROI.
    """
    (x_min, y_min), (x_max, y_max) = rect_coords
    roi_data = data[x_min:x_max, y_min:y_max]
    roi_m_data = m_data[x_min:x_max, y_min:y_max]
    
    return roi_data, roi_m_data

def add_quiver_3D(ax, X, Y, Z, u, v, w=None, density=1):
    # Subsample the grid points for clearer visualization
    x_idx = slice(None, None, density)
    y_idx = slice(None, None, density)
    
    if w is None:
        # Plot 2D quiver on the surface
        ax.quiver(X[x_idx, y_idx],
                  Y[x_idx, y_idx],
                  Z[x_idx, y_idx],
                  u[x_idx, y_idx],
                  v[x_idx, y_idx],
                  0,  # Set Z component to 0
                  linewidths=1.5,
                  length=0.4,
                  normalize=True,
                  color='black',
                  zorder=10,
                  alpha=1.0, 
                  arrow_length_ratio=0)
    else:
        # Original 3D quiver plot
        ax.quiver(X[x_idx, y_idx],
                  Y[x_idx, y_idx],
                  Z[x_idx, y_idx],
                  u[x_idx, y_idx],
                  v[x_idx, y_idx],
                  w[x_idx, y_idx],
                  linewidths=1.5,
                  length=0.4,
                  normalize=True,
                  color='black',
                  zorder=10,
                  alpha=1.0, 
                  arrow_length_ratio=0)




# #### only doing the ROI part 2D
# def plotting_2D(data, m_data, slice, frame, eigenvector):
#     # Extract data
#     indexed_data = data[:, :, slice, frame, eigenvector, :2]  # Only take X and Y components
#     indexed_m_data = m_data[:, :, slice, frame]

#     # Select ROI
#     draw_rectangle(indexed_m_data)
#     roi_data, roi_m_data = extract_roi(indexed_data, indexed_m_data, rect_coords, slice, frame)

#     # Create meshgrid
#     x = np.arange(roi_data.shape[1])
#     y = np.arange(roi_data.shape[0])
#     X, Y = np.meshgrid(x, y, indexing='ij')

#     # Prepare mask and data
#     mask = roi_m_data.T[:, :, np.newaxis] > 0
#     roi_data = np.swapaxes(roi_data, 0, 1)

#     components = []
#     for i in range(2):  # Only process X and Y components
#         Z = np.where(mask.squeeze(), roi_data[:, :, i], np.nan)
#         valid_mask = ~np.isnan(Z)
#         points = np.column_stack((X[valid_mask], Y[valid_mask]))
#         values = Z[valid_mask]

#         xi = np.linspace(X.min(), X.max(), X.shape[0])
#         yi = np.linspace(Y.min(), Y.max(), Y.shape[1])
#         xi, yi = np.meshgrid(xi, yi, indexing='ij')
        
#         Z = griddata(points, values, (xi, yi), method='cubic')
#         Z = gaussian_filter(Z, sigma=1.5)
#         components.append(Z)

#     U, V = components

#     # Create figure and axes
#     fig, ax = plt.subplots(figsize=(10, 8))

#     ###### added to show then magnitude image #####
#     # Plot the magnitude image
#     im = ax.imshow(roi_m_data, cmap='gray', origin='lower', extent=[xi.min(), xi.max(), yi.min(), yi.max()])
    
#     # Add colorbar for the magnitude image
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label('Magnitude')

#     ###### end oif added section

#     # Plot streamlines
#     ax.streamplot(xi.T, yi.T, U.T, V.T, density=2, color='red', linewidth=1.5, arrowsize=0) #do transpose and it works

#     # Set labels and title
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_title('2D Streamplot of Vector Field')

#     # Show the plot
#     plt.show()

# # # Usage example (assuming data and m_data are already defined)
# plotting_2D(L_Vector, m_data, slice=15, frame=17, eigenvector=0)

"""whole image"""

def plotting_2D(data, m_data, slice, frame, eigenvector, threshold=0.03):
    # Extract data
    indexed_data = data[:, :, slice, frame, eigenvector, :2]  # Only take X and Y components
    indexed_m_data = m_data[:, :, slice, frame]

    # Create meshgrid
    x = np.arange(indexed_data.shape[1])
    y = np.arange(indexed_data.shape[0])
    X, Y = np.meshgrid(x, y, indexing='ij')

    X = X.T
    Y = Y.T

    # Prepare mask and data
    mask = indexed_m_data > threshold
    
    print(f"shape of X", X.shape)
    print(f"shape of Y", Y.shape)

    components = []
    for i in range(2):  # Only process X and Y components
        Z = np.where(mask, indexed_data[:, :, i], np.nan)
        valid_mask = ~np.isnan(Z)
        points = np.column_stack((X[valid_mask], Y[valid_mask]))
        values = Z[valid_mask]

        xi = np.linspace(X.min(), X.max(), X.shape[0])
        yi = np.linspace(Y.min(), Y.max(), Y.shape[1])
        xi, yi = np.meshgrid(xi, yi, indexing='ij')
        
        Z = griddata(points, values, (xi, yi), method='cubic')
        Z = gaussian_filter(Z, sigma=1.5)
        components.append(Z)

    U, V = components

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    indexed_m_data = np.flipud(indexed_m_data)

    ###### added to show then magnitude image #####
    # Plot the magnitude image
    im = ax.imshow(indexed_m_data, cmap='gray', origin='lower', extent=[xi.min(), xi.max(), yi.min(), yi.max()])
    
    # Add colorbar for the magnitude image
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Magnitude')

    ###### end of added section

    # Plot streamlines
    ax.streamplot(xi.T, yi.T, np.flipud(U.T), np.flipud(V.T), density=9, color='red', linewidth=1.5, arrowsize=0) #do transpose and it works and increase Density for more lines


    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('2D Streamplot of Vector Field')

    # Show the plot
    plt.show()



# Usage example (assuming data and m_data are already defined)
plotting_2D(L_Vector, m_data, slice=11, frame=11, eigenvector=0)


def plotting_3D(data, m_data, slice, frame, eigenvector):
    # Extract data
    indexed_data = data[:, :, slice, frame, eigenvector, :]  # Take all 3 components
    indexed_m_data = m_data[:, :, slice, frame]

    # Select ROI
    draw_rectangle(indexed_m_data)
    roi_data, roi_m_data = extract_roi(indexed_data, indexed_m_data, rect_coords, slice, frame)

    # Create meshgrid
    x = np.arange(roi_data.shape[1])
    y = np.arange(roi_data.shape[0])
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Prepare mask and data
    mask = roi_m_data.T[:, :, np.newaxis] > 0
    roi_data = np.swapaxes(roi_data, 0, 1)

    components = []
    for i in range(3):  # Process X, Y, and Z components
        Z = np.where(mask.squeeze(), roi_data[:, :, i], np.nan)
        valid_mask = ~np.isnan(Z)
        points = np.column_stack((X[valid_mask], Y[valid_mask]))
        values = Z[valid_mask]

        xi = np.linspace(X.min(), X.max(), X.shape[0])
        yi = np.linspace(Y.min(), Y.max(), Y.shape[1])
        xi, yi = np.meshgrid(xi, yi, indexing='ij')
        
        Z = griddata(points, values, (xi, yi), method='cubic')
        Z = gaussian_filter(Z, sigma=1.5)
        components.append(Z)

    #faster 
    U, V, W = components

    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # # Plot the surface
    # surf = ax.plot_surface(xi, yi, W, cmap='viridis', alpha=0.8)

    # Create streamlines using scatter plot
    n_points = 2000  # Reduced number of starting points
    n_steps = 15  # Reduced number of steps per streamline
    step_size = 1  # Increased step size for faster traversal

    start_points = np.random.rand(n_points, 2) * np.array([xi.max(), yi.max()])

    # Prepare interpolation functions
    from scipy.interpolate import RegularGridInterpolator
    u_interp = RegularGridInterpolator((xi[:,0], yi[0,:]), U)
    v_interp = RegularGridInterpolator((xi[:,0], yi[0,:]), V)
    w_interp = RegularGridInterpolator((xi[:,0], yi[0,:]), W)

    all_points = []
    for start_point in start_points:
        x, y = start_point
        points = [start_point]
        for _ in range(n_steps):
            u = u_interp((x, y))
            v = v_interp((x, y))
            x += u * step_size
            y += v * step_size
            if x < xi.min() or x > xi.max() or y < yi.min() or y > yi.max():
                break
            points.append([x, y])
        all_points.extend(points)

    all_points = np.array(all_points)
    z = w_interp((all_points[:, 0], all_points[:, 1]))

    # Plot all points at once
    scatter = ax.scatter(all_points[:, 0], all_points[:, 1], z, c='b',  s=1, alpha=1)

    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Surface with Streamlines (Scatter Plot)')

    # # Add colorbar
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    # fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Z value')

    # Show the plot
    plt.show()





# plotting_3D(L_Vector, m_data, slice=15, frame=17, eigenvector=0)