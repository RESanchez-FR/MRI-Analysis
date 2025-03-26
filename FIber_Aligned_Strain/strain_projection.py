import numpy as np 
import matplotlib.pyplot as plt
from DTI_preprocessing import process_and_save_nii_slices
import os
import scipy.io
import cv2
from scipy.spatial.transform import Rotation
from scipy.stats.mstats import winsorize
import sys
from scipy import odr

from scipy.signal import savgol_filter

"""
    STEP 1: Fiber Aligned Projection Strain
    This Script is Dedicated to the Fiber Aligned Strain Projection. 
    We look at a region of intrest and determine the Strain Values in the fiber aligned strain
    at each voxel. We calculate the angular deviation between e3 - compressive strain 
    and v1 our principal eigenvector DTI data """

main_path = os.getcwd()
poly_path = os.path.join(os.path.dirname(__file__), main_path + '\smoothing')
# Add the directory to sys.path
sys.path.insert(0, poly_path)



# Global variables
contour_points = []
magnitude_image = None
strain_vec_prev_frame = None 

def draw_contour(event):
    global contour_points, magnitude_image
    
    if event.button == 1:  # Left mouse button
        if event.name == 'button_press_event':
            contour_points = [(event.xdata, event.ydata)]
        elif event.name == 'motion_notify_event':
            contour_points.append((event.xdata, event.ydata))
            x, y = zip(*contour_points)
            plt.gca().plot(x, y, 'r-', linewidth = 1.5)
            plt.draw()
        elif event.name == 'button_release_event':
            print("Contour drawing completed")




def visualize_magnitude_image(m_data, z_slice):
    global magnitude_image
    
    magnitude_image = np.squeeze(m_data[:, :, z_slice, 0])  # Assuming first time point
    
    plt.figure(figsize=(10, 10))
    plt.imshow(magnitude_image, cmap='gray')
    plt.title('Draw contour on the magnitude image')
    plt.connect('button_press_event', draw_contour)
    plt.connect('motion_notify_event', draw_contour)
    plt.connect('button_release_event', draw_contour)
    plt.show()


def compute_fiber_aligned_strain(dti_data, strain_data, contour, z_slice, t, eigenvector_index=0):
    """
    Compute fiber-aligned strain from DTI and strain tensor data within the contour for a single slice and time point.
    
    Parameters:
    dti_data: np.array of shape (X, Y, Z, 3) - principal eigenvector (ν₁)
    strain_data: np.array of shape (X, Y, Z, T, 3, 3) - strain tensor (E)
    contour: list of (x, y) tuples defining the ROI
    z_slice: int, the slice number to analyze
    t: int, the frame we are looking at
    eigenvector_index: int, which eigenvector to use (default is ε₃ = 2)
    
    Returns:
    E_fiber_avg: float - average fiber-aligned strain within ROI
    E_fiber_std: float - standard deviation of fiber-aligned strain within ROI
    theta_avg: float - average angle between ε₃ and ν₁ within ROI (in degrees)
    theta_std: float - standard deviation of angles (in degrees)
    """
    global strain_vec_prev_frame  # Declare global variable to modify it

    # Create a mask from the contour
    mask = np.zeros(dti_data.shape[:2], dtype=np.uint8)
    contour_array = np.array(contour, dtype=np.int32)
    cv2.fillPoly(mask, [contour_array], 1)

    # Normalize DTI principal eigenvector ν₁
    v1 = dti_data[:, :, z_slice]
    v1_norm = np.linalg.norm(v1, axis=2, keepdims=True)
    v1_normalized = np.where(v1_norm > 1e-6, v1 / v1_norm, 0)  # Avoid division by zero

    # Extract strain tensor for the current frame and slice
    E = strain_data[:, :, z_slice, t]  # This reduces to a [x,y,3,3] index now


   
    def smooth_strain_data(data, mask, window_length=2, polyorder=1):
        """
        Smooths strain data along the last dimension of E[:, :, 0, :] within the masked region using Savitzky-Golay filtering.

        Parameters:
        - data: array_like, shape (x, y, 3, 3) - Strain tensor data.
        - mask: array_like, shape (x, y) - Mask defining the ROI.
        - window_length: int - Length of the filter window (must be odd).
        - polyorder: int - Order of the polynomial used for smoothing.

        Returns:
        - smoothed_data: array_like - Smoothed strain tensor data within the masked region.
        """
        smoothed_data = np.copy(data)  # Initialize with original data
        
        # Get indices of masked pixels
        x_idx, y_idx = np.where(mask == 1)
        
        # Apply smoothing only to the masked region for E[:, :, 0, :]
        for x, y in zip(x_idx, y_idx):
            component_values = data[x, y, 0, :]
            
            # Apply Savitzky-Golay filter to smooth the component values
            smoothed_component_values = savgol_filter(
                component_values,
                window_length=window_length,
                polyorder=polyorder, 
                mode = 'constant'
            )

            #             # Flip values into Quadrant IV (negative) for the last index only if positive
            # smoothed_component_values[-1] = (
            #     -smoothed_component_values[-1] if smoothed_component_values[-1] > 0 else smoothed_component_values[-1]
            # )
            
            # Assign smoothed values back to the original data
            smoothed_data[x, y, 0, :] = smoothed_component_values
        
        return smoothed_data


    # Smooth the strain data
    E_smooth = smooth_strain_data(E , mask )

    # Get mask indices for ROI
    x_idx, y_idx = np.where(mask == 1)

    # Vectorized eigenvalue decomposition for masked pixels
    E_masked = E[x_idx, y_idx]
    E_masked_smooth = E_smooth[x_idx, y_idx]

    # Compute angles between ν₁ and selected strain vector
    v1_masked = v1_normalized[x_idx, y_idx]  # extracts the information based on the contour we drew ROI

    # Directly use the principal eigenvector from strain data
    strain_vec_principal = E_masked_smooth[:, 0, :]  # Assuming E_masked_smooth is your smoothed strain data
    
    # Normalize the strain vector
    norm_strain_vec = strain_vec_principal / np.linalg.norm(strain_vec_principal, axis=1, keepdims=True)
    # Compute angles between ν₁ and selected strain vector
    dot_products = np.sum(norm_strain_vec * v1_masked, axis=1)
    theta = np.arccos(np.clip(np.abs(dot_products), -1.0, 1.0))  # Angle in radians

  
    # Compute fiber-aligned strain projection
    E_fiber = np.matmul(np.matmul(v1_masked, E_masked_smooth), v1_masked.T)
   

    # Update global variable for next frame's orientation consistency
    if strain_vec_prev_frame is None:
        strain_vec_prev_frame = np.zeros_like(v1_normalized)  # Initialize once globally

    strain_vec_prev_frame[x_idx, y_idx] = strain_vec_principal

    # No outliers
    E_avg = np.mean(E_fiber)
    E_avg = -E_avg if E_avg > 0 else E_avg
    E_std = np.std(E_fiber)
    theta_avg = np.degrees(np.mean(theta))
    theta_std = np.degrees(np.std(theta))

    "before quadrant flip"

    component_x = 0 #0 for x , 1 for y, 2 for z
    component_y = 1

    # smoothed_strain_x = savgol_filter(E_masked_smooth[:,0,component_x], window_length=11, polyorder=3, mode = "nearest")
    # smoothed_strain_y = savgol_filter(E_masked_smooth[:,0,component_y], window_length=11, polyorder=3, mode = "nearest")

    average_strain_x = np.mean(E_masked_smooth[:,0,component_x])  # Average of E_11 component, spatial coordinates and avg principal eigenvector values

    average_strain_x_std = np.std(average_strain_x)

    average_strain_y = np.mean(E_masked_smooth[:,0,component_y])  # Average of E_11 component, spatial coordinates and avg principal eigenvector values

    average_strain_y_std = np.std(average_strain_y)

    # Step 2: Flip quadrants (make all values below y=0)
    flipped_average_strain_x = -average_strain_x if average_strain_x > 0 else average_strain_x
    flipped_average_strain_y = -average_strain_y if average_strain_y > 0 else average_strain_y


    return (
        E_avg,               # Average fiber-aligned strain
        E_std,                # Standard deviation of fiber-aligned strain
        theta_avg,     # Average angle in degrees
        theta_std,        # Standard deviation of angles in degrees
        flipped_average_strain_x,         #Average strain for E1xx
        flipped_average_strain_y,  #Average Strain for E1xy
        average_strain_x_std, 
        average_strain_y_std          
    )



#Main execution
if __name__ == "__main__":

    path = os.getcwd()

    mat_data_path = path + '\Patient_Data\JH_Data\JH.mat'

    mat_data = scipy.io.loadmat(mat_data_path)

    L_vector = mat_data['L_vector']
    L_vector = L_vector[1:-1, 2:-2, :, :, :, : ] ## should be 1 instead of two but check later the nii file

    print(L_vector.shape)

    m_data = mat_data['m_data']

    dti_data = process_and_save_nii_slices('20231129_F020Y_2023112916JH_s03_fiber2.nii', 'Images/JH_DTI_Images')

    dti_data = np.swapaxes(dti_data, 0 , 2)
   
    z_slice = 10
    
    # Visualize magnitude image and draw contour
    visualize_magnitude_image(m_data, z_slice = z_slice)  # Adjust slice number as needed


    # Initialize storage
    E_avgs, E_stds = [], []
    theta_avgs, theta_stds = [], []

    # Initialize storage for new metrics
    strain_avgs_x, strain_avgs_y = [], []
    strain_x_stds, strain_y_stds = [] , []

    # Get number of time frames
    num_frames = L_vector.shape[3]  



    # Inside your processing loop:
    for t in range(num_frames):
        e_avg, e_std, theta_avg, theta_std, strain_avg_x, strain_avg_y, strain_x_std, strain_y_std = compute_fiber_aligned_strain(
            dti_data, L_vector, contour_points, z_slice, t)
        
        # Append new metrics
        if t == 0:
            strain_avgs_x.append(0)
            E_avgs.append(0)
            strain_avgs_y.append(0)
            theta_avgs.append(0)
            theta_stds.append(0)
        else:
            strain_avgs_x.append(strain_avg_x)
            E_avgs.append(e_avg)
            strain_avgs_y.append(strain_avg_y)
            theta_avgs.append(theta_avg)
            theta_stds.append(theta_std)
       

        E_stds.append(e_std)
        strain_x_stds.append(strain_x_std)
        strain_y_stds.append(strain_y_std)



    # Modified plotting section
    plt.figure(figsize=(16, 10))

    # Original Strain Plot
    plt.subplot(2, 2, 1)
     # Plot with error bars
    plt.errorbar(range(num_frames), E_avgs, yerr=E_stds, 
                fmt='o', capsize=10, label='Fiber-aligned Strain')
    plt.xlabel('Time Frame')
    plt.ylabel('E₁₁')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Fiber Aligned Strain with Error Bars')
    plt.grid(False)

    # Angle Plot
    plt.subplot(2, 2, 2)
    plt.errorbar(range(num_frames), theta_avgs, yerr=theta_stds,
                fmt='o', capsize=5, color='orange', label='Angle Deviation')
    plt.xlabel('Time Frame')
    plt.ylabel('θ (degrees)')
    plt.title('Angular Deviation with Error Bars')
    plt.grid(False)

    # New Strain Component Plot
    plt.subplot(2, 2, 3)
    plt.errorbar(range(num_frames), strain_avgs_x, yerr = strain_x_stds, 
                    fmt = 'o-' ,  color='green')
    plt.xlabel('Time Frame')
    plt.ylabel('Strain Averages E1xx')
    plt.ylim(-1,1)
    # Add horizontal line at y=0 for clarity
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Average Strains E1 xx for ROI')
    plt.grid(False)

    # New DTI Magnitude Plot
    plt.subplot(2, 2, 4)
    plt.errorbar(range(num_frames), strain_avgs_y, yerr = strain_y_stds, 
                     fmt = 'o-', color='purple')
     # Add horizontal line at y=0 for clarity
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Time Frame')
    plt.ylabel('Strain Averages E1 xy')
    plt.ylim(-1,1)
    plt.title('Average Strains E1 xy in ROI')
    plt.grid(False)

    plt.tight_layout()
    plt.show()

