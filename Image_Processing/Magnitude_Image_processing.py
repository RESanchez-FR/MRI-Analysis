import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 
import cv2.ximgproc as ximgproc
import scipy.io 
import os

#get current working directory
path = os.getcwd()

#checking if the mat file loads
mat = scipy.io.loadmat(path + '\Strain_Visuals\data_BC.mat')

#acessing the varibales we need
L_Vector = mat['strain_data'] #shape 160 80 22 24 3 3
m_data = mat['m_data'] # shape 160 80 24 24
m_data = m_data[: , :, :-2, :] #to get into the correct shape




def denoise_magnitude_image(magnitude_data, slice, frame, alpha=0.5, K=0.02, niters=10, plot=False):
    magnitude_slice = np.squeeze(magnitude_data[:, :, slice, frame])
    raw_data = magnitude_slice.copy()

    # Normalize and preprocess
    magnitude_norm = cv.normalize(magnitude_slice, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # Apply bilateral filter with d = 5,  
    equalized = cv.equalizeHist(magnitude_norm)

    # Anisotropic diffusion
    magnitude_3ch = cv.cvtColor(equalized, cv.COLOR_GRAY2BGR)
    filtered_mag_3ch = ximgproc.anisotropicDiffusion(magnitude_3ch, alpha=alpha, K=K, niters=niters)
    filtered_mag = cv.cvtColor(filtered_mag_3ch, cv.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv.Canny(filtered_mag, 100, 200)
    # Edge detection
    t_lower =  65 # Lower Threshold , change this one for a good edge detection
    t_upper = 200  # Upper threshold 
    apertureSize = 3 # 3, 5, 7, higher size detects more edges
    edges = cv.Canny(filtered_mag, t_lower, t_upper, apertureSize = apertureSize)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
        
        im1 = ax1.imshow(raw_data, cmap='gray')
        ax1.set_title("Raw Magnitude Image")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(filtered_mag, cmap='gray')
        ax2.set_title("Filtered Magnitude Image")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        plt.colorbar(im2, ax=ax2)

        im3 = ax3.imshow(edges, cmap='gray')
        ax3.set_title("Edge Detection")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")

        plt.show()

    return filtered_mag, edges



def process_and_visualize_edges(filtered_mag, edges, raw_data):
    # Manual region selection (you'll need to implement this based on your needs)
    # For example, you could use cv.selectROI() to allow the user to select a region

    # Morphological operations
    kernel = np.ones((1,2), np.uint8)
    dilated_edges = cv.dilate(edges, kernel, iterations=2)
    closed_edges = cv.morphologyEx(dilated_edges, cv.MORPH_CLOSE, kernel)

    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    
    axs[0, 0].imshow(raw_data, cmap='gray')
    axs[0, 0].set_title("Raw Magnitude Image")
    
    axs[0, 1].imshow(filtered_mag, cmap='gray')
    axs[0, 1].set_title("Filtered Magnitude Image")
    
    axs[1, 0].imshow(edges, cmap='gray')
    axs[1, 0].set_title("Edge Detection")
    
    axs[1, 1].imshow(closed_edges, cmap='gray')
    axs[1, 1].set_title("Processed Edges")

    for ax in axs.flat:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()

    return closed_edges


slice = 11
frame = 11

filtered_mag, edges = denoise_magnitude_image(m_data, slice = slice , frame = frame)



# raw_data = np.squeeze(m_data[:, :, slice, frame])
# processed_edges = process_and_visualize_edges(filtered_mag, edges, raw_data)
