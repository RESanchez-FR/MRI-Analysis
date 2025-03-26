import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt 
import os 
import cv2
from cv2 import ximgproc



def process_and_save_nii_slices(nii_file_name, output_folder, brightness_factor=2, contrast_factor=1, start_slice=None, end_slice=None):
    """
    Process a .nii file, create grayscale images for each slice, and save them.

    Parameters:
    nii_file_name (str): Name of the .nii file (should be in a 'Nii_files' folder in the current directory)
    output_folder (str): Name of the folder to save output images (will be created in the current directory)
    brightness_factor (float): Factor to adjust brightness (default 2)
    contrast_factor (float): Factor to adjust contrast (default 1)
    start_slice (int): Starting slice number (default None, which means start from the first slice)
    end_slice (int): Ending slice number (default None, which means process all slices)

    Returns:
    nii_data
    """
    # Get the current working directory
    path = os.getcwd()

    # Load the .nii file
    nii_img = nib.load(os.path.join(path, 'Nii_files', nii_file_name))
    nii_data = nii_img.get_fdata()

    # print("Shape after loading:", nii_data.shape)

    # Create the output folder if it doesn't exist
    images_path = os.path.join(path, output_folder)
    os.makedirs(images_path, exist_ok=True)

    # Get the range of z_slices
    if start_slice is None:
        start_slice = 0
    if end_slice is None:
        end_slice = nii_data.shape[0]
    z_slices = range(start_slice, end_slice)

    for z_slice in z_slices:
        # Extract the slice data
        data_slice = np.abs(nii_data[z_slice, :, :])

        # Normalize and convert to uint8 with brightness adjustment
        min_val = np.nanmin(data_slice)
        max_val = np.nanmax(data_slice)

        normalized_data = ((data_slice - min_val) / (max_val - min_val) * 255 * brightness_factor).clip(0, 255).astype(np.uint8)

        # Adjust contrast
        normalized_data = cv2.addWeighted(normalized_data, contrast_factor, normalized_data, 0, 0)

        # Ensure the image is in the correct format (uint8, single channel)
        normalized_data = cv2.convertScaleAbs(normalized_data)

        # Convert back to float and rescale to [0, 1]
        grayscale_image = normalized_data.astype(float) / 255.0

        # # Plot the result
        # plt.figure(figsize=(10, 10))
        # plt.imshow(grayscale_image, cmap='gray')
        # plt.title(f'Slice {z_slice + 1} - Grayscale')
        # plt.axis('off')

        # # Save the image
        # plt.savefig(os.path.join(images_path, f'fiber_direction_slice_{z_slice + 1:03d}.png'))
        # plt.close()  # Close the figure to free up memory

        # print(f"Processed and saved slice {z_slice + 1}")

    return nii_data



# process_and_save_nii_slices('20231129_F020Y_2023112916JH_s03_fiber2.nii', 'Images/JH_DTI_Images')
