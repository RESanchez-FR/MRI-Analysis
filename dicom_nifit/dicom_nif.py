
import os
from glob import glob
import pydicom
import numpy as np
import nibabel as nib

"Create a folder path where the dicom images can be read "

def dicom_to_nifti(dicom_dir, output_file):
    # Find all DICOM files in the directory
    dicom_files = glob(os.path.join(dicom_dir, '*.dcm'))
    
    # Sort the files to ensure correct order
    dicom_files.sort()
    
    # Read the first DICOM file to get metadata
    first_slice = pydicom.dcmread(dicom_files[0])
    
    # Extract necessary metadata, handling potentially missing attributes
    pixel_spacing = getattr(first_slice, 'PixelSpacing', None)
    if pixel_spacing is None:
        pixel_spacing = getattr(first_slice, 'ImagerPixelSpacing', None)
    if pixel_spacing is None:
        pixel_spacing = [1, 1]  # Default to 1mm if no spacing information is available

    slice_thickness = getattr(first_slice, 'SliceThickness', None)
    if slice_thickness is None:
        slice_thickness = getattr(first_slice, 'SpacingBetweenSlices', 1)  # Default to 1mm if not available
    
    # Read all DICOM files and stack them into a 3D array
    slices = [pydicom.dcmread(dcm) for dcm in dicom_files]
    image_3d = np.stack([s.pixel_array for s in slices])
    
    # Create affine matrix
    affine = np.eye(4)
    affine[0, 0] = pixel_spacing[0]
    affine[1, 1] = pixel_spacing[1]
    affine[2, 2] = slice_thickness
    
    # Create NIfTI image
    nifti_image = nib.Nifti1Image(image_3d, affine)
    
    # Save NIfTI file
    nib.save(nifti_image, output_file)

# Keep the original input and output sections
path = os.getcwd()

# Example usage
dicom_folder = path + "\\dicom_images\\KJ\\DTI_30_dir50pFOVonlyb0sagPAmCS_16"
output_nifti = path + "\\Nii_files\\KJ_PA.nii.gz"
dicom_to_nifti(dicom_folder, output_nifti)
