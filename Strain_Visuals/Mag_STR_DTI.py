import numpy as np
import matplotlib.pyplot as plt
import scipy.io 
import os
import cv2
import cv2.ximgproc as ximgproc
from scipy.signal import medfilt2d
import imageio
from matplotlib.widgets import Button
import mat73

"""This Python Script plots the RGB image of the strain data
    Optional at the end to display the data. Just uncomment the line showing the plot below"""

#get current working directory
path = os.getcwd()

#checking if the mat file loads
mat = scipy.io.loadmat(path + '\Strain_Visuals\AL_30MVC.mat')

#acessing the varibales we need
L_Vector = mat['L_vector'] #shape 160 80 24 32 3 3
L_Vector = L_Vector[1:-1 , 1:-1 , : , :, :, :]
m_data = mat['m_data'] # shape 160 80 24 32

# Global variable to store contour points
contour_points = []
lines = []

img_path = path + '\Images\Computing_Image'
# Use os.path.join for robust path construction
mag_img_path = os.path.join(img_path, 'Mag_IMG.png') #Changed path
fiber_img_path = os.path.join(img_path, 'DTI_Slice12_AL.png') #Changed path


mag_img = cv2.imread(mag_img_path) #path from join

fiber_img = cv2.imread(fiber_img_path) #path from join

# strain_img = cv2.imread(strain_img_path)
z_slice = 17 # indicate for slice + 1 on matlab
frame = 17 # indicate for frame + 1 on matlab



def save_figure():
    plt.savefig('contour_plot.png', dpi=300, bbox_inches='tight')
    print("Figure saved as contour_plot.png")



def visualize_strain_data(eigenVectors, m_data, eigen_index, z_slice, frame, threshold=0.03):
    global contour_points, line1, line2, lines, mag_img_colored_cumulative, grayscale_image, rgb_image, fiber_img, lines_ax2 #add the global vars to the definition
    contour_points = []  # Initialize contour_points
    lines = []
    lines_ax2 = [] # Store separate lines for ax2

    # Prepare the mask
    mask = np.squeeze(m_data[:, :, z_slice, frame]) <= threshold

    """Fix the brightness of Magnitude Image"""

    magnitude_data = np.squeeze(m_data[:, :, z_slice, frame])

    # Extract the slice data
    mag_slice = np.abs(magnitude_data)

    # Normalize and convert to uint8 with brightness adjustment
    min_val = np.nanmin(mag_slice)
    max_val = np.nanmax(mag_slice)

    brightness_factor = 1.6  # Increase this value to increase brightness
    normalized_data = ((mag_slice - min_val) / (max_val - min_val) * 255 * brightness_factor).clip(0, 255).astype(np.uint8)

    # Adjust contrast
    contrast_factor = 1  # Increase this value to increase contrast
    normalized_data = cv2.addWeighted(normalized_data, contrast_factor, normalized_data, 0, 0)

    # Ensure the image is in the correct format (uint8, single channel)
    normalized_data = cv2.convertScaleAbs(normalized_data)

    # Convert back to float and rescale to [0, 1]
    grayscale_image = normalized_data.astype(float) / 255.0

    """end of magnitude data manipulation"""

    """Working on the RGB format now"""
    # Extract and process the eigenVector data for color map
    data_slice = np.abs(np.squeeze(eigenVectors[:, :, z_slice, frame, eigen_index, :]))

    # Apply mask
    data_slice[np.repeat(mask[:, :, np.newaxis], 3, axis=2)] = np.nan

    # Normalize and convert to uint8
    normalized_data = np.zeros_like(data_slice)
    for i in range(3):
        channel_data = data_slice[:, :, i]
        min_val = np.nanmin(channel_data)
        max_val = np.nanmax(channel_data)
        normalized_data[:, :, i] = ((channel_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Ensure the image is in the correct format (uint8, 3 channels)
    normalized_data = cv2.convertScaleAbs(normalized_data)

    # Apply anisotropic diffusion filter
    filtered_data = ximgproc.anisotropicDiffusion(
        normalized_data,
        alpha=1, K=0.03, niters=8
    )

    # Convert back to float and rescale to [0, 1]
    rgb_image = filtered_data.astype(float) / 255.0

    # Get image dimensions
    img_height, img_width = magnitude_data.shape[:2]  # Use the actual image shape

    # Create figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))  # Adjusted figure size

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    ax1.grid(False)  # hide the grid
    ax2.grid(False)
    ax3.grid(False)

    # Initialize the cumulative image
    global mag_img_colored_cumulative
    if 'mag_img_colored_cumulative' not in globals():
        # Check if magnitude data is grayscale and convert to BGR
        if len(grayscale_image.shape) == 2 or grayscale_image.shape[2] == 1:
            # Normalize magnitude_data to the range 0-255 and convert to uint8
            magnitude_data_normalized = cv2.normalize(grayscale_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            mag_img_colored_cumulative = cv2.cvtColor(magnitude_data_normalized, cv2.COLOR_GRAY2BGR)
        else:
            mag_img_colored_cumulative = grayscale_image.copy()

    # Display RGB Color Map
    ax1.imshow(cv2.cvtColor(mag_img_colored_cumulative, cv2.COLOR_BGR2RGB), cmap='gray', origin='upper')  # Display as grayscale

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0, img_width)  # Ensure x-axis matches image width
    ax1.set_ylim(img_height, 0)  # Ensure y-axis matches image height and is in the top left

    # Display Strain
    ax2.imshow(cv2.cvtColor(fiber_img, cv2.COLOR_BGR2RGB), origin='upper')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Display the Fiber
    ax3.imshow(rgb_image, origin='upper')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlim(0, img_width)  # Ensure x-axis matches image width
    ax3.set_ylim(img_height, 0)  # Ensure y-axis matches image height and is in the top left

    def overlay_region():
        nonlocal ax1, ax3, fig
        global contour_points, grayscale_image, rgb_image, mag_img_colored_cumulative

        if len(contour_points) < 3:
            print("Not enough points to create a contour")
            return

        # 1. Transform contour points to pixel coordinates:
        transformed_contour_points = []
        for x_data, y_data in contour_points:
            x_pixel = int(round(x_data))
            y_pixel = int(round(y_data))
            transformed_contour_points.append((x_pixel, y_pixel))

        # 2. Create the contour array:
        contour_array = np.array(transformed_contour_points, dtype=np.int32).reshape((-1, 1, 2))

        # 3. Create the mask:
        mask = np.zeros(grayscale_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour_array], 255)
        mask_bool = mask > 0

        # 4. Convert rgb_image to uint8 *FIRST*, then convert to RGB (if it's BGR):
        rgb_image_uint8 = cv2.convertScaleAbs(rgb_image, alpha=(255.0))
        rgb_image_rgb = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_BGR2RGB)

        # 5. Debugging Checks:
        print(f"mag_img_colored_cumulative shape: {mag_img_colored_cumulative.shape}, dtype: {mag_img_colored_cumulative.dtype}")
        print(f"rgb_image_rgb shape: {rgb_image_rgb.shape}, dtype: {rgb_image_rgb.dtype}")
        print(f"mask_bool shape: {mask_bool.shape}, dtype: {mask_bool.dtype}, True count: {np.sum(mask_bool)}")

        # 6. Copy from rgb_image to mag_img_colored_cumulative:
        try:
            mag_img_colored_cumulative[mask_bool] = rgb_image_rgb[mask_bool]
        except Exception as e:
            print(f"Error during copy: {e}")
            # Additional debugging: check min/max values in the region
            print(f"rgb_image_rgb min/max in region: {np.min(rgb_image_rgb[mask_bool])}, {np.max(rgb_image_rgb[mask_bool])}")
            print(f"mag_img_colored_cumulative min/max in region: {np.min(mag_img_colored_cumulative[mask_bool])}, {np.max(mag_img_colored_cumulative[mask_bool])}")

        # 7. Update the display:
        ax1.imshow(cv2.cvtColor(mag_img_colored_cumulative, cv2.COLOR_BGR2RGB), origin='upper')
        fig.canvas.draw_idle()


    def draw_contour(event):
        nonlocal ax1, ax2, ax3, fig
        global contour_points, lines, line1, line2, line3, lines_ax2

        if event.button == 1:  # Left mouse button
            if event.name == 'button_press_event':
                contour_points = [(event.xdata, event.ydata)]
                print("Starting new contour")
            elif event.name == 'motion_notify_event':
                if event.inaxes == ax1:
                    contour_points.append((event.xdata, event.ydata))
                    print(f"Adding point: {event.xdata}, {event.ydata}")
                    if len(contour_points) > 1:
                        x, y = zip(*contour_points)
                        line1, = ax1.plot(x, y, color='white', linewidth=1)
                        line3, = ax3.plot(x, y, color='white', linewidth=1)
                        lines.append((line1, line3)) # Store all three lines in lines list
                        fig.canvas.draw_idle()
                elif event.inaxes == ax2:
                    contour_points.append((event.xdata, event.ydata))
                    print(f"Adding point to ax2: {event.xdata}, {event.ydata}")
                    if len(contour_points) > 1:
                        x, y = zip(*contour_points)
                        line2, = ax2.plot(x, y, color='white', linewidth=1)  # Draw independent line on ax2
                        lines_ax2.append(line2)  # Store the line in the separate list
                        fig.canvas.draw_idle()

            elif event.name == 'button_release_event':
                print("Finishing contour")
                if event.inaxes == ax1:
                    overlay_region()  # Call overlay function here only for ax1
                contour_points = []  # Reset for next line


    # Connect the mouse events
    fig.canvas.mpl_connect('button_press_event', draw_contour)
    fig.canvas.mpl_connect('motion_notify_event', draw_contour)
    fig.canvas.mpl_connect('button_release_event', draw_contour)

    # Set black background for first ax
    ax1.set_facecolor("black")

    # Adjust layout
    plt.tight_layout()


    plt.show()

visualize_strain_data(L_Vector, m_data, eigen_index=0, z_slice=z_slice , frame = frame)