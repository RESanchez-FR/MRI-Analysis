import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Initialize global variables
contour_points = []

def visualize_data(mag_img, fiber_img):
    global contour_points

    # Resize images to ensure they have the same dimensions
    fiber_img = cv2.resize(fiber_img, (mag_img.shape[1], mag_img.shape[0]))

    # Normalize grayscale image (mag_img) to range 0-255 and convert to uint8
    mag_img_normalized = cv2.normalize(mag_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert grayscale magnitude image to BGR for consistency
    mag_img_colored_cumulative = cv2.cvtColor(mag_img_normalized, cv2.COLOR_GRAY2BGR)

    # Ensure fiber image is uint8 (if it isn't already)
    if fiber_img.dtype != np.uint8:
        fiber_img = cv2.convertScaleAbs(fiber_img)

    # Get image dimensions
    img_height, img_width = mag_img.shape[:2]  # Use mag_img's dimensions

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Reduced to two axes
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    ax1.grid(False)
    ax2.grid(False)

    # Display initial images
    ax1.imshow(cv2.cvtColor(mag_img_colored_cumulative, cv2.COLOR_BGR2RGB), origin='upper')
    ax2.imshow(cv2.cvtColor(fiber_img, cv2.COLOR_BGR2RGB), origin='upper')

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0, img_width)
    ax1.set_ylim(img_height, 0)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(0, img_width)
    ax2.set_ylim(img_height, 0)

    def overlay_region():
        nonlocal mag_img_colored_cumulative

        if len(contour_points) < 3:
            print("Not enough points to create a contour")
            return

        # Transform contour points to pixel coordinates:
        transformed_contour_points = []
        for x_data, y_data in contour_points:
            x_pixel = int(round(x_data))
            y_pixel = int(round(y_data))
            transformed_contour_points.append((x_pixel, y_pixel))

        # Create the contour array:
        contour_array = np.array(transformed_contour_points, dtype=np.int32).reshape((-1, 1, 2))

        # Create the mask:
        mask = np.zeros(mag_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour_array], 255)
        mask_bool = mask > 0

        # Cut and paste: Copy region from fiber_img to mag_img_colored_cumulative
        try:
            mag_img_colored_cumulative[mask_bool] = fiber_img[mask_bool]
        except Exception as e:
            print(f"Error during copy: {e}")

        # Update the display:
        ax1.imshow(cv2.cvtColor(mag_img_colored_cumulative, cv2.COLOR_BGR2RGB), origin='upper')
        fig.canvas.draw_idle()

    def draw_contour(event):
        nonlocal ax1, ax2, fig  # Include ax2 for mirroring

        if event.button == 1:  # Left mouse button
            if event.name == 'button_press_event':
                contour_points.clear()
                contour_points.append((event.xdata, event.ydata))
                print("Starting new contour")
            elif event.name == 'motion_notify_event':
                if event.inaxes == ax1:
                    contour_points.append((event.xdata, event.ydata))
                    print(f"Adding point: {event.xdata}, {event.ydata}")

                    # Extract x and y coordinates
                    x, y = zip(*contour_points)

                    # Plot on ax1
                    ax1.plot(x, y, color='white', linewidth=1)

                    # Mirror the line on ax2
                    ax2.plot(x, y, color='white', linewidth=1)

                    # Update the figure
                    fig.canvas.draw_idle()
            elif event.name == 'button_release_event':
                print("Finishing contour")
                overlay_region()  # Call overlay function
                
    # Connect the drawing function to the first subplot (ax1)
    fig.canvas.mpl_connect('button_press_event', draw_contour)
    fig.canvas.mpl_connect('motion_notify_event', draw_contour)
    fig.canvas.mpl_connect('button_release_event', draw_contour)

    plt.show()



# Example usage:

# Get current working directory
path = os.getcwd()

img_path = path + '\Images\Computing_Image'

mag_img_path = os.path.join(img_path, 'MAG_AL_Slice18.png')  # Path for magnitude image
fiber_img_path = os.path.join(img_path, 'DTI_AL_Slice18.png')  # Path for fiber image

# Load images using OpenCV
mag_img = cv2.imread(mag_img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
fiber_img = cv2.imread(fiber_img_path)  # Load as RGB (default BGR in OpenCV)

# visualize_data(mag_img=mag_img, fiber_img=fiber_img)

# Function to enhance brightness and vibrancy
def enhance_brightness_vibrancy(image, alpha, beta):
    """
    Enhance the brightness and vibrancy of an image.

    Parameters:
        image (numpy.ndarray): Input image (BGR format).
        alpha (float): Contrast control (1.0-3.0).
        beta (int): Brightness control (0-100).

    Returns:
        numpy.ndarray: Enhanced image.
    """
    # Convert to HSV to adjust saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Increase saturation
    s = cv2.add(s, 50)  # Increase saturation by 50 (adjust as needed)
    s = np.clip(s, 0, 255)  # Clip values to valid range

    # Merge back and convert to BGR
    hsv_enhanced = cv2.merge([h, s, v])
    image_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    # Apply brightness and contrast adjustment
    image_enhanced = cv2.convertScaleAbs(image_enhanced, alpha=alpha, beta=beta)

    return image_enhanced

def enhance_grayscale(image, alpha=1.0, beta=0):
    """
    Enhance the brightness and contrast of a grayscale image.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        alpha (float): Contrast control (1.0-3.0).
        beta (int): Brightness control (0-100).

    Returns:
        numpy.ndarray: Enhanced grayscale image.
    """
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply brightness and contrast adjustment
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return enhanced_image

enhanced_fiber = enhance_brightness_vibrancy(fiber_img, alpha = 3, beta = 30)

enhanced_mag = enhance_grayscale(mag_img, alpha = 3, beta = 30)

visualize_data(mag_img=enhanced_mag, fiber_img=enhanced_fiber)
# Example usage
# # Replace 'fiber_img' with your actual image variable
# if 'fiber_img' in globals():
#     enhanced_image = enhance_brightness_vibrancy(fiber_img)

#     # Display the original and enhanced images
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(cv2.cvtColor(fiber_img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title("Enhanced Image")
#     plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')

#     plt.show()
# else:
#     print("Please ensure 'fiber_img' is loaded as an image.")
