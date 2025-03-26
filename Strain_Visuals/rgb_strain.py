import numpy as np
import matplotlib.pyplot as plt
import scipy.io 
import os
import cv2
import cv2.ximgproc as ximgproc
from scipy.signal import medfilt2d
import imageio
from matplotlib.widgets import Button

"""This Python Script plots the RGB image of the strain data
    Optional at the end to display the data. Just uncomment the line showing the plot below"""

#get current working directory
path = os.getcwd()

#checking if the mat file loads
mat = scipy.io.loadmat(path + '\Strain_Visuals\data_BC.mat')

#acessing the varibales we need
L_Vector = mat['strain_data'] #shape 160 80 22 24 3 3
m_data = mat['m_data'] # shape 160 80 24 24
m_data = m_data[: , :, :-2, :] #to get into the correct shape


"""This works"""

# def visualize_strain_data(eigenVectors, m_data, eigen_index, z_slice, frame, threshold=0.03):
   
#     # Prepare the mask
#     mask = np.squeeze(m_data[:, :, z_slice, frame]) <= threshold

#     magnitude_data = np.squeeze(m_data[:, :, z_slice, frame])

#     # denoised_magnitude = medfilt2d(magnitude_data, kernel_size=1)

    
#     # Extract and process the eigenVector data for color map
#     data_slice = np.abs(np.squeeze(eigenVectors[:, :, z_slice, frame, eigen_index, :]))

#     # Apply mask
#     data_slice[np.repeat(mask[:, :, np.newaxis], 3, axis=2)] = np.nan

#     # Normalize and convert to uint8
#     normalized_data = np.zeros_like(data_slice)
#     for i in range(3):
#         channel_data = data_slice[:,:,i]
#         min_val = np.nanmin(channel_data)
#         max_val = np.nanmax(channel_data)
#         normalized_data[:,:,i] = ((channel_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

#     # Ensure the image is in the correct format (uint8, 3 channels)
#     normalized_data = cv2.convertScaleAbs(normalized_data)
    
#     # Apply anisotropic diffusion filter
#     filtered_data = ximgproc.anisotropicDiffusion(
#         normalized_data,
#         alpha=1, K=0.03, niters=8
#     )

#     # Convert back to float and rescale to [0, 1]
#     rgb_image = filtered_data.astype(float) / 255.0

#     # Create figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

#     # Display RGB Color Map
#     im1 = ax1.imshow(rgb_image, interpolation='bilinear')
#     plt.colorbar(im1, ax=ax1)
#     ax1.set_title('RGB Image')
#     ax1.set_xlabel('X')
#     ax1.set_ylabel('Y')

#     # Display Magnitude
#     im2 = ax2.imshow(magnitude_data, cmap='gray')
#     plt.colorbar(im2, ax=ax2)
#     ax2.set_title('Magnitude Image')
#     ax2.set_xlabel('X')
#     ax2.set_ylabel('Y')
    

#     # Add overall title
#     plt.suptitle(f'Frame: {frame+1}, Z-Slice: {z_slice+1}, Eigenvector: {eigen_index+1}')

#     # Adjust layout
#     plt.tight_layout()

#     return plt.show()


# Global variable to store contour points
contour_points = []
lines = []



def save_figure():
    plt.savefig('contour_plot.png', dpi=300, bbox_inches='tight')
    print("Figure saved as contour_plot.png")



def visualize_strain_data(eigenVectors, m_data, eigen_index, z_slice, frame, threshold=0.03):
    global contour_points, line1, line2
    
     # Prepare the mask
    mask = np.squeeze(m_data[:, :, z_slice, frame]) <= threshold

    magnitude_data = np.squeeze(m_data[:, :, z_slice, frame])
    
    # Extract and process the eigenVector data for color map
    data_slice = np.abs(np.squeeze(eigenVectors[:, :, z_slice, frame, eigen_index, :]))

    # Apply mask
    data_slice[np.repeat(mask[:, :, np.newaxis], 3, axis=2)] = np.nan

    # Normalize and convert to uint8
    normalized_data = np.zeros_like(data_slice)
    for i in range(3):
        channel_data = data_slice[:,:,i]
        min_val = np.nanmin(channel_data)
        max_val = np.nanmax(channel_data)
        normalized_data[:,:,i] = ((channel_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Ensure the image is in the correct format (uint8, 3 channels)
    normalized_data = cv2.convertScaleAbs(normalized_data)
    
    # Apply anisotropic diffusion filter
    filtered_data = ximgproc.anisotropicDiffusion(
        normalized_data,
        alpha=1, K=0.03, niters=8
    )

    # Convert back to float and rescale to [0, 1]
    rgb_image = filtered_data.astype(float) / 255.0


    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Display RGB Color Map
    im1 = ax1.imshow(rgb_image, interpolation='bilinear')
    # plt.colorbar(im1, ax=ax1)
    ax1.set_title('RGB Image')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Display Magnitude
    im2 = ax2.imshow(magnitude_data, cmap='gray')
    # plt.colorbar(im2, ax=ax2)
    ax2.set_title('Magnitude Image')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')



    def draw_contour(event):
        global contour_points, lines
        
        if event.inaxes != ax2:
            return

        if event.button == 1:  # Left mouse button
            if event.name == 'button_press_event':
                contour_points = [(event.xdata, event.ydata)]
                print("Starting new contour")
            elif event.name == 'motion_notify_event':
                contour_points.append((event.xdata, event.ydata))
                print(f"Adding point: {event.xdata}, {event.ydata}")
                if len(contour_points) > 1:
                    line2, = ax2.plot(*zip(*contour_points), color='white', linewidth=3)
                    line1, = ax1.plot(*zip(*contour_points), color='white', linewidth=3)
                    lines.append((line1, line2))
                    fig.canvas.draw_idle()
            elif event.name == 'button_release_event':
                print("Finishing contour")
                contour_points = []  # Reset for next line

    # Connect the mouse events
    fig.canvas.mpl_connect('button_press_event', draw_contour)
    fig.canvas.mpl_connect('motion_notify_event', draw_contour)
    fig.canvas.mpl_connect('button_release_event', draw_contour)

    # Add a button to trigger saving
    save_button = plt.Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Save')
    save_button.on_clicked(lambda event: save_figure())

    # Set black background for first ax
    ax1.set_facecolor("black")

    # Add overall title
    plt.suptitle(f'Frame: {frame+1}, Z-Slice: {z_slice+1}, Eigenvector: {eigen_index+1}')

    # Adjust layout
    plt.tight_layout()

    plt.show()



visualize_strain_data(L_Vector, m_data, eigen_index=0, z_slice=11 , frame  = 11)

# def visualize_strain_data(eigenVectors, m_data, eigen_index, z_slice, threshold=0.03):
#     frames = []
#     for frame in range(eigenVectors.shape[3]):  # Iterate through all frames
#         # Prepare the mask
#         mask = np.squeeze(m_data[:, :, z_slice, frame]) <= threshold

#         magnitude_data = np.squeeze(m_data[:, :, z_slice, frame])

#         # denoised_magnitude = medfilt2d(magnitude_data, kernel_size=1)

       
#         # Extract and process the eigenVector data for color map
#         data_slice = np.abs(np.squeeze(eigenVectors[:, :, z_slice, frame, eigen_index, :]))

#         # Apply mask
#         data_slice[np.repeat(mask[:, :, np.newaxis], 3, axis=2)] = np.nan

#         # Normalize and convert to uint8
#         normalized_data = np.zeros_like(data_slice)
#         for i in range(3):
#             channel_data = data_slice[:,:,i]
#             min_val = np.nanmin(channel_data)
#             max_val = np.nanmax(channel_data)
#             normalized_data[:,:,i] = ((channel_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

#         # Ensure the image is in the correct format (uint8, 3 channels)
#         normalized_data = cv2.convertScaleAbs(normalized_data)
        
#         # Apply anisotropic diffusion filter
#         filtered_data = ximgproc.anisotropicDiffusion(
#             normalized_data,
#             alpha=1, K=0.03, niters=8
#         )

#         # Convert back to float and rescale to [0, 1]
#         rgb_image = filtered_data.astype(float) / 255.0

#         # Create figure with two subplots
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

#         # Display RGB Color Map
#         im1 = ax1.imshow(rgb_image, interpolation='bilinear')
#         plt.colorbar(im1, ax=ax1)
#         ax1.set_title('RGB Image')
#         ax1.set_xlabel('X')
#         ax1.set_ylabel('Y')

#         # Display Magnitude
#         im2 = ax2.imshow(magnitude_data, cmap='gray')
#         plt.colorbar(im2, ax=ax2)
#         ax2.set_title('Magnitude Image')
#         ax2.set_xlabel('X')
#         ax2.set_ylabel('Y')

#         # Set black background for first ax
#         ax1.set_facecolor("black")

#         # Add overall title
#         plt.suptitle(f'Frame: {frame+1}, Z-Slice: {z_slice+1}, Eigenvector: {eigen_index+1}')

#         # Adjust layout
#         plt.tight_layout()

#         # Save the figure to a BytesIO object
#         from io import BytesIO
#         buf = BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
        
#         # Read the image from the BytesIO object
#         img = imageio.imread(buf)
#         frames.append(img)

#         # Close the figure to free up memory
#         plt.close(fig)

#     # Save the frames as a GIF
#     exportname = f'slice_{z_slice+1}_strain_eigenvector_{eigen_index+1}_CMAP.gif'

#     # Save the frames as a GIF
#     imageio.mimsave(exportname, frames, duration=0.5, loop = 0)  # 0.5 seconds per frame

#     print(f"GIF saved as '{exportname}'")

#     return frames


""" For Multiple Frames"""
# for frame_num in range(m_data.shape[2]):

#     visualize_strain_data(L_Vector, m_data, eigen_index=2, z_slice=frame_num)

