import cv2
import os 

def resize_image(image_path, output_width, output_height):
  """
  Loads an image from a file, resizes it to the specified dimensions,
  and saves the resized image.

  Args:
    image_path: The path to the input image file (e.g., "cropped_image.png").
    output_width: The desired width of the resized image (e.g., 80).
    output_height: The desired height of the resized image (e.g., 160).

  Returns:
    A numpy array representing the resized image.
  """
  try:
    # Load the image using cv2
    img = cv2.imread(image_path)

    if img is None:
      raise ValueError(f"Could not open or find the image at {image_path}")

    # Resize the image
    resized_img = cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_LINEAR) #can also change interpolations methods depending on your use case.
    return resized_img

  except Exception as e:
    print(f"An error occurred: {e}")
    return None


#get current working directory
path = os.getcwd()
  
# Example usage:
input_image_path = path +  "\\Images\\Computing_Image\\MAG_Slice18.png"  # Replace with your image path
output_width = 80
output_height = 160
output_image_path = "MAG_AL_Slice18.png"  # Optional: save the resized image

resized_image = resize_image(input_image_path, output_width, output_height)

if resized_image is not None:
  print(f"Image resized to {resized_image.shape}")
   # Optionally, save the resized image to a file:
  cv2.imwrite(output_image_path, resized_image)
  print(f"Resized image saved to {output_image_path}")

#   # Optionally, display the resized image using matplotlib
#   import matplotlib.pyplot as plt
#   plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)) #Correct the color using cv2's colorspace convert since cv2 reads it as BGR
#   plt.title("Resized Image")
#   plt.axis('off')
#   plt.show()
else:
  print("Image resizing failed.")