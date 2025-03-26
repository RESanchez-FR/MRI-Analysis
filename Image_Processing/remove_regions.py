import cv2
import numpy as np
import os

def draw_contour(event, x, y, flags, param):
    global drawing, ix, iy, img, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.line(mask, (ix, iy), (x, y), 255, 2)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.line(mask, (ix, iy), (x, y), 255, 2)



#get current working directory
path = os.getcwd()
  
# Example usage:
input_image_path = path +  "\Images\Slice_7.png"  # Replace with your image path

# Load the image
img = cv2.imread(input_image_path)
original = img.copy()

# Create a mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_contour)

drawing = False
ix, iy = -1, -1

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC key to exit
        break
    elif k == ord('f'):  # 'f' key to fill the contour
        # Fill the contour
        cv2.fillPoly(mask, [np.array(np.where(mask > 0)[::-1]).T], 255)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(original, original, mask=cv2.bitwise_not(mask))
        
        cv2.imshow('result', result)
        cv2.imwrite('result.png', result)

cv2.destroyAllWindows()
