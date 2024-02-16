import cv2
import numpy as np

"""
    Workflow
    1. Adjust the size of the logo if needed
    2. Seperate the color and alpha channels from the logo
    3. Determine the region of interest (ROI) in the image for logo placement
    4. Use logical operations to create ROI and logo masks
    5. Use logical operations to combine the ROIs into a watermarked patch
    6. Insert the watermark patch in the image
"""

# Read the image and watermark logo
img = cv2.imread('leaves.jpg', cv2.IMREAD_UNCHANGED)
logo = cv2.imread('opencv_logo.png', cv2.IMREAD_UNCHANGED)

# Check the dimensions
# print(img.shape)
# print(logo.shape)

# Reduce the size of the logo to 10% of its original dimensions
logo = cv2.resize(logo, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)

# Retrieve the image and logo shapes
img_h, img_w, _ = img.shape
logo_h, logo_w, _ = logo.shape

# Print the shape of the resized logo
# print(logo.shape)

# Seperate the color and alpha channels

logo_bgr = logo[:, :, 0:3]
logo_alpha = logo[:, :, 3]

# cv2.imshow('logo_bgr', logo_bgr)
# cv2.imshow('logo_a', logo_alpha)
# cv2.imshow('img', img)

# Cx and Cy are center of the image

cx = int(img_w / 2)
cy = int(img_h / 2)

# tlc: top left corner
tlc_x = int(cx - logo_w / 2)
tlc_y = int(cy - logo_h / 2)

# brc: bottom right corner
brc_x = int(cx + logo_w / 2)
brc_y = int(cy + logo_h / 2)

# cv2.rectangle(img, (tlc_x, tlc_y), (brc_x, brc_y), (0, 255, 0), 3)

# Get region of interest from the original image
roi = img[tlc_y:brc_y, tlc_x:brc_x]

# Make the dimensions of the mask same as the input image
# Since the background image is a 3-channel image, we create a 3 channel image for the mask

logo_mask = cv2.merge([logo_alpha, logo_alpha, logo_alpha])

logo_mask_inv = cv2.bitwise_not(logo_mask)

# Use the mask to create the masked ROI region
masked_roi = cv2.bitwise_and(roi, logo_mask_inv)

# Used the mask to create the masked logo region
masked_logo = cv2.bitwise_and(logo_bgr, logo_mask)

# Combine the masked ROI with the masked logo to get the combined ROI image
roi_final = cv2.bitwise_or(masked_roi, masked_logo)

img_1 = img.copy()
roi_1 = roi.copy()

# Insert the ROI patch in the image
img_1[tlc_y:brc_y, tlc_x:brc_x] = roi_final

cv2.imwrite('watermarked_method_1.jpg', img_1)

# Adding a semi-transparent watermark
# Use addWeighted() function to blend the logo with the image ROI
roi_2 = roi.copy()

# Blend ROI and the logo
watermarked = cv2.addWeighted(roi_2, 1, logo_bgr, 0.45, 0)

img_2 = img.copy()

img_2[tlc_y:brc_y, tlc_x:brc_x] = watermarked

cv2.imwrite('watermarked_method_2.jpg', img_2)
