###Unsharp masking
import cv2

# Step 1: Read the Image
image = cv2.imread('phd/dev/sambody/tstimgs/body_12.png')
if image is None:
    raise ValueError("Could not open or find the image")

# Step 2: Blur the Image
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Step 3: Create the Mask
mask = cv2.subtract(image, blurred_image)

# Step 4: Apply the Mask
sharpened_image = cv2.addWeighted(image, 1.5, mask, -0.5, 0)

# Step 5: Save or Display the Result
cv2.imwrite('OriginalImage.png', image)
cv2.imwrite('SharpenedImage.png', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

###Laplacian filters
import cv2
import numpy as np

# Step 1: Read the Image
image = cv2.imread('phd/dev/sambody/tstimgs/body_12.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Could not open or find the image")

# Step 2 (optional): Convert to Grayscale
# If your image is not already in grayscale, convert it
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply the Laplacian Filter
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Step 4: Convert Back to Uint8
laplacian_abs = np.absolute(laplacian)
laplacian_scaled = np.uint8((laplacian_abs / laplacian_abs.max()) * 255)

# Step 5: Enhance the Image
sharpened_image = cv2.addWeighted(image, 1.5, laplacian_scaled, -0.5, 0)

# Saving the images
cv2.imwrite('original_image.jpg', image)
cv2.imwrite('laplacian.jpg', laplacian)
cv2.imwrite('sharpened_image.jpg', sharpened_image)

###High-pass filters
import cv2

# Step 1: Read the Image
image = cv2.imread('phd/dev/sambody/tstimgs/body_12.png')
if image is None:
    raise ValueError("Could not open or find the image")

# Step 2 (optional): Convert to Grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Low-Pass (Blur) Filter
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Step 4: Create High-Pass Filter
high_pass = cv2.subtract(image, blurred_image)

# Step 5: Combine with Original Image
high_pass_scaled = cv2.multiply(high_pass, 2.0)
sharpened_image = cv2.add(image, high_pass_scaled)

# Step 6: Save or Display the Result
cv2.imshow('Original Image', image)
cv2.imshow('High-Pass Filtered Image', high_pass)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saving the images
cv2.imwrite('original_image.jpg', image)
cv2.imwrite('high_pass_filtered_image.jpg', high_pass)
cv2.imwrite('sharpened_image.jpg', sharpened_image)


###Edge enhancement
from PIL import Image, ImageFilter

# Step 1: Open the Image
image = Image.open('phd/dev/sambody/tstimgs/body_12.png')
if image is None:
    raise ValueError("Could not open or find the image")
if image.mode != 'RGB':
    image = image.convert('RGB')

# Step 2: Apply Edge Enhancement
# For a subtle effect, use ImageFilter.EDGE_ENHANCE
# For a more pronounced effect, use ImageFilter.EDGE_ENHANCE_MORE
enhanced_image = image.filter(ImageFilter.EDGE_ENHANCE)

# Save the enhanced image
enhanced_image.save('enhanced_image.jpg')

###Wavelet transforms