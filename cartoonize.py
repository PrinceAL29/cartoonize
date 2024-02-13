import cv2
import numpy as np

# Read the image
image = cv2.imread("image.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply median blur
gray_blur = cv2.medianBlur(gray, 5)

# Use adaptive thresholding to create an edge mask
edges = cv2.adaptiveThreshold(gray_blur, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 9, 9)

# Convert back to color so it can be bit-ANDed with color image
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Reduce the number of colors
# Convert to the RGB color space
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Use k-means clustering to reduce colors
data = np.float32(img_rgb).reshape((-1, 3))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
ret, label, center = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
result = center[label.flatten()]
result_image = result.reshape(img_rgb.shape)

# Convert reduced color image back to BGR color space for OpenCV
result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

# Combine edges and color
cartoon = cv2.bitwise_and(result_image_bgr, edges_colored)

# Save the cartoonized image
cv2.imwrite("cartoonized_image.jpg", cartoon)

print("The cartoonized image has been saved as 'cartoonized_image.jpg'")
