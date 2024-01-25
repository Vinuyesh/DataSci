#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

# Load the image
image = cv2.imread('led.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to separate LEDs from the background
_, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to store LED information
led_info = []

# Iterate through the detected contours
for i, contour in enumerate(contours):
    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # Filter out small noise by setting a minimum area threshold
    if area > 100:
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Mark the LED by drawing a circle around it
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # Store LED information
        led_info.append((cX, cY, area))

# Save the number of LEDs, centroid coordinates, and area to a text file
with open('led_info.txt', 'w') as file:
    file.write(f'Number of LEDs detected: {len(led_info)}\n\n')
    for i, (cX, cY, area) in enumerate(led_info):
        file.write(f'LED {i + 1}: (X={cX}, Y={cY}), Area={area}\n')

# Save the processed image with LEDs marked
cv2.imwrite('marked_image.png', image)

