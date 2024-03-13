import numpy as np
import cv2
import os

def split_merge_segmentation(image, threshold):
    print(image.shape)
    if image.shape[0] == 0 or image.shape[1] == 0:
        return image
    segmented_image = np.zeros_like(image)
    if np.max(image) - np.min(image) > threshold:
        height, width = image.shape
        half_height = height // 2
        half_width = width // 2
        
        segmented_image[0:half_height, 0:half_width] = split_merge_segmentation(image[0:half_height, 0:half_width], threshold)
        segmented_image[0:half_height, half_width:] = split_merge_segmentation(image[0:half_height, half_width:], threshold)
        segmented_image[half_height:, 0:half_width] = split_merge_segmentation(image[half_height:, 0:half_width], threshold)
        segmented_image[half_height:, half_width:] = split_merge_segmentation(image[half_height:, half_width:], threshold)
    else:
        if np.max(image) >= 127:
            segmented_image[:, :] = np.mean(image)
        else:
            segmented_image[:, :] = 0
            
    return segmented_image


image_path = './lena.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold = float(input('Enter threshold value : '))
segmented_image = split_merge_segmentation(gray_image, threshold)

padded_image = cv2.copyMakeBorder(segmented_image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
averaged_image = np.zeros_like(segmented_image)

window = 3
pad_size = window // 2

for i in range(segmented_image.shape[0]):
    for j in range(segmented_image.shape[1]):
        neighborhood = padded_image[i:i+window, j:j+window]
        average_value = np.mean(neighborhood)
        averaged_image[i, j] = average_value


output_folder = './output'
output_image_path = os.path.join(output_folder,'Segment.jpg')
cv2.imwrite(output_image_path, segmented_image)
   
cv2.imshow('Original Image', image)

cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
