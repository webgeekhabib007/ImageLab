import cv2

image_path = '.\images\\lena.jpg'

def image_loader(path=image_path):
    image = cv2.imread(path)
    return image

