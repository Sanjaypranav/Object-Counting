# File: opencv.py
# Author: Sanjaypranav 

import cv2 as cv
import matplotlib.pyplot as plt

def read_image(image):
    img = cv.imread(image, cv.IMREAD_COLOR)
    img = cv.resize(img, (200 , 200))
    return img

def Mask(image):
    img = read_image(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    return img

def Segment(image):
    img = read_image(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    img = cv.medianBlur(img, 3)
    img = cv.dilate(img, None, iterations=2)
    img = cv.erode(img, None, iterations=1)
    return img

def Contours(image):
    img = read_image(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    img = cv.medianBlur(img, 3)
    img = cv.dilate(img, None, iterations=2)
    img = cv.erode(img, None, iterations=1)
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

if __name__ == "__main__":
    print("OpenCV Version: ", cv.__version__)
    # print("OpenCV Build: ", cv.getBuildInformation())
    image_path = "data/lemon.jpg"
    # Read Image
    img = read_image(image_path)
    # Mask Image
    img = Mask(image_path)
    plt.imsave("data/mask.jpg", img)
    # Segment Image
    img = Segment(image_path)
    plt.imsave("data/segment.jpg", img)
    
    
    