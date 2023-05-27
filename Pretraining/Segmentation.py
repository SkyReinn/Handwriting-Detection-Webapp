import cv2
import numpy as np

def lineSegmentation(image):
    # Dialate the image to form lines
    kernel = np.ones((3, 77), np.uint8)
    img = cv2.dilate(image, kernel, iterations=1)

    # Find all the contours and sort them in order
    contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sortedContourLines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    return sortedContourLines

def wordSegmentation(image):
    # Dialate the image to form words
    kernel = np.ones((3, 14), np.uint8)
    img = cv2.dilate(image, kernel, iterations=1)

    # Find all the contours and sort them in order
    contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sortedContourWords = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    return sortedContourWords

def characterSegmentation(image):
    # Find the sum of pixels along each column of the image
    sumCols = np.sum(image, axis=0)
    validCols = []

    # Loop through each column's sum of pixels
    for i, colSum in enumerate(sumCols):
        if colSum == 0 and (sumCols[max(0, i-1)] != 0 or sumCols[min(0, i+1)] != 0):
            validCols.append(i)

    return validCols


