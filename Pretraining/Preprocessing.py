import cv2

def preprocessing(image):
    # Uses Otsu's thresholding after Gaussian filtering to convert the image into binary form
    img = cv2.GaussianBlur(image, (1, 1), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove high intensity dots/patches to smoothen the image
    img = cv2.fastNlMeansDenoising(img, None, 15, 7, 21)

    # Resize & Invert Image
    img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
    img = cv2.bitwise_not(img)

    # Use Guo Hall's algorithm to skeletonize the image
    img = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_GUOHALL)

    return img



