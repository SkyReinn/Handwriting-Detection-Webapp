import cv2
import numpy as np
import Segmentation
import Preprocessing
from keras.models import load_model

def main():
    # Variables
    image = cv2.imread('test.png')
    words = []
    characters = []
    model = load_model('LetterRecognitionModel.h5')
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    result = ""

    # Preprocess all images
    image = Preprocessing.preprocessing(image)

    # Detect line contours
    contourLines = Segmentation.lineSegmentation(image)

    # Detect word contours and store them in sequence
    for lines in contourLines:
        x, y, w, h = cv2.boundingRect(lines)
        line = image[y:y+h, x:x+w]
        contourWords = Segmentation.wordSegmentation(line)
        for word in contourWords:
            if cv2.contourArea(word) > 700:
                xx, yy, ww, hh = cv2.boundingRect(word)
                words.append((x+xx+1, y+yy+1, x+xx+ww, y+yy+hh))

    # Detect individual characters and store them in sequence
    for x, y, xx, yy in words:
        word = image[y:yy, x:xx]
        cols = Segmentation.characterSegmentation(word)
        if len(cols) > 0:
            for i in range(len(cols)+1):
                if i == 0:
                    characters.append(word[:, :cols[i]])
                elif i == len(cols):
                    characters.append(word[:, cols[i-1]:])
                else:
                    characters.append(word[:, cols[i-1]:cols[i]])

    # Predict each character using the pre-trained CNN model
    for character in characters:

        # Check whether the character is a blank space or not
        if cv2.minMaxLoc(character)[1] == 255:

            # Create a border around the image for better model compatibility
            thickness = 7
            character = cv2.copyMakeBorder(character, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=0)
            
            # make the character thicker
            kernel = np.ones((4, 4), np.uint8)
            character = cv2.dilate(character, kernel, iterations=1)

            # Reseize the character to 28*28
            character = cv2.resize(character, (28, 28), interpolation=cv2.INTER_AREA)

            # Show each character
            cv2.imshow('character', character)
            cv2.waitKey(0)

            # Further expand the image from (28, 28) to (1, 28, 28, 1) 
            character = np.array(character)
            character = np.expand_dims(character, axis=-1)
            character = np.expand_dims(character, axis=0)
            
            # Predict and append the resulting character to result
            prediction = np.argmax(model.predict(character)[0])
            prediction = letters[prediction]
            result += prediction

        else:
            result += " "
    
    print(result)

if __name__ == "__main__":
    main()
