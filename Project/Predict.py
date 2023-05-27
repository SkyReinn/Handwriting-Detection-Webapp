import easyocr
import TextSequencing

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the scanned image
image = 'test3.png'

# Perform OCR on the image
predictions = reader.readtext(image, paragraph=True)
result = ""

# Turn the result into a single list
for prediction in predictions:
    result += prediction[1]
    result += " "

print(result)
result = TextSequencing.Normalize(result)
print(result)
