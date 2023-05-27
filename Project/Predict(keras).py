# Imports
import keras_ocr
import matplotlib.pyplot as plt

# Load images and model
image = keras_ocr.tools.read('test.png')
pipline = keras_ocr.pipeline.Pipeline()

# Predict
prediction = pipline.recognize([image])[0]

# Show predictions
fig, ax = plt.subplots(figsize=(10, 10))
keras_ocr.tools.drawAnnotations(image=image, predictions=prediction, ax=ax)

# Save plots
fig.savefig("result.png")