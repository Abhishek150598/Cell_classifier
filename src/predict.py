import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

df = pd.read_csv('../data/sample.csv')

# Model reconstruction from JSON file
with open('../data/model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('../data/model_weights.h5')

no_of_images = 15984
images = []

# Adding the images to array
for i in range(1, no_of_images + 1):
    image_name = '../data/test_images/f' + str(i) + '.png'
    image = load_img(image_name, target_size = (224,224), color_mode = "grayscale")
    image = img_to_array(image)
    images.append(image)

images = np.array(images)

# Making predictions on images
predictions = model.predict(images)

# Storing the predictions in dataframe
for i in range(len(predictions)):
    df['label'][i] = np.argmax(predictions[i])

# Generating output file
df.to_csv('../output/output.csv')