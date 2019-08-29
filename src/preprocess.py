from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd

# Loading the training labels
df = pd.read_csv("../data/train.csv")

images = []
labels = []
no_of_images = 10600
count = 0

# Loading the training images
for i in range(1, no_of_images + 1):
    try:
      image_name = '../data/train_images/f' + str(i) + '.png'
      image = load_img(image_name, target_size = (224,224), color_mode = "grayscale")
      image = img_to_array(image)

      images.append(image)
      labels.append(df['label'][i-1])
      print(i)
      count += 1
    except:
      pass
      
print("Count: ", count)

np_images = np.array(images)
np_labels = np.array(labels)

# Saving the training images and labels
np.save('../data/images', np_images)
np.save('../data/labels', np_labels)