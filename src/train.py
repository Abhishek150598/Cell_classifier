import numpy as np
from CNNmodel import CnnModel
from sklearn.model_selection import train_test_split

images = np.load('../data/images.npy')
labels = np.load('../data/labels.npy')

(train_images, test_images, train_labels, test_labels) = train_test_split(images, labels, test_size=0.02)

# Build the model
model = CnnModel.build(width = 224, height = 224, depth = 1, classes = 15)
# Compile the model
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
# Fit the model with training images and labels
model.fit(train_images, train_labels, epochs = 10)
# Save the weights
model.save_weights('../data/model_weights.h5')
# Save the model architecture
with open('../data/model_architecture.json', 'w') as f:
	f.write(model.to_json())
# Calculate test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
# Print test accuracy
print("Test accuracy: ", test_acc)