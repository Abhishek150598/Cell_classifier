from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.layers import MaxPooling2D

size = 50

class CnnModel:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        
        model.add(Conv2D(size,(3,3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (3, 3)))
        
        model.add(Conv2D(size,(3,3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (3, 3)))
        
        model.add(Conv2D(size,(3,3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (3, 3)))
        model.add(Dropout(0.5))
        
        model.add(Conv2D(size,(3,3), padding = "same" , input_shape = inputShape))
        model.add(Activation("relu"))
        
        model.add(Flatten())
        model.add(Activation("relu"))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model





