import helpful_functions as hf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
import os
import cv2
import random
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

weeb_dir = 'prediction model/moeimouto-faces/'

(X_train, Y_train, label_train), (X_test,
                                  Y_test, label_test) = hf.test_train_split()


model = Sequential()
model.add(Conv2D(32, (3, 3),
                 input_shape=(160, 160, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=173, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
                 156, 10, validation_split=0.3)
