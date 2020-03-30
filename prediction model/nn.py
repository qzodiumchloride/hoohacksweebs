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


FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE = 32
MAXPOOL_SIZE = 2
BATCH_SIZE = 250
EPOCHS = 10

model = Sequential()
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE),
                 input_shape=(INPUT_SIZE, INPUT_SIZE, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Conv2D(NUM_FILTERS*2, (FILTER_SIZE, FILTER_SIZE), activation='relu'))
model.add(MaxPooling2D(pool_size=(MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Flatten())
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=173, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

try:
    hist = model.fit(X_train, Y_train,
                     BATCH_SIZE, EPOCHS, validation_split=0.3)
except:
    pass


model.evaluate(X_test, Y_test)

probabilities = model.predict(np.array(X_test[0]))
