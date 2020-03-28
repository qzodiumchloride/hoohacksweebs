# code borrowed from the book Neural Network Projects with Python by James Loy


'''
Main code for training a Siamese neural network for face recognition
'''
import helpful_functions
import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

weeb_dir = 'prediction model/moeimouto-faces'

# Import Training and Testing Data
(X_train, Y_train), (X_test, Y_test) = helpful_functions.get_data(weeb_dir)
num_classes = len(np.unique(Y_train))

# Create Siamese Neural Network
input_shape = 32
shared_network = helpful_functions.create_shared_network(input_shape)
input_top = Input(shape=input_shape)
input_bottom = Input(shape=input_shape)
output_top = shared_network(input_top)
output_bottom = shared_network(input_bottom)
distance = Lambda(helpful_functions.euclidean_distance, output_shape=(1,))(
    [output_top, output_bottom])
model = Model(inputs=[input_top, input_bottom], outputs=distance)

# Train the model
training_pairs, training_labels = helpful_functions.create_pairs(
    X_train, Y_train, num_classes=num_classes)
model.compile(loss=helpful_functions.contrastive_loss,
              optimizer='adam', metrics=[helpful_functions.accuracy])
model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels,
          batch_size=128,
          epochs=10)

# Save the model
model.save('siamese_nn.h5')
