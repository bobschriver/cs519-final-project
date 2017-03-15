'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function

# Weird Hack
import pydot
pydot.find_graphviz = lambda: True

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class CropCenter(Layer):

	def __init__(self, **kwargs):
		super(CropCenter, self).__init__(**kwargs)
	
	def build(self, input_shape):
		print('Building Crop Center')
		print(input_shape)
		self.crop_left = input_shape[1] / 4
		self.crop_right = (input_shape[1] / 4) * 3
		self.crop_top = input_shape[2] / 4
		self.crop_bottom = (input_shape[2] / 4) * 3
		super(CropCenter, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, inputs):
		return inputs[:, self.crop_left:self.crop_right, self.crop_top:self.crop_bottom, :]

	def compute_output_shape(self, input_shape):
		print('Computing Output Shape')
		print(input_shape)
		return (input_shape[0], input_shape[1] / 2, input_shape[2] / 2, input_shape[3])

batch_size = 32
nb_classes = 10
nb_epoch = 10
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(CropCenter())

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(CropCenter())

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(CropCenter())

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)

