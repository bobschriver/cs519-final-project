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
import numpy as np
import keras.backend as K


from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D,Convolution2D, MaxPooling2D
from keras.utils import np_utils




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

w = [np.array([[[[1],
		[1],
		[1]],
		[[3],
		[3],
		[3]],
		[[3],
		[3],
		[3]],
		[[1],
		[1],
		[1]]],

		[[[3],
		[3],
		[3]],
		[[9],
		[9],
		[9]],
		[[9],
		[9],
		[9]],
		[[3],
		[3],
		[3]]],

		[[[3],
		[3],
		[3]],
		[[9],
		[9],
		[9]],
		[[9],
		[9],
		[9]],
		[[3],
		[3],
		[3]]],

		[[[1],
		[1],
		[1]],
		[[3],
		[3],
		[3]],
		[[3],
		[3],
		[3]],
		[[1],
		[1],
		[1]]]],dtype='float32')]



model.add(Convolution2D(3, 3, 3, border_mode='same',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

conv = Conv2D(1, 4, strides=[2,2], weights =w, border_mode='same', activation='relu',bias=False,trainable=False)
#conv.set_weights(w)
model.add(conv)
#model.add(MaxPooling2D(pool_size=(2, 2)))

print("Shape of Weights: ", conv.get_weights())

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()
#Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)
