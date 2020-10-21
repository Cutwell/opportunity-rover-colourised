from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import save_model
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf


model = Sequential()
model.add(InputLayer(input_shape=(64, 64, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')

# Get images
directory = os.fsencode("data/curiosity/color/slice").decode('ascii')

super_folder = os.listdir(directory)

for index in range(0, 10):
    folder = super_folder[index]

    X = []

    for filename in os.listdir(f"{directory}/{folder}"):
        img = load_img(f"{directory}/{folder}/{filename}")
        arr = img_to_array(img)
        X.append(arr)


    X = np.array(X, dtype=float)

    # Set up train and test data
    split = int(0.95*len(X))
    Xtrain = X[:split]
    Xtrain = 1.0/255*Xtrain

    # Image transformer
    datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True)

    # Generate training data
    batch_size = 10
    def image_a_b_gen(batch_size):
        for batch in datagen.flow(Xtrain, batch_size=batch_size):
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:,:,:,0]
            Y_batch = lab_batch[:,:,:,1:] / 128
            yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

    # Train model
    model.fit_generator(image_a_b_gen(batch_size), epochs=1, steps_per_epoch=10)

    # Test images
    Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
    Xtest = Xtest.reshape(Xtest.shape+(1,))
    Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
    Ytest = Ytest / 128
    print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

    del X

directory = os.fsencode("data/opportunity/grey/slice").decode('ascii')
super_folder = os.listdir(directory)

if not os.path.exists(f"data/opportunity/color"):
    os.mkdir(f"data/opportunity/color")

for index in range(0, 1):
    folder = super_folder[index]

    color_me = []
    # save file names in load order, for later identification
    filenames = []

    for file in os.listdir(f"{directory}/{folder}"):
        img = load_img(f"{directory}/{folder}/{file}")
        arr = img_to_array(img)
        color_me.append(arr)
        filenames.append(file)

    color_me = np.array(color_me, dtype=float)
    color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))

    # Test model
    output = model.predict(color_me)
    output = output * 128

    if not os.path.exists(f"data/opportunity/color/{folder}"):
        os.mkdir(f"data/opportunity/color/{folder}")

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((64, 64, 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        filename = filenames[i]
        imsave(f"data/opportunity/color/{folder}/{filename}", lab2rgb(cur))
    
    del color_me

