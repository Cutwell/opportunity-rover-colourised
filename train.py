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
from tensorflow import keras

# load model
model = keras.models.load_model('/floyd/input/model/model')

# tensorboard tracking
tensorboard = TensorBoard(log_dir="tensorboard")

# Get images
directory = "/floyd/input/curiosity"
super_folder = os.listdir(directory)

max_count = len(super_folder)
current_count = 0

"""
Due to a memory leak in tensorflow, the following loop consumes memory each iteration until an out-of-memory error occurs.
To account for this, we need to train the dataset on 50% of the dataset per job.
"""
for index in range(3500, 5475):
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
    model.fit_generator(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=10, steps_per_epoch=10)

    # Test images
    Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
    Xtest = Xtest.reshape(Xtest.shape+(1,))
    Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
    Ytest = Ytest / 128
    print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

    current_count = index
    percentage = (current_count / max_count) * 100
    print(f"Completion: {percentage}%, {current_count} / {max_count}")

    del X


# Save model
model.save("model")