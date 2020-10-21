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

# Load the model
model = keras.models.load_model('"/floyd/input/model/model"')

directory = "/floyd/input/opportunity"
super_folder = os.listdir(directory)

max_count = len(super_folder)
current_count = 0

if not os.path.exists(f"output"):
    os.mkdir(f"output")

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

    if not os.path.exists(f"output/{folder}"):
        os.mkdir(f"output/{folder}")

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((64, 64, 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        filename = filenames[i]
        imsave(f"output/{folder}/{filename}", lab2rgb(cur))
    
    del color_me

    current_count = index
    percentage = (current_count / max_count) * 100
    print(f"Completion: {percentage}%, {current_count} / {max_count}")