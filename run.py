
from tensorflow.keras.models import Sequential, save_model, load_model
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf

# Load the model
model = load_model(
    "data/model/model.h5",
    custom_objects=None,
    compile=True
)

directory = os.fsencode("data/opportunity/grey/slice").decode('ascii')
super_folder = os.listdir(directory)

if not os.path.exists(f"data/opportunity/color"):
    os.mkdir(f"data/opportunity/color")

for index in range(len(super_folder)):
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