"""
Filter images in "/colourImages" based on file size.
Also scales images to 1024x1024 for uniformality.
"""

import os
from scipy.misc import imread, imsave, imresize
from shutil import copyfile

import PIL
from PIL import Image

from imutils import paths
import cv2

if not os.path.exists("filteredImages"):
    os.mkdir("filteredImages")

directory = os.fsencode("colourImages").decode('ascii')

# threshold for acceptable image blur (above threshold = passing image)
threshold = 60

for file in os.listdir(directory):

    filename = os.fsdecode(file)

    if filename.endswith(".jpg"):

        path = os.path.join(directory, filename)

        # file size in bytes
        size = os.path.getsize(path)

        if size > 50000:

            # open image
            image = imread(path)

            if(len(image.shape)<3):
                print(f"Rejecting: {filename} for reason: greyscale")

            elif len(image.shape)==3:

                # calculate blur
                grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                variance_of_laplacian = cv2.Laplacian(grayscale, cv2.CV_64F).var()

                if variance_of_laplacian > threshold:

                    print(f"Scaling: {filename}")

                    # images in dataset must be 1024x1024

                    # scale image
                    img = Image.open(path)
                    img = img.resize((1024, 1024), PIL.Image.ANTIALIAS)

                    # save image in new location
                    img.save(f"filteredImages/{filename}")

                    # release memory
                    img.close()

                else:
                    print(f"Rejecting: {filename} for reason: blurry")
                

            else:
                print(f"Rejecting: {filename} for reason: other")

            # release memory
            del image

        else:
                print(f"Rejecting: {filename} for reason: size")

print("Process end")