"""
Slice color images into 256 64*64px subsections.
"""

import os

import PIL
from PIL import Image

if not os.path.exists(f"color/slice"):
    os.mkdir(f"color/slice")

directory = os.fsencode("color").decode('ascii')

for file in os.listdir(directory):

    filename = os.fsdecode(file)

    if filename.endswith(".png"):

        path = os.path.join(directory, filename)
    
        # open image, convert to RGB, convert to greyscale
        image = Image.open(f"color/{filename}").convert('RGB')

        if not os.path.exists(f"color/slice/{filename}"):
            os.mkdir(f"color/slice/{filename}")

        # iterate for rows
        for x in range(0, 1024, 64):
            # iterate for columns
            for y in range(0, 1024, 64):
                # get subsection
                sub = image.crop((x, y, x+64, y+64))
                # save
                sub.save(f"color/slice/{filename}/x{x}y{y}.png")

        print(f"{filename} converted successfully.")

        # release memory
        del image