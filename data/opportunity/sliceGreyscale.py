"""
Slices mono images into 256 64*64px subsections.
"""

import os

import PIL
from PIL import Image

if not os.path.exists("grey/slice"):
    os.mkdir("grey/slice")

directory = os.fsencode("grey").decode('ascii')

for file in os.listdir(directory):

    filename = os.fsdecode(file)

    if filename.endswith(".png"):

        path = os.path.join(directory, filename)
    
        # open image, convert to RGB, convert to greyscale
        image = Image.open(f"grey/{filename}")
    
        if not os.path.exists(f"grey/slice/{filename}"):
            os.mkdir(f"grey/slice/{filename}")

        # iterate for rows
        for x in range(0, 1024, 64):
            # iterate for columns
            for y in range(0, 1024, 64):
                # get subsection
                sub = image.crop((x, y, x+64, y+64))
                # save to subfolder
                sub.save(f"grey/slice/{filename}/x{x}y{y}.png")


        print(f"{filename} converted successfully.")

        # release memory
        del image