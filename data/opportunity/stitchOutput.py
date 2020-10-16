"""
Stitch 64x64px images into a single 1024x1024px image.
"""

import os
import PIL
from PIL import Image

directory = os.fsencode("color").decode('ascii')

if not os.path.exists(f"stitch"):
    os.mkdir(f"stitch")

for folder in os.listdir(directory):
    image = Image.new('RGB', (1024, 1024))

    for filename in os.listdir(f"color/{folder}"):

        if filename.endswith(".png"):

            sub = Image.open(f"color/{folder}/{filename}")

            a = filename.split("y")
            x = int(a[0].replace("x", ""))
            y = int(a[1].replace(".png", ""))

            image.paste(im=sub, box=(x, y))

    image.save(f"stitch/{folder}")

    print(f"{folder} converted successfully.")

    # release memory
    del image