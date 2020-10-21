"""
Stitch 64*64px images into a single 1024*1024px image.
"""

import os
import PIL
from PIL import Image

directory = "/floyd/input/data/output"

if not os.path.exists(f"output"):
    os.mkdir(f"output")

for folder in os.listdir(directory):
    image = Image.new('RGB', (1024, 1024))

    for filename in os.listdir(f"{directory}/{folder}"):

        if filename.endswith(".png"):

            sub = Image.open(f"{directory}/{folder}/{filename}")

            a = filename.split("y")
            x = int(a[0].replace("x", ""))
            y = int(a[1].replace(".png", ""))

            image.paste(im=sub, box=(x, y))

    image.save(f"output/{folder}")

    print(f"{folder} converted successfully.")

    # release memory
    del image