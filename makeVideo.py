"""
Take all images from the {imageDir} directory and compile them into a video
"""

from cv2 import VideoWriter, VideoWriter_fourcc, destroyAllWindows, imread
import os

if not os.path.exists(f"output"):
    os.mkdir(f"output")

width = 1024
height = 1024
fps = 5
directory = "/floyd/input/data/output"
title = f"output/opportunity-rover-colourised-{fps}fps.mp4"

folder = os.listdir(directory)

fourcc = VideoWriter_fourcc(*'MP4V')
video = VideoWriter(title, fourcc, float(fps), (width, height))

# sort files by sol
sols = {}
for name in folder:
    a = name.split("-")
    sols.setdefault(a[0], []).append(name)

for key in sorted(sols):
    # sort sol files by timestamp
    files = sorted(sols[key])

    for filename in files:
        print(f"Processing {filename}")
        video.write(imread(os.path.join(directory, filename)))

# release all loaded resources
destroyAllWindows()
video.release()