"""
Uses the NASA API to get images from Curiosity's cameras.
"""

from requests_html import HTMLSession
import os
import json
from urllib.parse import urlparse

if not os.path.exists("colourImages"):
    os.mkdir("colourImages")

# template for getting images
# camera = "mast"
solRaw = "https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?sol={sol}&camera=mast&api_key=DEMO_KEY"

session = HTMLSession()

#  get images from first 1000 sols

"""
Can reach end of Sol 40 from 1 before hitting the rate limit.
Change 'start' by ~40 each time the ratelimit resets, or refer to the Sol the process stalled at.
"""
start = 160

for sol in range(start, 1000):

    print(f"Getting images for Sol: {sol}")

    solLink = solRaw.format(sol = sol)

    r = session.get(solLink)

    data_json = json.loads(r.content)

    try:

        for item in data_json["photos"]:

            rawImgUrl = item["img_src"]

            a = urlparse(rawImgUrl)
            newFilename = os.path.basename(a.path)

            print(f"Found image from {rawImgUrl}")
            if os.path.isfile(f"colourImages/{newFilename}"):
                print("Already downloaded, passing")
                continue

            else:
                rawImg = session.get(rawImgUrl).content
                print(rawImgUrl, "downloaded")

                with open(f"colourImages/{newFilename}", "wb") as imageOut:
                    imageOut.write(rawImg)
    
    except KeyError:

        if data_json["error"]:
            print(f"Error: {data_json['error']['code']}")
            break
        else:
            print(f"No images found for sol: {sol}.")

print("Process end")