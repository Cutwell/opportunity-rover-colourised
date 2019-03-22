"""
Scrape NASAs {roverName} gallery page and get all the URLs for images from each sol
Using this list of links, scrape and download all of {roverName}s images that fulfill
the variables below
Helper functions use info from https://mars.nasa.gov/mer/gallery/edr_filename_key.html
"""

from requests_html import HTMLSession
import os

# Variables to change
imageDir = "oppyImages"
roverName = "opportunity"  # opportunity or spirit
cameraPos = "f"  # f for front or r for rear (Hazcam position)
cameraSide = "l"  # l for left, r for right (Hazcam side)
# - - - - - - - - - -


def fromCorrectCamera(imageName):
    """
    Detect if image is from {roverName}s {cameraSide} {cameraPos} Hazcam
    """
    if (
        imageName[0] == ("1" if roverName == "opportunity" else "2")
        and imageName[1] == cameraPos.upper()
        and imageName[23] == cameraSide.upper()
    ):
        return True
    return False


def isFullFrame(imageName):
    """
    Detect if image is "Full Frame"
    """
    if imageName[11:14] == "EFF":
        return True
    return False


def getRawImageURL(relativePath):
    return "https://mars.nasa.gov/mer/gallery/all/" + relativePath


def getImageURLFromTag(tagObject):
    """
    Return either the URL to the image or False
    """
    try:
        imageURL = tagObject.attrs["href"]
        if imageURL.endswith(".JPG"):
            return imageURL
    except KeyError:
        pass
    return False


solImageLinks = []
galleryURL = f"https://mars.nasa.gov/mer/gallery/all/{roverName}.html"
session = HTMLSession()

if not os.path.exists(imageDir):
    os.mkdir(imageDir)

print("\nCollecting links to images from each sol")

r = session.get(galleryURL)

# Get each URL that points to each sol photos were taken on
for option in r.html.find("option"):
    imageURL = option.attrs["value"]

    if imageURL.startswith(f"{roverName}_{cameraPos}"):
        imageURL = imageURL.replace(".html", "")
        solImageLinks.append(
            f"https://mars.nasa.gov/mer/gallery/all/{imageURL}_text.html"
        )

print(f"Found {len(solImageLinks)} sols containing images")
print("\nDownloading individual images\n")

for solImages in solImageLinks:
    r = session.get(solImages)

    for anchorTag in r.html.find("a"):
        currentURL = getImageURLFromTag(anchorTag)
        if not currentURL:
            continue  # Skip non-image URLs

        filename = currentURL.split("/")[-1]

        if fromCorrectCamera(filename) and isFullFrame(filename):

            sol = int(currentURL.split("/")[2])
            timestamp = filename[2:11]

            # :04d = always 4 digits (e.g. sol 0006)
            newFilename = f"{sol:04d}-{timestamp}"

            # Sometimes there are multiple photos with the same {newFilename}, as there
            # are multiple versions of some images, the latest version of which is
            # better quality / less corrupted. Because the <a> tags are read in the
            # correct order, the "best" version will be read last and will overwrite the
            # previous version(s). This does however mean that bandwidth will be wasted
            # downloading images to be immediatley be overwritten

            # For testing the corrupted version overwriting
            # solImageLinks = ["https://mars.nasa.gov/mer/gallery/all/opportunity_f3644_text.html"]

            # TODO: Instead of overwriting, look at all the filenames and check for
            # duplicates with different / higher versions

            print(f"Found image from sol {newFilename}")
            rawImgURL = getRawImageURL(currentURL)
            rawImg = session.get(rawImgURL).content

            with open(f"{imageDir}/{newFilename}.jpg", "wb") as imageOut:
                imageOut.write(rawImg)
            print(currentURL, "downloaded")

print("Process end")