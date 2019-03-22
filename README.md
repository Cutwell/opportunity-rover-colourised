# Color Oppy
A colourised tribute to the Opportunity Mars rover, recently passed.

### Premise
This repository works by feeding a neural network trained to colourise greyscale images the photos taken by Opportunity's front facing hazcam over the course of it's operational life. These images are then stitched together into a video, the results of which can be found [here]().

### Requirements
To run the various scripts in this project, you must install the Keras and Tensorflow packages. SciPy and Skimage are also requirred, but may already be available with your Python installation.

### File and folder structure
```
-- /journey
    -- /colourImages
    -- /filteredImages
    -- /oppyImages
    -- filterSet.py
    -- getColour.py
    -- getMonochrome.py
    -- makeVideo.py
-- /output
-- /result
-- colourise.py
```

### Usage
1. Download monochrome images using `getMonochrome.py`.
2. Download colour training images using `getColour.py`.
3. Filter the training images using `filterSet.py`.
4. Run `coulourise.py` on the monochrome images using the training set.
5. Generate a video from the output using `makeVideo.py`.

### Thanks
This project makes use and is inspired by two seperate projects - [Colouring Grayscale Images](https://github.com/emilwallner/Coloring-greyscale-images/blob/master/floydhub/Beta-version/beta_version.ipynb) by [Emilwallner](https://github.com/emilwallner/) and [Opportunity's Journey](https://github.com/thatguywiththatname/Opportunitys-Journey) by [ThatGuyWithThatName](https://github.com/thatguywiththatname/) - both of which you should take time to explore.
