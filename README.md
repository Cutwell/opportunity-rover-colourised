# Opportunity-Rover-Colourised

### Floydhub commands
Generate the fresh model:
```floyd run "python3 new.py" --gpu --tensorboard```

Train the model:
```floyd run "python3 train.py" --data zachary946/datasets/curiosity-rover-dataset/1:curiosity --data zachary946/projects/opportunity-rover-colourised/JOB:model --gpu --tensorboard```

Running the model on unseen data:
```floyd run "python3 colourise.py" --data zachary946/datasets/opportunity-rover-dataset/1:opportunity --data zachary946/projects/opportunity-rover-colourised/JOB:model --gpu```

Stitching model output into full images:
```floyd run "python3 data/opportunity/stitchOutput.py" --data zachary946/projects/opportunity-rover-colourised/JOB:data --gpu --tensorboard```

### Thanks
This project makes use and is inspired by two separate projects - [Colouring Grayscale Images](https://github.com/emilwallner/Coloring-greyscale-images/blob/master/floydhub/Beta-version/beta_version.ipynb) by [Emilwallner](https://github.com/emilwallner/) and [Opportunity's Journey](https://github.com/thatguywiththatname/Opportunitys-Journey) by [Psidex](https://github.com/Psidex/) - both of which you should take time to explore.
