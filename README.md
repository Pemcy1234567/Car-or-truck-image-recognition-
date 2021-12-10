# Introduction to Convelutional Neural Network

Our objective is to figure out how a neural network can interpret a natural image good enough to solve issues similar to those that a human can. The neural networks which exel in this area are called convolutional neural network. The idea of convolutional neural networks are inspired by the human visual cortex. Next I will be taking you through a simple CNN moldel, predicting if an image is a Car or a Truck.

## Preprocessing

<details>
  <summary>Imports</summary>
  
```python
import numpy as np
import pandas as pd
from sklearn import preprocessing
import datetime
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, r2_score
import time
import sounddevice as sd
import shutil
import keyboard
from scipy.io.wavfile import write
from scipy.signal import find_peaks
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Input, Reshape, Flatten, GlobalMaxPooling1D, MaxPooling1D, \
    BatchNormalization, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, TensorBoard
from tensorflow.keras.optimizers import Adam
import librosa
from tensorboard.plugins.hparams import api as hp
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
```
</details>

```python
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
```

To start processing our images into a neural network approved form, we will first create a instance the ImageDataGenerator class. This class will help us generate batches from an image, to use for our model. Here, one of the parameters is called 'rescale'. What this does is it turns all the RGB values of every pixel into a number ranging between 1 and 0.

```python
train_generator = train_datagen.flow_from_directory(
    '/Users/maximilian/Downloads/archive (3)/train',
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary',
    shuffle=True,
    )

test_generator = test_datagen.flow_from_directory(
    '/Users/maximilian/Downloads/archive (3)/valid',
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary',
    shuffle=False,
    )
```

The flow_from_directory() function generates a tf.data.Dataset, from images files in a directory. As shown before now every images has a RGB value ranging between 0 and 1. I won't go into detail of every parameter only the most imporatant. Firstly, the target_size parameter simply resize's the image to a specific amount of pixels. This is essential, because the model can only take a specific amount of pixels.

![A broad overview](./Broad overview of CNN.png.png)

<iframe src="https://www.kaggle.com/embed/ryanholbrook/the-convolutional-classifier?cellIds=3&kernelSessionId=79128175" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="The Convolutional Classifier"></iframe>

This site was built using [GitHub Pages](https://pages.github.com/).

<div align="center">
  <img src="https://github.com/fmfn/BayesianOptimization/blob/master/examples/func.png"><br><br>
</div>

![BayesianOptimization in action](./bayesian_optimization.gif)
