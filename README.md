# Introduction to Convelutional Neural Network

Our objective is to figure out how a neural network can interpret a natural image good enough to solve issues similar to those that a human can. The neural networks which exel in this area are called convolutional neural network. The idea of convolutional neural networks are inspired by the human visual cortex. Next I will be taking you through a simple CNN moldel, predicting if an image is a Car or a Truck.

## Preprocessing

<details>
  <summary>Imports</summary>
  
```python
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, TensorBoard
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from tensorflow.keras.layers.experimental import preprocessing
import datetime
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
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

The flow_from_directory() function generates a tf.data.Dataset, from images files in a directory. This function acctually makes it that every image in this directory has a RGB value ranging between 0 and 1. I won't go into detail of every parameter I will only be going through the most imporatant. Firstly, the target_size parameter - this simply resize's the image to a specific amount of pixels by zooming into the picture. This is important, because the neurons can only take a specific amount of pixels. Secondly, the class_mode - this simply is telling the fuction if I have one or more classes to classify. The two main strings herer are 'binray' and 'catagorical'. 'Binary' means the labels will be 1D binary labels, whilst the 'catagorical' string means the labels will be 2D one-hot encoded.

## Model

```python
model = Sequential([

    preprocessing.RandomContrast(0.3),

    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(units=12, activation="relu"),
    layers.Dropout(0.4),
    layers.BatchNormalization(),
    layers.Dense(units=12, activation="relu"),
    layers.Dense(units=1, activation="sigmoid"),
])
```

### Data Augmentation
### Convelution and ReLu
### Max Pooling
### Sliding window
### Dropout
### BatchNormalization



![A broad overview](./Broad overview of CNN.png.png)

<iframe src="https://www.kaggle.com/embed/ryanholbrook/the-convolutional-classifier?cellIds=3&kernelSessionId=79128175" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="The Convolutional Classifier"></iframe>

This site was built using [GitHub Pages](https://pages.github.com/).

<div align="center">
  <img src="https://github.com/fmfn/BayesianOptimization/blob/master/examples/func.png"><br><br>
</div>

![BayesianOptimization in action](./bayesian_optimization.gif)
