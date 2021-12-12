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

The best way to improve the performance of a any model is to let it learn from more data. One way of getting more data is by transforming the data which you already have. This process is called data augmentation. In my case (as shown below), only the contrast is changed, but you can change it depending on the type of image classification you are performing.

```python
preprocessing.RandomContrast(0.3)
```

For example this image here is from the data set. This image will be augemnetion various times, and the model can than learn frmo 16x more images.

Before data augmentation   |  After data augmentation
:-------------------------:|:-------------------------:
![](https://www.kaggleusercontent.com/kf/79128151/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..LfH1zOGWvLB6Qoz4qA_08A.0E0NiLLd-wnXT8Jbqv6sFYpBp6mWk7XyNpmxXTH0ySqfZ9lzphI0ipYWGbmqv00n9lD810Lr4LfvKPu_9OJidoFZy6L7AWzOsHV7PD1APlIB2VjZKxQt_g6pacaVYhE5u0Si65I7MVnqmyXsbe61bOabdgviE8EOO-b4IMpqkO_z-pFVdMG_5_84JKxOuuqJ9tWydfBqshLeKY02s-lGHdIhc2mi8iZbHtzoKl4T3AmaITSGxQoxqWSEyb-ybHV65MjjVJHDrF9SIuDhh5_f5p_FKxaVhOOsDG8vxShhD6rYmWBrxu4tXyZCa9A7LF5rq2wfhPx8g4Z5wylzI1rm7yxuPucbCOQHwyDeyj_uQG7lt9fOWDHyhf7ql5mOzpLZaICksmlp0gvOscyH_cO_khSmW8f-BPAdBC1CukPfXpRjrvol5cy80woAMTHihBNlSq2JvfjHgdpfC6H9BBHnohGJMGg0xX2u3kkkHnHVQB17igTcq1ysqxdidF_5us_b2xZ2RUYOI5TPFdk589RGkWKubR29adLvcONBW-wqB6e74qkT55Bnb9Pa3SPbyWvd7DNssAMdDoST7yuYNiZx40Rl7hxpcM_MvzeTLizXGOXaL_OW_HKZyrupteHUshw517PXEMH2WbhYVCW9GV0MWUFdL4BTkdWImunM3wL1NVk.Wrh7ygJaPpOCLC9J2EBSJQ/__results___files/__results___7_0.png)  |  ![](https://i.imgur.com/UaOm0ms.png)

### Convelution, ReLu and Max Pooling

Now the CNN; The purpose of a CNN is feature extraction. There are three basic operations in the feature extraction process. 

1. Filter an image for a particular feature (convolution)
2. Detect that feature within the filtered image (ReLU)
3. Condense the image to enhance the features (maximum pooling)

The convolution is layers.Conv2D(), the ReLu is layers.ReLU and the max pooling is layers.MaxPool2D(). The entire process looks similar to this picture.

<div align="center">
  <img src="https://i.imgur.com/IYO9lqp.png"><br><br>
</div>

During the filtering step a kernel scans over over an image. Without going into detail, the kernel creates a new image, which has certain features attached to it. In this case the kernel filtered in horizontal lines of the car. After this process, this convoluted image is then passed onto the ReLU function, which allowed for the neural network to 'understnad' the image. In this case all pixels which are not horizontal lines will be turned to 0 and all pixels which are horizontal will be turned into yellow. 

### Sliding window
### Dropout
### BatchNormalization



![A broad overview](./Broad overview of CNN.png)

<iframe src="https://www.kaggle.com/embed/ryanholbrook/the-convolutional-classifier?cellIds=3&kernelSessionId=79128175" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="The Convolutional Classifier"></iframe>

This site was built using [GitHub Pages](https://pages.github.com/).

<div align="center">
  <img src="https://github.com/fmfn/BayesianOptimization/blob/master/examples/func.png"><br><br>
</div>

![BayesianOptimization in action](./bayesian_optimization.gif)
