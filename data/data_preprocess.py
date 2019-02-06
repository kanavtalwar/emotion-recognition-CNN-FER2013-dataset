import os
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self):
        self.img_width = 48
        self.img_height = 48

    def get_data(self):
        self.dataset = pd.read_csv(os.path.abspath('./data/data.csv'))
        X = self.dataset.iloc[:, 1].values
        y = self.dataset.iloc[:, 0].values

        self.images = np.empty((len(X), self.img_height, self.img_width, 1))
        i=0
        for pixel_string in X:
            temp = [float(pixel) for pixel in pixel_string.split(' ')]
            temp = np.asarray(temp).reshape(self.img_height, self.img_width)
            temp = resize(temp, (self.img_height, self.img_width), order=3, mode='constant')

            channel = np.empty((self.img_height, self.img_width, 1))
            channel[:, :, 0] = temp
            
            self.images[i, :, :, :] = channel
            i = i + 1

        self.images /= 255.0
        self.labels = keras.utils.to_categorical(y, 7)    
