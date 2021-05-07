import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(64, 3, 3, padding='same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=1)
        self.drop1 = Dropout(0.25)

        self.conv2 = Conv2D(64, 3, 3, padding='same', activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=1)
        self.drop2 = Dropout(0.25)

        self.conv3 = Conv2D(64, 3, 3, padding='same', activation='relu')
        self.pool3 = MaxPool2D(pool_size=(2, 2), strides=1)
        self.drop3 = Dropout(0.25)

        self.flat = Flatten()
        self.fc = Dense(200, activation='relu')
        self.output_layer = Dense(2, activation='softmax')


    def call(self, x):
        h_conv1 = self.conv1(x)
        h_pool1 = self.pool1(h_conv1)
        h_drop1 = self.drop1(h_pool1)

        h_conv2 = self.conv2(h_drop1)
        h_pool2 = self.pool2(h_conv2)
        h_drop2 = self.drop2(h_pool2)

        h_conv3 = self.conv3(h_drop2)
        h_pool3 = self.pool3(h_conv3)
        h_drop3 = self.drop3(h_pool3)

        h_flat = self.flat(h_drop3)
        h_fc = self.fc(h_flat)
        logits = self.output_layer(h_fc)

        return logits


















