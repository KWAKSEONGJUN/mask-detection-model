import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2D()
        self.pool1 = MaxPool2D()
        self.drop1 = Dropout()

        self.conv2 = Conv2D()
        self.pool2 = MaxPool2D()
        self.drop2 = Dropout()

        self.conv3 = Conv2D()
        self.pool3 = MaxPool2D()
        self.drop3 = Dropout()

        self.flat = Flatten()
        self.fc = Dense()
        self.output = Dense()


    def call(self, x, is_training):
        h_conv1 = self.conv1(x)
        h_pool1 = self.pool1(h_conv1)
        h_drop1 = self.drop1(h_pool1, training=is_training)

        h_conv2 = self.conv1(h_drop1)
        h_pool2 = self.pool1(h_conv2)
        h_drop2 = self.drop1(h_pool2, training=is_training)

        h_conv3 = self.conv1(h_drop2)
        h_pool3 = self.pool1(h_conv3)
        h_drop3 = self.drop1(h_pool3, training=is_training)

        h_flat = self.flat(h_drop3)
        h_fc = self.fc(h_flat)
        logits = self.output(h_fc)

        return logits


















