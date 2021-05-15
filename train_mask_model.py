import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from cnn_model import CNN
from dataset.load_data import load_dataset
import os
os.chdir('dataset')


def compute_accuracy(y_pred, y):
    prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return accuracy


# data load
x_train, y_train = load_dataset(flag='train', size=100, is_random=True)
x_valid, y_valid = load_dataset(flag='valid', size=800, is_random=True)
x_test, y_test = load_dataset(flag='test', size=800, is_random=False)
test_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
test_data = test_data.batch(1)
test_data_iter = iter(test_data)
os.chdir('..')

mask_model = CNN()

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model_checkpoint = ModelCheckpoint('my_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

mask_model.fit(x_test, y_test,
               batch_size=50, epochs=20,
               validation_data=(x_train, y_train),
               callbacks=[early_stopping, model_checkpoint])

avg_accuracy = 0.
for i in range(len(x_test)):
    x_batch, y_batch = next(test_data_iter)
    avg_accuracy += compute_accuracy(mask_model(x_batch), y_batch)

avg_accuracy /= len(x_test)
print('Accuracy : %f' % avg_accuracy)







