import tensorflow as tf
from cnn import CNN
from dataset.load_data import load_data
import os
os.chdir('dataset')

EPOCH = 80
batch_size = 10

x_train, y_train = load_data(flag='train')
x_train /= 255.

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(len(x_train)).batch(batch_size)
train_data_iter = iter(train_data)

x_test, y_test = load_data(flag='test')
x_test /= 255.
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.batch(1)
test_data_iter = iter(test_data)

optimizer = tf.optimizers.Adam(1e-3)


def cross_entropy_loss(y_pred, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))


def train_step(model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def compute_accuracy(y_pred, y):
    prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return accuracy


mask_model = CNN()

for epoch in range(EPOCH):
    x_batch, y_batch = next(train_data_iter)
    train_step(mask_model, x_batch, y_batch)
    loss = cross_entropy_loss(mask_model(x_batch), y_batch)
    print('EPOCH : %d\t Loss: %f' % (epoch+1, loss))

avg_accuracy = 0.
for i in range(len(x_test)):
    x_batch, y_batch = next(test_data_iter)
    avg_accuracy += compute_accuracy(mask_model(x_batch), y_batch)

avg_accuracy /= len(x_test)
print('Accuracy : %f' % avg_accuracy)

os.chdir('..')
mask_model.save('my_model')






