import tensorflow as tf
from cnn import CNN

optimizer = tf.optimizers.Adam(1e-2)


def cross_entropy_loss(y_pred, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))


def train_step(model, x, y, is_training):
    with tf.GradientTape() as tape:
        logits = model(x, is_training)
        loss = cross_entropy_loss(logits, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


mask_model = CNN()




