import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # TensorFlow Lite 작업을 활성화합니다.
    tf.lite.OpsSet.SELECT_TF_OPS # TensorFlow 작업을 활성화합니다.
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
open('lite_model.tflite', 'wb').write(tflite_model)



