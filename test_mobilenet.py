import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import cv2

IMG_SIZE = 224

face_model = cv2.dnn.readNet('caffe_model/deploy.prototxt.txt',
                             'caffe_model/res10_300x300_ssd_iter_140000.caffemodel')

my_model = tf.lite.Interpreter('lite_mobile_model.tflite')
my_model.allocate_tensors()
input_details, output_details = my_model.get_input_details(), my_model.get_output_details()

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('video/04.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104., 117., 123.))
    face_model.setInput(blob)
    detections = face_model.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, dsize=(IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.array(face, dtype=np.float32)
        face = np.expand_dims(face, axis=0)
        face /= 255.

        my_model.set_tensor(input_details[0]['index'], face)
        my_model.invoke()
        mask, no_mask = my_model.get_tensor(output_details[0]['index'])

        if mask > no_mask:
            color = (0, 255, 0)
            label = 'Mask (%d%%)' % (mask * 100)
        else:
            color = (0, 0, 255)
            label = 'No Mask (%d%%)' % (no_mask * 100)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text=label, color=color, org=(x1, y1-10), fontScale=1, fontFace=cv2.FONT_HERSHEY_DUPLEX)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()



