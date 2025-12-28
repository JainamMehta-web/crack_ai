import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('crack_defect_model.keras')

classes = ['Flexural Crack', 'HoneyCombing', 'Shear Crack']

img = cv2.imread('test.jpg')
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
print("Detected:", classes[np.argmax(pred)])