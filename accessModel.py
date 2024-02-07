import cv2
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model

img = cv2.imread('test/happy.jpg')
resize = tf.image.resize(img, (256,256))

image_classifier_model = load_model(os.path.join('models', 'imageclassifier.h5'))
predicted = image_classifier_model.predict(np.expand_dims(resize/255, 0))

if predicted < 0.5: 
    print(f'The predicted class is Sad with {predicted*100}% chances')
else:
    print(f'The predicted class is Happy with {predicted*100}% chances')