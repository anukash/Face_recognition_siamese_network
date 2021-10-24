# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:35:12 2021

@author: Anurag
"""

from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

class L1_dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, input_embed, val_embed):
        return tf.math.abs(input_embed - val_embed)
    
def pre_process(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img
    
model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1_dist':L1_dist})

input_img = pre_process('C:/Users/Anurag/Desktop/Study/siamese_network/face_recognition_siamese/data/anchor/1efe46e2-3104-11ec-8b2d-04d4c47a9fee.jpg')
validation_img = pre_process('C:/Users/Anurag/Desktop/Study/siamese_network/face_recognition_siamese/data/negative/Aaron_Eckhart_0001.jpg')

# =============================================================================
# model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1_dist': L1_dist,
#                                                                       'BinaryCrossentropy': tf.losses.BinaryCrossentropy})
# =============================================================================

pred = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
print(np.round(pred[0][0], 7))
result = [True if prediction > 0.5 else False for prediction in pred][0]                  


plt.subplot(231)
plt.imshow(input_img)

plt.subplot(232)
plt.imshow(validation_img)

plt.subplot(233)
plt.text(0.5, 0.5, str(result))
# plt.set
plt.show()

# display_img = 200 * np.ones((700,250, 3), dtype=np.uint8)
# im1 = cv2.imread('C:/Users/Anurag/Desktop/Study/siamese_network/face_recognition_siamese/data/anchor/1efe46e2-3104-11ec-8b2d-04d4c47a9fee.jpg')
# im2 = cv2.imread('C:/Users/Anurag/Desktop/Study/siamese_network/face_recognition_siamese/data/anchor/8d2f83b1-3103-11ec-bd8a-04d4c47a9fee.jpg')

# display_img[0:250, 0:250] = im1
# display_img[250:500, 0:250] = im2
# cv2.putText(display_img, str(result), (75, 600), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
# cv2.imwrite('true.jpg', display_img)
# print(display_img.shape)
# cv2.imshow('result', display_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()