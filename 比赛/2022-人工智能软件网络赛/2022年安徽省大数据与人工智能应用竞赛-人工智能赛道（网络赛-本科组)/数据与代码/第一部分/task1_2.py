#task1_2.py
import tensorflow as tf
import sys
import sklearn
import cv2
print('Python version Info: ' + sys.version)
print('TF version Info: ' + tf.__version__)
print('OpenCV version Info: ' + cv2.__version__)
print("Sklearn verion Info {}".format(sklearn.__version__))

vec_1=tf.constant([1,2,2,1])
vec_2=tf.constant([2,1,1,2])
ver_add=tf.add(vec_1,vec_2)
print(ver_add)