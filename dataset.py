import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_data(db_name:str):
    if db_name == 'cifar10':
        return load_cifar10_data()
    elif db_name == 'mnist':
        return load_mnist_data()
    

def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x = tf.concat([x_train, x_test], axis=0)
    
    x = (tf.cast(x, tf.float32) / 127.5) - 1
    print(np.min(x), np.max(x))
    
    y = tf.concat([y_train, y_test], axis=0)
    print(x.shape, y.shape)
    
    x = tf.data.Dataset.from_tensor_slices(x)
    y = tf.data.Dataset.from_tensor_slices(y)
    ds = tf.data.Dataset.zip(x, y)
    return ds, get_cifar10_label_dict()


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x = tf.concat([x_train, x_test], axis=0)
    
    x = (tf.cast(x, tf.float32) / 127.5) - 1
    print(np.min(x), np.max(x))
    
    y = tf.concat([y_train, y_test], axis=0)
    y = tf.cast(y, tf.float32)
    print(x.shape, y.shape)
    
    x = tf.data.Dataset.from_tensor_slices(x)
    x = x.map(lambda x: tf.numpy_function(cv2.resize, [x, (32, 32)], [tf.float32]))
    x = x.map(lambda x: tf.reshape(x, (32, 32, 1)))
    y = tf.data.Dataset.from_tensor_slices(y)
    ds = tf.data.Dataset.zip(x, y)
    
    label_dict = {}
    for i in range(10):
        label_dict[i] = i
    
    ds = ds.shuffle(buffer_size=100)
    return ds.prefetch(tf.data.AUTOTUNE), label_dict


def get_cifar10_label_dict():
    label_dict = {
        0	: 'airplane',
        1	: 'automobile',
        2	: 'bird',
        3	: 'cat',
        4	: 'deer',
        5	: 'dog',
        6	: 'frog',
        7	: 'horse',
        8	: 'ship',
        9	: 'truck',
    }
    return label_dict
