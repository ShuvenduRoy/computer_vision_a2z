# importing packaged from keras
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import *


# defininig the capsul net
def CapsNet(input_shape, n_class, num_routing):
    """
    A Captul network on MNIST
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A kearas model with 2 input 2 output
    """
    
    x = layers.Input(shape=input_shape)
    
    # Layer 1 : Just a convolutional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, activation='relu', name='conv1')(x)
    
    # Layer 2: 


