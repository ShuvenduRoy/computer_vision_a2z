"""Some key layers user for constructing a Capsule network"""

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers

def squash(vectors, axis=-1):
    """
    The non linear activation used in Capsule. It drives the length as a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    
    s_square_norm = K.sum(K.square(vectors), axis=axis, keepdims=True)
    scale = s_square_norm / (1 + s_square_norm) / K.sqrt(s_square_norm + K.epsilon)
    return scale * vectors


def PrimaryCap(inputs, dim_vectors, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all captules
    :param inputs: 4D tensor, shape = [None, width, height, channels]
    :param dim_vectors: the dim of the output vector of captule
    :param n_channels: the number of types of captules
    :return: output tensor, shape = [None, num_captule, dim_vector]
    """
    output = layers.Conv2D(filters=dim_vectors*n_channels, kernel_size=kernel_size, strides=strides, padding=padding, name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vectors], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primary_squash')(outputs)

"""
# The following is another way to implement primary capsule layer. This is much slower.
# Apply Conv2D `n_channels` times and concatenate all capsules
def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    outputs = []
    for _ in range(n_channels):
        output = layers.Conv2D(filters=dim_vector, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
        outputs.append(layers.Reshape([output.get_shape().as_list()[1] ** 2, dim_vector])(output))
    outputs = layers.Concatenate(axis=1)(outputs)
    return layers.Lambda(squash)(outputs)
"""