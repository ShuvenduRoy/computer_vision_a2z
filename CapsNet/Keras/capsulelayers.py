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

