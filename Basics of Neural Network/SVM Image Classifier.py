""" Part 1 - Importing libraries"""
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

from __future__ import print_function

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


""" Part 2 - Importing and preprocessing dataset """
# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()



""" Resizing the dataset"""
num_trains = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Create a validation set form train set
mask = range(num_trains, num_trains+num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Resizing training
mask = range(num_trains)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make development set, which will be a small subset of the training set 
mask = np.random.choice(num_trains, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]








