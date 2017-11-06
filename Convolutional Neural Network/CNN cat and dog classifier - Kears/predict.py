from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# set the dimesion
img_width, img_height = 64, 64

# load the saved models
model=load_model('model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# predicting images
img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# img = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(img_width, img_height))
# y = image.img_to_array(img)
# y = np.expand_dims(y, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes)

# # predicting multiple images at once
# img = image.load_img('test2.jpg', target_size=(img_width, img_height))
# y = image.img_to_array(img)
# y = np.expand_dims(y, axis=0)
#
# # pass the list of multiple images np.vstack()
# images = np.vstack([x, y])
# classes = model.predict_classes(images, batch_size=10)
#
# # print the classes, the images belong to
# print classes
# print classes[0]
# print classes[0][0]
