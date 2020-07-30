from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.models import Sequential
# from keras.utils import to_categorical

import json
import numpy as np
import os
import cv2
import random
import pickle
from tqdm import tqdm
from proto import *

IMAGE_SIZE = 50
img_path = r'..\..\ImageGenerator\Targets'
target_path = r'..\..\ImageGenerator\Target Data'


def create_training_data(image_path, json_path, img_size):
    training_data = []
    for img in tqdm(os.listdir(image_path)):
        img_array = cv2.imread(os.path.join(image_path, img))
        new_array = cv2.resize(img_array, (img_size, img_size))
        json_file = img[0:-4] + '.json'
        json_dict = json.load(open(os.path.join(json_path, json_file)))
        class_number = Color[json_dict['shape_color']].value

        training_data.append([new_array, class_number])

        random.shuffle(training_data)

    return np.array(training_data)


def store_training_data():
    X_data = []
    y_data = []
    shape_training_data = create_training_data(img_path, target_path, IMAGE_SIZE)
    for feature, label in shape_training_data:
        X_data.append(feature)
        y_data.append(label)

    X = np.array(X_data)
    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y = np.array(y_data)

    pickle_out = open(r"Training_Data/shape_color_X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(r"Training_Data/shape_color_y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def define_model():
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))

    model.add(Dense(14))
    model.add(Activation('sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model