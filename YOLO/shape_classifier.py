#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

import json
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import random
import time
from tensorflow.keras.callbacks import TensorBoard

from proto import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMAGE_SIZE = 50
img_path = r'..\..\ImageGenerator\Targets'
target_path = r'..\..\ImageGenerator\Target Data'


def create_training_data(image_path, json_path, img_size):
    training_data = []
    for img in tqdm(os.listdir(image_path)):
        img_array = cv2.imread(os.path.join(image_path, img), cv2.IMREAD_GRAYSCALE)
        # img_array = cv2.Canny(img_array, 50, 50)
        new_array = cv2.resize(img_array, (img_size, img_size))
        json_file = img[0:-4] + '.json'
        json_dict = json.load(open(os.path.join(json_path, json_file)))
        class_number = Shape[json_dict['shape']].value
        training_data.append([new_array, class_number])

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

    pickle_out = open(r"Training_Data/shape_X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(r"Training_Data/shape_y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def train_model():
    pickle_in = open("Training_Data/shape_X.pickle", 'rb')
    X = pickle.load(pickle_in)

    pickle_in = open("Training_Data/shape_y.pickle", 'rb')
    y = pickle.load(pickle_in)

    X = X/255.
    y = to_categorical(y)

    conv_layer = 2
    layer_size = 256
    dense_layer = 1
    dense_size = 128

    NAME = "{}-conv-{}-nodes-{}-dense-{}-nodes-{}".format(conv_layer,
                                                          layer_size,
                                                          dense_layer,
                                                          dense_size,
                                                          int(time.time()))

    model = Sequential()
    model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for l in range(conv_layer - 1):
        model.add(Conv2D(layer_size, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    for _ in range(dense_layer):
        model.add(Dense(dense_size))
        model.add(Activation('relu'))

    model.add(Dense(14))
    model.add(Activation('softmax'))

    tensorboard = TensorBoard(log_dir=r"shape_classifier_logs\{}".format(NAME))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  )

    model.fit(X, y,
              batch_size=32,
              epochs=4,
              validation_split=0.3,
              callbacks=[tensorboard])

    model.save('shape_classifier.model')

    return model


def load_model():
    return tf.keras.models.load_model('shape_classifier.model')


def prep_image(img):
    # img_array = cv2.Canny(img, 50, 50)
    img_array = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img_array


if __name__ == '__main__':
    # pickle_in = open("./Training_data/shape_X.pickle", "rb")
    # X = pickle.load(pickle_in)
    # pickle_in = open("./Training_Data/shape_y.pickle", "rb")
    # y = pickle.load(pickle_in)
    # for i in range(1000):
    #     cv2.imshow('d', X[i])
    #     print(y[i])
    #     cv2.waitKey()

    # store_training_data()
<<<<<<< HEAD
    model = train_model()
    # loaded_model = load_model()
    # while True:
    #     filenum = random.randint(0, 1500)
    #     img = cv2.imread('../../ImageGenerator/Targets/' + str(filenum) + '.png', cv2.IMREAD_GRAYSCALE)
    #     # img = cv2.imread('foamie_test_frames/triangle.png', cv2.IMREAD_GRAYSCALE)
    #     img = prep_image(img)
    #     img = np.array(img)
    #     img = np.expand_dims(img, -1)
    #     arr = np.array([img])
    #     predictions = loaded_model.predict(arr)
    #     index = np.argmax(predictions[0])
    #     if index > 0:
    #         print(Shape(index).name, 'confidence:', predictions[0, index], sep=' ')
    #
    #     cv2.imshow('cropped_shape', img)
    #     key = cv2.waitKey(33)
    #     if key == 27:
    #         break
    #
    # cv2.destroyAllWindows()
=======
    # model = train_model()
    loaded_model = load_model()
    while True:
        filenum = random.randint(0, 1500)
        img = cv2.imread('../../ImageGenerator/Targets/' + str(filenum) + '.png', cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread('foamie_test_frames/triangle.png', cv2.IMREAD_GRAYSCALE)
        img = prep_image(img)
        img = np.array(img)
        img = np.expand_dims(img, -1)
        arr = np.array([img])
        predictions = loaded_model.predict(arr)
        index = np.argmax(predictions[0])
        if index > 0:
            print(Shape(index).name, 'confidence:', predictions[0, index], sep=' ')

        cv2.imshow('cropped_shape', img)
        key = cv2.waitKey()
        if key == 27:
            break

    cv2.destroyAllWindows()
>>>>>>> parent of 0865297... add shape_classifier.py
