#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pickle
from tqdm import tqdm
import os
import json
import random
from proto import *
from math import sqrt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import keras.backend as K

GRID_H, GRID_W = 13, 13

IMAGE_SIZE = (4000, 6000)
BATCH_SIZE = 16
WARM_UP_BATCHES = 0

NO_OBJECT_SCALE = 1.
OBJECT_SCALE = 5.
COORD_SCALE = 1.


img_path = r'..\..\ImageGenerator\Images'
data_path = r'..\..\ImageGenerator\Image Data'


def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:3]

    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])

    conf_mask = tf.zeros(mask_shape)
    # coord_mask = tf.zeros(mask_shape)

    # seen = tf.Variable(0.)
    # total_recall = tf.Variable(0.)

    pred_box_xy = tf.sigmoid(y_pred[..., 1:3])  # + cell_grid
    pred_box_wh = tf.exp(y_pred[..., 3:])  # * np.reshape(ANCHORS, [1, 1, 1, BOX, 2])

    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    true_box_xy = y_true[..., 1:3]
    true_box_wh = y_true[..., 3:]

    true_wh_half = true_box_wh / 2.
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy - pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)
    best_ious = tf.reduce_max(iou_scores)

    true_box_conf = iou_scores * y_true[..., 0]

    coord_mask = tf.expand_dims(y_true[..., 0], axis=-1) * COORD_SCALE

    conf_mask = conf_mask + tf.cast(best_ious < 0.6, tf.float32) * (1 - y_true[..., 0]) * NO_OBJECT_SCALE

    conf_mask = conf_mask + y_true[..., 0] * OBJECT_SCALE

    # warm up training
    # no_boxes_mask = tf.cast(coord_mask < COORD_SCALE/2., tf.float32)
    # seen.assign_add(1.)
    #
    # true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES),
    #                         lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask])

    nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, tf.float32))
    nb_conf_box = tf.reduce_sum(tf.cast(conf_mask > 0.0, tf.float32))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.

    loss = loss_xy + loss_wh + loss_conf

    # print(true_box_xy)
    # nb_true_box = tf.reduce_sum(y_true[..., 0])
    # nb_pred_box = tf.reduce_sum(tf.cast(true_box_conf > 0.5, tf.float32)) * tf.cast(pred_box_conf > 0.3, tf.float32)

    # current_recall = nb_pred_box / (nb_true_box + 1e-6)
    # total_recall.assign_add(current_recall)

    # loss = tf.print(loss, [tf.zeros(1)], message='Dummy line \t', summarize=1000)
    # loss = tf.print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    # loss = tf.print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    # loss = tf.print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    # loss = tf.print(loss, [loss], message='Total Loss \t', summarize=1000)
    # loss = tf.print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    # loss = tf.print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)

    return loss

# def custom_loss(y_true, y_pred):
#     print(tf.shape(y_true)[:4])
#     # print(y_pred)
#     # print(y_true[0, :, :, 0])
#     # for i in y_true:
#     #     print(i[:, :, 0])
#     # loss = []
#     # print(y_true.shape)
#     # for i in range(y_true.shape[0]):
#     grid_loss = 0
#     for m in range(y_true.shape[1]):
#         for n in range(y_true.shape[2]):
#             if tf.gather_nd(y_true, [m, n, 0]) == 1:
#                 print(1)
#                 # x = tf.gather_nd(y_true, [m, n, 0])
#                 # print('row', m, 'column', n, 'is', x)
#                 # print(y_true[0, m, n].shape)
#                 # if object in grid cell, add localization loss
#                 # grid cells are in [p, x, y, w, h] format
#                 # grid_loss += (y_true[m, n, 1] - y_pred[m, n, 1])**2 + \
#                 #              (y_true[m, n, 2] - y_pred[m, n, 2])**2
#                 # grid_loss += (sqrt(y_true[m, n, 3]) - sqrt(y_pred[m, n, 3]))**2 + \
#                 #              (sqrt(y_true[m, n, 4]) - sqrt(y_pred[m, n, 4]))**2
#                 # grid_loss += (1 - y_pred[m, n, 0])**2
#             else:
#                 # grid_loss += ((0 - y_pred[m, n, 0])**2) / 2.
#                 print('not 1')
#     # # loss[i] = grid_loss
#     # #
#     return grid_loss


def create_training_data(image_path, json_path):
    training_data = []
    for img in tqdm(os.listdir(image_path)):
        img_array = cv2.imread(os.path.join(image_path, img))
        img_array = cv2.resize(img_array, (6000, 6000), interpolation=cv2.INTER_AREA)
        json_file = img[0:-4] + '.json'
        json_dict = json.load(open(os.path.join(json_path, json_file)))
        grid_data = np.zeros((20, 20, 5))
        for target in json_dict:
            x = json_dict[target]['x'] + 0.5 * json_dict[target]['width']  # move x, y to middle of target
            y = json_dict[target]['y'] + 0.5 * json_dict[target]['height']
            grid_x = int(x * grid_data.shape[1])  # calculates grid location of target
            grid_y = int(y * grid_data.shape[0])

            cell_data = np.array([1, x, y, json_dict[target]['width'], json_dict[target]['height']])

            grid_data[grid_y, grid_x] = cell_data
        training_data.append([img_array, grid_data])
    return np.array(training_data)


def store_training_data():
    X_data = []
    y_data = []
    box_training_data = create_training_data(img_path, data_path)
    for feature, label in box_training_data:
        X_data.append(feature)
        y_data.append(label)
    X = np.array(X_data)
    print(X[:, :, :].shape)
    X = X.reshape(-1, 6000, 6000, 3)
    y = np.array(y_data)

    pickle_out = open(r"Training_Data/bounding_box_X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    pickle_out = open(r"Training_Data/bounding_box_y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def train_model():
    pickle_in = open("Training_Data/bounding_box_X.pickle", 'rb')
    X = pickle.load(pickle_in)

    pickle_in = open("Training_Data/bounding_box_y.pickle", 'rb')
    y = pickle.load(pickle_in)

    X = X/255.

    model = Sequential()
    model.add(Conv2D(64, (7, 7), input_shape=X.shape[1:], strides=2))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(192, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Conv2D(256, (1, 1), padding='same'))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Conv2D(256, (1, 1), padding='same'))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Conv2D(512, (1, 1), padding='same'))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Conv2D(1024, (3, 3), padding='same'))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(512, (1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(1024, (3, 3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(1024, (3, 3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(1024, (3, 3)))

    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(1024, (3, 3), strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Flatten())

    # model.add(Dense(4096))
    # model.add(Activation('relu'))

    model.add(Dense(2000))
    model.add(Activation('relu'))

    model.add(Reshape((20, 20, 5)))


    model.compile(loss=custom_loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X, y,
              batch_size=5,
              epochs=4,
              validation_split=0.2,
              )
    model.save('bounding_box.model')

    return model


def load_model():
    pass


if __name__ == '__main__':
    train_model()
    # store_training_data()
    # pickle_in = open(r"Training_Data/bounding_box_X.pickle", "rb")
    # X = pickle.load(pickle_in)
    #
    # pickle_in = open(r"Training_Data/bounding_box_y.pickle", "rb")
    # y = pickle.load(pickle_in)
    #
    # X = X/255.
    #
    # img = X[1]
    #
    # print('Original Dimensions : ', img.shape)
    #
    # scale_percent = 10  # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # # resize image
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #
    # cv2.imshow('d', resized)
    # print(y[1, :, :, 0])
    # cv2.waitKey()
    # cv2.destroyAllWindows()
