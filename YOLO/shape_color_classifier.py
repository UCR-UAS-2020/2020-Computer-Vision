import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import json
import numpy as np
import os
import cv2
import random
import pickle
from tqdm import tqdm
from proto import *

IMAGE_SIZE = 96
img_path = '../../ImageGenerator/Targets'
target_path = '../../ImageGenerator/Target Data'
json_path = '../../ImageGenerator/Target Data'

EPOCHS = 1
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

data = []
labels = []
lb = LabelBinarizer()

def create_training_data(image_path, json_path, img_size):
    global data
    global labels

    training_data = []
    training_labels = []

    random.seed(27)

    for img in tqdm(os.listdir(image_path)):

        if ( "DS" in os.path.join(image_path, img)):
            print(".DS_STORE DETECTED")
        else:
            image = cv2.imread(os.path.join(image_path, img))
            image = cv2.resize(image, (img_size, img_size))
            image = img_to_array(image)
            training_data.append(image)

            json_file = img[0:-4] + '.json'

            if (json_file != '.DS_S.json'):
                json_dict = json.load(open(os.path.join(json_path, json_file)))
                class_number = Color[json_dict["shape_color"]].value
                training_labels.append(class_number)

                random.shuffle(training_data)
                random.shuffle(training_labels)

    # print(training_data)
    data = np.array(training_data, dtype="float") / 255.0
    labels = np.array(training_labels)

    print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

    labels = lb.fit_transform(labels)

# classes should be 10 for 10 models
def create_model(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

def model_training_data():
    (xTrain, xTest, yTrain, yTest) = train_test_split(data, labels, test_size=0.2, random_state=27)

    # Adds more data by rotating/shifting/shearing etc.
    # DON'T USE ON WEAK COMPUTERS
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    model = create_model(96, 96, 3, len(lb.classes_))
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("TRAINING NETWORK...")
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), steps_per_epoch=len(xTrain) // BS, epochs=EPOCHS, verbose=1)

    print("SAVING MODEL...")
    model.save('m.model', save_format="h5")
    f = open(r"Training_Data/color.pickle", "wb")
    f.write(pickle.dumps(lb))
    f.close()

def test_model(imagePath):

    image = cv2.imread(imagePath)

    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    model = load_model('m.model')
    lb = pickle.loads(open('Training_Data/color.pickle', 'rb').read())

    print("Classifying Image...")
    prediction = model.predict(image)[0]
    idx = np.argmax(prediction)
    label = lb.classes_[idx]

    print("RESULT:")
    print(label)


# Main function used for testing purposes
# if __name__ == '__main__':
#     training_data = create_training_data(img_path, json_path, 96)
#     model_training_data()
#     test_model(os.path.join(img_path, '0.png'))

