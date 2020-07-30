from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

pickle_in = open("./Training_Data/shape_X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("./Training_Data/shape_y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128, 256]
conv_layers = [1, 2]
dense_sizes = [32, 64, 128, 256]
# dense_layers = [1]
# layer_sizes = [128]
# conv_layers = [2]


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            for dense_size in dense_sizes:
                NAME = "{}-conv-{}-nodes-{}-dense-{}-nodes-{}".format(conv_layer,
                                                                      layer_size,
                                                                      dense_layer,
                                                                      dense_size,
                                                                      int(time.time()))
                print(NAME)

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
                model.add(Activation('sigmoid'))

                tensorboard = TensorBoard(log_dir=r"shape_classifier_logs\{}".format(NAME))

                model.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'],
                              )

                model.fit(X, y,
                          batch_size=32,
                          epochs=5,
                          validation_split=0.3,
                          callbacks=[tensorboard])
