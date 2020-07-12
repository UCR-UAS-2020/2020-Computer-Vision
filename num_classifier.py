from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import KFold


def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100,
                    activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(dataX):
        model = define_model()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        history = model.fit(trainX,
                            trainY,
                            epochs=10,
                            batch_size=32,
                            validation_data=(testX, testY),
                            verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
    return scores, histories


def pre_run_prep():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    return trainX, trainY, testX, testY


def test_harness():
    trainX, trainY, testX, testY = pre_run_prep()
    trainX, testX = pre_run_prep()
    scores, histories = evaluate_model(trainX, trainY)

    model = define_model()
    model.fit(trainX,
              trainY,
              epochs=10,
              batch_size=32,
              verbose=0)
    model.save('final_model.h5')


def load_run_model():
    trainX, trainY, testX, testY = pre_run_prep()
    trainX, testX = pre_run_prep()
    model = load_model('final_model.h5')
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))


if __name__ == "__main__":
    test_harness()