import tensorflow as tf

from unicodedata import name
import tensorflow as tf
import keras.layers as layers
from keras.layers import BatchNormalization, Dense, Reshape, Flatten, Conv1D, Concatenate, Dropout
import tensorflow_addons as tfa
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import datasets, layers, models
import tensorflow_addons as tfa
from tensorflow import keras
import matplotlib.pyplot as plt
import data_processing as d
from sklearn.model_selection import train_test_split
import numpy as np

KERNEL_SIZE = 3
FILTERS = 128
DROPOUT_RATE = 0.3

EPOCHS = 200 # arbitrary

X, y = d.get_encoded_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

def getModel():
    return keras.Sequential(
        [
            layers.Conv1D(FILTERS, KERNEL_SIZE, padding='same', activation='relu', name="layer1", input_shape=X_train.shape[1:]),
            layers.LeakyReLU(),
            layers.Dropout(DROPOUT_RATE),
            layers.Conv1D(FILTERS / 2, KERNEL_SIZE, padding='same', activation='relu', name="layer2"),
            layers.LeakyReLU(),
            layers.Dropout(DROPOUT_RATE),
            layers.Conv1D(FILTERS / 4, KERNEL_SIZE, padding='same', activation='relu', name="layer3"),
            layers.LeakyReLU(),
            layers.Dropout(DROPOUT_RATE),
            # layers.Conv1D(FILTERS, KERNEL_SIZE, padding='same', activation='relu', name="layer4"),
            layers.Dense(13, activation='softmax', name='output_layer')
        ])

def plot(history, epochs, metric, type):
    loss_train = history[metric]
    epochs = range(1,epochs + 1)
    plt.plot(epochs, loss_train, 'g', label=metric)
    plt.title(metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # print(X_train[0], y_train)
    X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1])
    y_train = y_train.reshape(1, y_train.shape[0], y_train.shape[1])
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1])
    
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    
    
    # for i in train_dataset:
    #     print(i)
    model = getModel()
    optimizer = keras.optimizers.Nadam(learning_rate=1e-3, decay=5e-4)
   
    model.compile(optimizer, 'mse', metrics=[keras.metrics.Accuracy(),
                                                # keras.metrics.Precision(), 
                                                # keras.metrics.Recall()
                                                ])

    # model.build()
    # model.summary()
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=EPOCHS
    )
   
    for metric in history.history:
        plot(history.history, EPOCHS, metric, "Training")
   
    print("Evaluate on test data")
    results = model.evaluate(X_train, y_train)
    print("test loss, test acc:", results)
   
    # for metric in results:
    #     plt.plot(metric)
    # plt.show()