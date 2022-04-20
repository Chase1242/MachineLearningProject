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

KERNEL_SIZE = 5

X, y = d.get_encoded_data()
#y = d.get_y()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

#y_train = d.standarize_predictions(y_train).numpy().squeeze()
LAYERS = 52 # arbitrary number for now
EPOCHS = 500 # arbitrary

# TODO: define all the testing data

def getModel():
    # all three layers are arbitrarily chosen, don't know input shape
    return keras.Sequential(
        [
            layers.Conv1D(64, KERNEL_SIZE, padding='same', activation='relu', name="layer0", input_shape=X_train.shape[1:]),
            layers.Dropout(.3),
            layers.Conv1D(64, KERNEL_SIZE, padding='same', activation='relu', name="layer1"),
            layers.Dropout(.3),
            layers.Conv1D(64, KERNEL_SIZE, activation='relu', padding='same'),
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
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    
    # for i in train_dataset:
    #     print(i)
    model = getModel()
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
   
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