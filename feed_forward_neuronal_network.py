"""
# Created by Lukas on 09.12.2017#
Topic:
Task:
Description
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import class_weight
import numpy as np

def create_and_train(INFOS):
    X = INFOS["X"]
    y = INFOS["y"]

    X_14 = INFOS["X_14"]
    y_14 = INFOS["y_14"]

    X_15 = INFOS["X_eval"]
    y_15 = INFOS["y_eval"]


    y = keras.utils.to_categorical(y, num_classes=3)
    y_14 = keras.utils.to_categorical(y, num_classes=3)
    y_15 = keras.utils.to_categorical(y, num_classes=3)

    scaler = QuantileTransformer()
    X = scaler.fit_transform(X)
    X_14 = scaler.transform(X_14)
    X_15 = scaler.transform(X_15)
    print(class_weight.compute_class_weight('balanced', np.unique(INFOS["y"]),INFOS["y"]))

    # Model Parameter

    my_optimizer = RMSprop(lr=0.01)
    loss_func = "categorical_crossentropy"

    input_layer = "sigmoid"
    output_layer = "softmax"
    activation_func = "hard_sigmoid"
    nr_hidden_layer = 2  # 10

    layer_size = X.shape[1]
    nr_hidden_layer_nodes = X.shape[1]

    epochs = 100  # 50
    batch_size = 1000 # 50
## Callbacks
    early_stopping_mentor = EarlyStopping(monitor="categorical_crossentropy", patience=4, mode="auto")

    reduce_lr = ReduceLROnPlateau(monitor='categorical_crossentropy', factor=0.5,
                              patience=1, min_lr=0.0001)

    model = Sequential()
    model.add(Dense(nr_hidden_layer_nodes, input_dim=layer_size, activation=input_layer))

    for _ in range(nr_hidden_layer):
        model.add(Dense(int(nr_hidden_layer_nodes), activation=activation_func))
        model.add(Dropout(0.1))

    model.add(Dense(3, activation=output_layer))

    model.compile(loss=loss_func, optimizer=my_optimizer,
                  metrics=["categorical_crossentropy", "categorical_accuracy"])
    # # Fit the model

    model.fit(X,
              y,
              callbacks=[early_stopping_mentor, reduce_lr],
              epochs=epochs,
              batch_size=batch_size,
              verbose=1,
              class_weight=[1.56923407, 0.57954455, 1.56923407],

          )



    def mk_output(output):
        signal = []
        for step in output:
            step = list(step)
            one_hot = max(step)

            if step.index(one_hot) == 0:
                signal.append(0)
            elif step.index(one_hot) == 1:
                signal.append(1)
            elif step.index(one_hot) == 2:
                signal.append(-1)
            else:
                print("error", one_hot, np.where(output == one_hot))
        return signal
    predictions = {"X" : mk_output(model.predict(X)),
                   "X_14": mk_output(model.predict(X_14)),
                   "X_15": mk_output(model.predict(X_15))}


    return predictions
