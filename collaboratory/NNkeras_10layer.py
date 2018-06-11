import load_data as LoadData
import tensorflow as tf
import numpy as np
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  

import time
import pickle
from keras import regularizers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy
# fix random seed for reproducibility
np.random.seed(7)


class NN_10Layer():
    def __init__(self, x_train, y_train, x_test, y_test, input_dim, 
        num_classes, epochs, batch_size, hidden, dropout=0, l2=0, flatten=False):
        #*******************************
        # Assigning  + input size
        #*******************************
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.l2 = l2
        self.flatten = flatten
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden = hidden

    def run_NN(self):
        model = Sequential()
        
        model.add(Dense(self.hidden, input_shape=self.input_dim, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.hidden, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.hidden, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.hidden, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.hidden, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        if self.flatten:
            model.add(Flatten())
        model.add(Dense(self.hidden, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.hidden, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.hidden, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.hidden, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.num_classes, activation='sigmoid'))
        
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        r = model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # evaluate the model
        scores = model.evaluate(self.x_test, self.y_test)
        accuracy = scores[1]*100

        n_epochs = len(r.history['loss'])
        print('Completed %s epochs.\n', n_epochs)
        print('Accuracy: %s ', accuracy)

        return accuracy




