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

# Parameters
learning_rate = 0.1
num_steps = 100
batch_size = 128
display_step = 100
# Number of epochs = size of x_train/batch_size

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 256 # 3rd layer number of neurons
n_hidden_4 = 256 # 1st layer number of neurons
n_hidden_5 = 256 # 2nd layer number of neurons
n_hidden_6 = 256 # 3rd layer number of neurons
n_hidden_7 = 256 # 1st layer number of neurons
n_hidden_8 = 256 # 2nd layer number of neurons
n_hidden_9 = 256 # 3rd layer number of neurons
n_hidden_10 = 256 # 3rd layer number of neurons

class NN_10Layer():
    def __init__(self, x_train, y_train, x_test, y_test, input_dim, num_classes, dropout=0, l2=0):
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

    def run_NN(self):
        model = Sequential()
        model.add(Dense(n_hidden_1, input_dim=self.input_dim, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(n_hidden_2, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(n_hidden_3, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(n_hidden_4, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(n_hidden_5, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(n_hidden_6, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(n_hidden_7, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(n_hidden_8, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(n_hidden_9, kernel_regularizer=regularizers.l2(self.l2), activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.num_classes, activation='sigmoid'))
        
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        model.fit(self.x_train, self.y_train, epochs=150, batch_size=10)

        # evaluate the model
        scores = model.evaluate(self.x_test, self.y_test)
        accuracy = scores[1]*100
        return accuracy

def make_dropout_data_points(x_train, y_train, x_test, y_test, input_dim, num_classes, dropout, dir='./result_data/'):
    baseline_time = []
    time_dropout = []
    time_l2 = []
    accuracy_dropout = []
    accuracy_l2 = []

    for d in dropout:
        nn10layer = NN_10Layer(x_train, y_train, x_test, y_test, input_dim, num_classes, dropout=0, l2=0)
        start = time.time()
        result = nn10layer.run_NN()
        end = time.time()
        time_dropout.append(float(end)-float(start))
        accuracy_dropout.append(result)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = dir + 'accuracy_dropout_' + timestr
    with open(filename, 'wb') as fp:
        pickle.dump(accuracy_dropout, fp)
    filename = dir + 'dropout_' + timestr
    with open(filename, 'wb') as fp:
        pickle.dump(dropout, fp)
    filename = dir + 'time_dropout_' + timestr
    with open(filename, 'wb') as fp:
        pickle.dump(time_dropout, fp)

    return time_dropout, accuracy_dropout

def make_l2_data_points(x_train, y_train, x_test, y_test, input_dim, num_classes, l2, dir='./result_data/'):
    time_dropout = []
    time_l2 = []
    accuracy_dropout = []
    accuracy_l2 = []

    for l in l2:
        nn10layer = NN_10Layer(x_train, y_train, x_test, y_test, input_dim, num_classes, dropout=0, l2=0)
        start = time.time()
        result = nn10layer.run_NN()
        end = time.time()
        time_dropout.append(float(end)-float(start))
        accuracy_dropout.append(result)

    # Writing data to files
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = dir + 'accuracy_l2_' + timestr
    with open(filename, 'wb') as fp:
        pickle.dump(accuracy_l2, fp)
    filename = dir + 'l2_' + timestr
    with open(filename, 'wb') as fp:
        pickle.dump(l2, fp)
    filename = dir + 'time_l2_' + timestr
    with open(filename, 'wb') as fp:
        pickle.dump(time_l2, fp)

    return time_l2, accuracy_l2
 

def make_dropout_graphs(dropout, accuracy_dropout, time_dropout,
    dataset_name='Mnist', time_stamp='20180520-132113', data_load = False):
    """
    Creates graphs for dropout
    """

    # Loading in data
    if data_load:
        dropout = pickle.load(open("dropout_" + time_stamp, "rb"))
        accuracy_dropout = pickle.load(open("accuracy_dropout_" + time_stamp, "rb"))
        time_dropout = pickle.load(open("time_dropout_" + time_stamp, "rb"))

    # Dropout graphs
    #horiz_accuracy_data = np.array([np.mean(baseline_accuracy) for i in range(len(dropout))])
    #plt.plot(dropout, horiz_accuracy_data, label='No Dropout')
    plt.scatter(dropout, accuracy_dropout, label='Dropout')
    plt.title (dataset_name + ' Accuracy vs. Dropout')
    plt.xlabel('Dropout Layer %')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./result_data/graphs/' + timestr + data_name + '_accuracy.png')

    #horiz_time_data = np.array([np.mean(baseline_time) for i in range(len(dropout_dropout))])
    #plt.plot(horiz_time_data, time_list, label='No dropout')
    plt.scatter(dropout_inverse, time_dropout_list, label='Dropout')
    plt.title (dataset_name + ' Time vs. Dropout')
    plt.xlabel('Dropout Layer %')
    plt.ylabel('Time')
    plt.legend()
    plt.savefig('./result_data/graphs/' + timestr + data_name + '_time.png')

def make_l2_graphs(l2, accuracy_l2, time_l2,
    dataset_name='Mnist', time_stamp='20180520-132113', data_load = False):
    """
    Creates l2 graphs
    """

    # Loading in data
    if data_load:
        l2 = pickle.load(open("l2_" + time_stamp, "rb"))
        accuracy_l2 = pickle.load(open("accuracy_dropout_" + time_stamp, "rb"))
        time_l2 = pickle.load(open("time_dropout_" + time_stamp, "rb"))

    # L2 Graphs
    #horiz_accuracy_data = np.array([np.mean(baseline_accuracy) for i in range(len(l2))])
    #plt.plot(dropout, horiz_accuracy_data, label='No Dropout')
    plt.scatter(l2, accuracy_l2, label='Dropout')
    plt.title (dataset_name + ' Accuracy vs. Dropout')
    plt.xlabel('Dropout Layer %')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./result_data/graphs/' + timestr + data_name + '_accuracy.png')

    #horiz_time_data = np.array([np.mean(baseline_time) for i in range(len(l2))])
    #plt.plot(horiz_time_data, baseline_time, label='No dropout')
    plt.scatter(l2, time_l2, label='Dropout')
    plt.title (dataset_name + ' Time vs. Dropout')
    plt.xlabel('Dropout Layer %')
    plt.ylabel('Time')
    plt.legend()
    plt.savefig('./result_data/graphs/' + timestr + data_name + '_time.png')

def frange(start, stop, step):
    i = start
    result = []
    while i < stop:
        result.append(i)
        i += step
    return result

def main():
    #x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_pima()
    #x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_iris()
    x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_titanic()
    #x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_wine()

    dropout = frange(0.1, 0.3, 0.1)
    l2 = [0.00001, 0.0001, 0.001, 0.01, 0.1]

    time_dropout, accuracy_dropout = make_dropout_data_points(x_train, y_train, x_test, y_test, input_dim, num_classes, dropout)
    time_l2, accuracy_l2 = make_l2_data_points(x_train, y_train, x_test, y_test, input_dim, num_classes, l2)
    make_dropout_graphs(dropout, time_dropout, accuracy_dropout, dataset_name='Pima')
    make_l2_graphs(l2, time_l2, accuracy_l2, dataset_name='Pima')

main()







