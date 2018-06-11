from NNkeras_10layer import NN_10Layer
import load_data as LoadData
import numpy as np
import time
import pickle
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  

def make_dropout_data_points(x_train, y_train, x_test, y_test, input_dim, 
    num_classes, epochs, batch_size, hidden, dropout, dir='./result_data/', input_type='single', flatten=False):
    baseline_time = []
    time_dropout = []
    accuracy_dropout = []

    for d in dropout:
        nn10layer = NN_10Layer(x_train, y_train, x_test, y_test, input_dim, num_classes, 
            epochs, batch_size, hidden, dropout=0, l2=0, flatten=False)
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

def make_l2_data_points(x_train, y_train, x_test, y_test, input_dim, 
    num_classes, epochs, batch_size, hidden, l2, dir='./result_data/', input_type='single', flatten=False):
    time_l2 = []
    accuracy_l2 = []

    for l in l2:
        nn10layer = NN_10Layer(x_train, y_train, x_test, y_test, input_dim, num_classes, 
            epochs, batch_size, hidden, dropout=0, l2=0, flatten=False)
        start = time.time()
        result = nn10layer.run_NN()
        end = time.time()
        time_l2.append(float(end)-float(start))
        accuracy_l2.append(result)

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
 

def make_dropout_graphs(dropout, time_dropout, accuracy_dropout, 
    dataset_name='Mnist', time_stamp='20180520-132113', data_load = False, dir='./result_data/graphs/'):
    """
    Creates graphs for dropout
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Loading in data
    if data_load:
        dropout = pickle.load(open("dropout_" + time_stamp, "rb"))
        accuracy_dropout = pickle.load(open("accuracy_dropout_" + time_stamp, "rb"))
        time_dropout = pickle.load(open("time_dropout_" + time_stamp, "rb"))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    # Dropout graphs
    #horiz_accuracy_data = np.array([np.mean(accuracy_dropout[0]) for i in range(len(dropout))])
    #ax1.plot(dropout, horiz_accuracy_data, label='No Dropout')
    ax1.scatter(dropout, accuracy_dropout, label='Dropout')
    ax1.set_title (dataset_name + ' Accuracy vs. Dropout')
    ax1.set_xlabel('Dropout Layer %')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    fig.savefig(dir + timestr + dataset_name + '_dropout_accuracy.png')

    ax2 = fig.add_subplot(122)
    #horiz_time_data = np.array([np.mean(baseline_time) for i in range(len(dropout_dropout))])
    #plt.plot(horiz_time_data, time_list, label='No dropout')
    ax2.scatter(dropout, time_dropout, label='Dropout')
    ax2.set_title (dataset_name + ' Time vs. Dropout')
    ax2.set_xlabel('Dropout Layer %')
    ax2.set_ylabel('Time')
    ax2.legend()
    fig.savefig(dir + timestr + dataset_name + '_dropout_time.png')

def make_l2_graphs(l2, time_l2, accuracy_l2,
    dataset_name='Mnist', time_stamp='20180520-132113', data_load = False, dir='./result_data/graphs/'):
    """
    Creates l2 graphs
    """

    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Loading in data
    if data_load:
        l2 = pickle.load(open("l2_" + time_stamp, "rb"))
        accuracy_l2 = pickle.load(open("accuracy_dropout_" + time_stamp, "rb"))
        time_l2 = pickle.load(open("time_dropout_" + time_stamp, "rb"))

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(121)
    # L2 Graphs
    #horiz_accuracy_data = np.array([np.mean(baseline_accuracy) for i in range(len(l2))])
    #plt.plot(dropout, horiz_accuracy_data, label='No Dropout')
    ax3.scatter(l2, accuracy_l2, label='L2')
    ax3.set_title (dataset_name + ' Accuracy vs. L2')
    ax3.set_xlabel('L2 Layer %')
    ax3.set_xscale('log')
    ax3.set_xlim([0.00001, 1.0])
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    fig2.savefig(dir + timestr + dataset_name + '_accuracy.png')
    
    ax4 = fig2.add_subplot(122)
    #plt.savefig(dir + timestr + dataset_name + '_accuracy.png')
    #horiz_time_data = np.array([np.mean(baseline_time) for i in range(len(l2))])
    ax4.scatter(l2, time_l2, label='L2')
    ax4.set_title (dataset_name + ' Time vs. L2')
    ax4.set_xlabel('L2 Layer %')
    ax4.set_xlim([0.00001, 1.0])
    ax4.set_xscale('log')
    ax4.set_ylabel('Time')
    ax4.legend()
    fig2.savefig(dir + timestr + dataset_name + '_time.png')


def frange(start, stop, step):
    i = start
    result = []
    while i < stop:
        result.append(i)
        i += step
    return result

def main():
    #print('\n\n\n\n\n')
    x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_pima()
    #x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_iris()
    #x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_titanic()
    #x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_wine()
    #x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_cifar()
    #x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_mnist()
    #x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_imdb()
    #x_train, y_train, x_test, y_test, input_dim, num_classes = LoadData.load_boston()

    dropout = frange(0, 0.99, 0.02)
    l2 = [0.00001, 0.0001, 0.001, 0.01, 0.1]

    # Parameters
    epochs=100
    batch_size = 32
    # Number of epochs = size of x_train/batch_size

    # Network Parameters
    n_hidden = 24 # 1st layer number of neurons


    #time_dropout, accuracy_dropout = make_dropout_data_points(x_train, y_train, x_test, 
    #    y_test, input_dim, num_classes, epochs, batch_size, n_hidden, dropout, dir='./result_data/', flatten=False)
    #time_l2, accuracy_l2 = make_l2_data_points(x_train, y_train, x_test, 
    #    y_test, input_dim, num_classes, epochs, batch_size, n_hidden, l2, dir='./result_data/', flatten=False)

    #pickle.load('20180520-132701none_accuracy.')
    make_dropout_graphs(dropout, time_dropout, accuracy_dropout, dataset_name='Pima')
    make_l2_graphs(l2, time_l2, accuracy_l2, dataset_name='Pima')

main()

