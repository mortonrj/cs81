import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils

def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

def load_mnist():
    """
        Download MNIST dataset
        len = 55000
    """
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels


    input_dim = (784,) # MNIST data input (img shape: 28*28)
    num_classes = 10 # MNIST total classes (0-9 digits)
    return x_train, y_train, x_test, y_test, input_dim, num_classes
    

    """
    from keras.datasets import mnist

    num_classes = 10
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # input image dimensions
    img_rows, img_cols = 28, 28 
    
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_dim = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('\n\n\n')
    print(input_dim)
    print(num_classes)
    return x_train, y_train, x_test, y_test, input_dim, num_classes
    """
def load_boston():
    from sklearn.datasets import load_boston
    boston = load_boston()
    x_train = boston.data
    y_train = boston.target
    x_test = x_train
    y_test = y_train


    print('\n\n\n\n')
    print(x_train.shape)
    print('\n\n\n\n')

    input_dim = (13,)
    num_classes = 1

    return x_train, y_train, x_test, y_test, input_dim, num_classes

def load_titanic():

    import tflearn
    import numpy as np

    """
        Download Titanic dataset
        len = 1309
    """
    from tflearn.datasets import titanic
    titanic.download_dataset('titanic_dataset.csv')

    # Load CSV file, indicate that the first column represents labels
    from tflearn.data_utils import load_csv
    data, titanic_labels = load_csv('titanic_dataset.csv', target_column=0,
                            categorical_labels=True, n_classes=2)

    # Preprocessing function
    def preprocess(passengers, columns_to_delete):
        # Sort by descending id and delete columns
        for column_to_delete in sorted(columns_to_delete, reverse=True):
            [passenger.pop(column_to_delete) for passenger in passengers]
        for i in range(len(passengers)):
            # Converting 'sex' field to float (id is 1 after removing labels column)
            passengers[i][1] = 1. if passengers[i][1] == 'female' else 0.
        return np.array(passengers, dtype=np.float32)

    # Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
    to_ignore=[1, 6]

    # Preprocess data
    x_train = preprocess(data, to_ignore)
    y_train = titanic_labels

    x_test = x_train
    y_test = y_train

    input_dim = (6,)
    num_classes = 2

    return x_train, y_train, x_test, y_test, input_dim, num_classes

def load_imdb():

    """
        Download IMDB dataset

    """
    from tflearn.datasets import imdb
    from keras.preprocessing import sequence as prep


    import keras 
    input_dim=1000 # only use top 1000 words
    num_classes=1   # word index offset

    top_words = 5000
    test_split = 0.33
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()
    # pad dataset to a maxumum review length in words
    max_words = 500
    input_dim = (max_words,)
    x_train = prep.pad_sequences(x_train, maxlen=max_words)
    x_test = prep.pad_sequences(x_test, maxlen=max_words)

    return x_train, y_train, x_test, y_test, input_dim, num_classes


def load_pima():
    """
        Download PIMA dataset
        len = 768

        recommended:
        epochs = 150
        batch_size = 10
        hidden = 10
    """

    # load pima indians dataset
    dataset = np.loadtxt("/Users/rachael/Documents/Yaser_research/scripts/collaboratory/data/pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    x_train = dataset[:,0:8]
    y_train = dataset[:,8]
    x_test = x_train
    y_test = y_train
    input_dim = (8,)
    num_classes = 1

    return x_train, y_train, x_test, y_test, input_dim, num_classes

def load_iris():
    """
        Download Iris dataset
        len = 150

        recommended:
        epochs = 10
        batch_size=2
    """

    from sklearn import datasets
    import seaborn as sns
    from sklearn.cross_validation import train_test_split

    iris = sns.load_dataset("iris")
    X=iris.values[:,:4]
    y=iris.values[:,4]

    x_train, x_test,y_train,y_test=train_test_split(X,y,train_size=0.5,random_state=1)
    y_train_ohe=one_hot_encode_object_array(y_train)
    y_test_ohe=one_hot_encode_object_array(y_test)

    input_dim = (4, )
    #num_classes = (len(x_train), 3)
    num_classes = 3
    return x_train, y_train_ohe, x_test, y_test_ohe, input_dim, num_classes


def load_wine():
    from sklearn import preprocessing
    """
        Download wine dataset
        length = 1,599
     
        recommended:
        epochs = 100
        batch_size = 32
    """
    dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(dataset_url, sep=';')

    y_train = data.quality
    x_train = data.drop('quality', axis=1)
    x_train_scaled = preprocessing.scale(x_train)
    x_test = x_train_scaled
    y_test = y_train

    y_train_ohe=one_hot_encode_object_array(y_train)
    y_test_ohe=one_hot_encode_object_array(y_test)

    input_dim = (11, )
    num_classes = 6

    return x_train, y_train_ohe, x_test, y_test_ohe, input_dim, num_classes

def load_cifar():
    """
    Download cifar dataset
    length = 50,000

    recommended:
    epochs = 100
    batch_size = 32
    """
    from keras.datasets import cifar10

    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    input_dim = x_train.shape[1:]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train_ohe=one_hot_encode_object_array(y_train)
    y_test_ohe=one_hot_encode_object_array(y_test)

    print('\n\n\n')
    print(input_dim)
    print(x_train.shape)
    print(y_train.shape)
    print(y_train_ohe.shape)


    return x_train, y_train, x_test, y_test, input_dim, num_classes








