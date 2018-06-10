import numpy as np
import pandas as pd

def load_mnist():
	# Import MNIST data
	# len = 55000
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	x_train = mnist.train.images
	y_train = mnist.train.labels
	x_test = mnist.test.images
	y_test = mnist.test.labels

	input_dim = 784 # MNIST data input (img shape: 28*28)
	num_classes = 10 # MNIST total classes (0-9 digits)
	return x_train, y_train, x_test, y_test, input_dim, num_classes

def load_titanic():

	import tflearn
	import numpy as np

	#************************************
	# Download the Titanic dataset
	# len = 1309
	#************************************
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

	input_dim = 6
	num_classes = 2

	return x_train, y_train, x_test, y_test, input_dim, num_classes

def load_imdb():

	#************************************
	# Download the IMDB Dataset
	#************************************
	from tflearn.datasets import imdb

	import keras 
	input_dim=1000 # only use top 1000 words
	num_classes=2   # word index offset

	train,test = keras.datasets.imdb.load_data()
	x_train, y_train = train
	x_test, y_test = test

	return x_train, y_train, x_test, y_test, input_dim, num_classes


def load_pima():
	#************************************
	# Download PIMA dataset
	# len = 768
	#************************************

	# load pima indians dataset
	dataset = np.loadtxt("/Users/rachael/Documents/Yaser_research/scripts/collaboratory/data/pima-indians-diabetes.csv", delimiter=",")
	# split into input (X) and output (Y) variables
	x_train = dataset[:,0:8]
	y_train = dataset[:,8]
	x_test = x_train
	y_test = y_train
	input_dim = 8
	num_classes = 1
	return x_train, y_train, x_test, y_test, input_dim, num_classes

def load_iris():
	#************************************
	# Download Iris dataset
	# len = 150
	#************************************
	from sklearn import datasets

	iris = datasets.load_iris()
	x_train = iris.data
	y_train = iris.target
	x_test = x_train
	y_test = y_train

	input_dim = 4
	num_classes = 1

	return x_train, y_train, x_test, y_test, input_dim, num_classes


def load_wine():
	from sklearn import preprocessing
	#************************************
	# Download wine dataset
	# len = 1,599
	#************************************
	dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
	data = pd.read_csv(dataset_url, sep=';')

	y_train = data.quality
	x_train = data.drop('quality', axis=1)
	x_train_scaled = preprocessing.scale(x_train)
	x_test = x_train_scaled
	y_test = y_train

	input_dim = 11
	num_classes = 1

	return x_train, y_train, x_test, y_test, input_dim, num_classes

def load_cifar():
	from keras.datasets import cifar10

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	input_dim = 50000
	num_classes = 10

	return x_train, y_train, x_test, y_test, input_dim, num_classes









