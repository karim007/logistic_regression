
import numpy as np  
from sklearn.datasets import fetch_openml
from lg_one import logistic_regression
import idx2numpy
import gzip


def hot_encoding(labels, digits):
    examples = labels.shape[0]  
    labels = labels.reshape(1, examples)  

    label_new = np.eye(digits)[labels.astype('int32')] 
    label_new = label_new.T.reshape(digits, examples)  
    return label_new

def reshapping_dimension(data):
    return data.reshape((len(data),np.prod(data.shape[1:])))


#MNIST contains 70,000 images of hand-written digits, each 28 x 28 pixels
# data=  idx2numpy.convert_from_file(gzip.open('data/mnist/digits/train-images-idx3-ubyte.gz','r'))
# data_test=  idx2numpy.convert_from_file(gzip.open('data/mnist/digits/t10k-images-idx3-ubyte.gz','r'))
# labels=idx2numpy.convert_from_file(gzip.open('data/mnist/digits/train-labels-idx1-ubyte.gz','r'))
# labels_test=idx2numpy.convert_from_file(gzip.open('data/mnist/digits/t10k-labels-idx1-ubyte.gz','r'))

# data=reshapping_dimension(data).T
# data_test=reshapping_dimension(data_test).T


# labels_test =hot_encoding(labels_test, 10)
# labels =hot_encoding(labels, 10)
# classes=10


#See https://github.com/karim007/converter_grayscale_image_dataset_to_numpy_array to generate this kind of dataset

# data=  np.load('data/numpy/letters/training/data.npy')
# data_test=  np.load('data/numpy/letters/test/data.npy')
# labels= np.load('data/numpy/letters/training/label.npy')
# labels_test=np.load('data/numpy/letters/test/label.npy')
# classes=10

# data=  np.load('data/numpy/asl_alphabet/training/data.npy')
# data_test=  np.load('data/numpy/asl_alphabet/test/data.npy')
# labels= np.load('data/numpy/asl_alphabet/training/label.npy')
# labels_test=np.load('data/numpy/asl_alphabet/test/label.npy')
# classes=29




logistic_regression(data, labels ,data_test ,labels_test ,classes = 10, learning_rate=1, epochs=1000, stochastic_gradient_descent=True )











