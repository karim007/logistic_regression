
import numpy as np  
from sklearn.datasets import fetch_openml
from lg_one import logistic_regression

#MNIST contains 70,000 images of hand-written digits, each 28 x 28 pixels
mnist = fetch_openml('mnist_784')
data, labels = mnist["data"], mnist["target"]

logistic_regression(data, labels, digits = 10, learning_rate=1)











