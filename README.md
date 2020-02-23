# Logistic regression (single layer)


# Install dependencies
```bash
pip install -r requirements
```

# Select the dataset
Under the folder "data" there is different dataset. Example Mnist: MNIST is a dataset consisting of 70,000 images of handwritten digits. Each training example comes with an associated label (0 to 9) indicating what digit it is. Each digit will be a greyscale with pixel-values from 0 to 255.

You can create your own dataset from sorted pictures with the following library: https://github.com/karim007/converter_grayscale_image_dataset_to_numpy_array


# Run the algorithm

```python
Python main.py
```
## Logistic regression
It is a model that is used to predict the probabilities of the different possible outcomes. In our case the 0-9 digits.

## Softmax
It's a function that takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers. In our case the input will be the wieght * data, on softmax will output a vector with a proabibility for each digit.

## Stochastique gradient descent
Stochastic gradient descent is a method to find the optimal parameter configuration for a machine learning algorithm. It iteratively makes small adjustments to a machine learning network configuration to decrease the error of the network.

## Cross entropy
Measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. 

## Feed forward and Back propagration
Backpropagation is a supervised learning technique for neural networks that calculates the gradient of descent for weighting different variables. It’s short for the backward propagation of errors, since the error is computed at the output and distributed backwards throughout the network’s layers.

In a feedforward network, the information moves in only one direction – forward – from the input nodes, through the hidden nodes (if any) and to the output nodes. There are no cycles or loops in the network 

![schema](/images/fb.jpg)




## Sources

- https://www.researchgate.net/publication/303744090_Forecasting_East_Asian_Indices_Futures_via_a_Novel_Hybrid_of_Wavelet-PCA_Denoising_and_Artificial_Neural_Network_Models
- https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
- https://deepai.org/machine-learning-glossary-and-terms/stochastic-gradient-descent
- https://en.wikipedia.org/wiki/Softmax_function
- https://en.wikipedia.org/wiki/Multinomial_logistic_regression
- https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%202/
- https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
- https://iq.opengenus.org/implementing-cnn-python-tensorflow-mnist-data/
- https://deepai.org/machine-learning-glossary-and-terms/backpropagation
- https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/
