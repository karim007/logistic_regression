import numpy as np   
from sklearn.metrics import classification_report, confusion_matrix

class logistic_regression:
    def __init__(self, data, labels, digits, learning_rate):
        
        self.data, self.labels_hot_encoded_training, self.data_test, self.labels_hot_encoded_test = self.split_training_test(data, label_new = self.hot_encoding(labels, digits))
        ##28*28
        img_format_size = self.data.shape[0]
        ## 60 000
        images_total = self.data.shape[1]

        ### scale to keep our gradients manageable: each image pixel is greyscale with pixel-values from 0 to 255. 
        self.data=self.data/255
        self.data_test=self.data_test/255

        ## step size at each iteration while moving toward a minimum of a loss function
        self.learning_rate=learning_rate
        ## the number of samples that will be propagated through the network
        self.batch_size = 128
        self.batches = -(- images_total // self.batch_size)

        ## weight: normal distribution center in 0
        self.W1 =np.random.randn(img_format_size, digits) 
        # One epoch is when an entiree dataset is passed forward and backward through the neural network.
        self.epochs=10
        self.learn()


    ## activation function that turns the ouptut of the first layer into probabilities that sum to one 
    def softmax(self, first_layer_input):
        return np.exp(first_layer_input) / np.sum(np.exp(first_layer_input), axis=0)

    def feed_forward(self, X):

        cache = {}
        cache["Z1"] = np.matmul(self.W1.T, X) 
        cache["A1"] = self.softmax(cache["Z1"])     
        return cache
    
    def back_propagate(self, X, Y, cache, m_batch):

        dZ = cache["A1"] - Y
        dW1 = (1./m_batch) * np.matmul(X, dZ.T)
        return dW1



    def hot_encoding(self, labels, digits):
        examples = labels.shape[0]  
        labels = labels.reshape(1, examples)  

        label_new = np.eye(digits)[labels.astype('int32')] 
        label_new = label_new.T.reshape(digits, examples)  
        return label_new

    def split_training_test(self, data, label_new):
        m = data.shape[0] -10000

        data_train, data_test = data[:m].T, data[m:].T
        labels_train, labels_test = label_new[:,:m], label_new[:,m:]

        shuffle_index = np.random.permutation(m)

        data_train, labels_train = data_train[:, shuffle_index], labels_train[:, shuffle_index]

        return data_train, labels_train, data_test, labels_test

    def compute_loss(self, predicted):

        L_sum = np.sum(np.multiply(self.labels_hot_encoded_training, np.log(predicted)))
        total_image = self.labels_hot_encoded_training.shape[1]
        cost = -(1./total_image) * L_sum
        
        return cost

    def learn(self):
        #Start training
        for i in range(self.epochs):

            for j in range(self.batches):

                begin = j * self.batch_size
                end = min(begin + self.batch_size, self.data.shape[1] - 1)

                batch_data = self.data[:, begin:end]
                batch_label = self.labels_hot_encoded_training[:, begin:end]
                m_batch = end - begin

                cache = self.feed_forward(batch_data)
                grad = self.back_propagate(batch_data, batch_label, cache, m_batch)


                self.W1  = self.W1 - self.learning_rate * grad

            cache = self.feed_forward(self.data)
            train_cost = self.compute_loss( cache["A1"])
            cache = self.feed_forward(self.data_test)
           
            print("Epoch {}: training cost = {}".format(i+1 ,train_cost))
        
        print("Training Done.")
        
        self.test()



    def test(self):

        cache = self.feed_forward(self.data_test)
        predictions = np.argmax(cache["A1"], axis=0)
        labels = np.argmax(self.labels_hot_encoded_test, axis=0)

        print(classification_report(predictions, labels))

  