#Libaries 
from mnist import MNIST
import numpy as np
import random as rand
import gzip

#Standard Libaries for model trainging and testing

from keras.utils.np_utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import gzip
import time

#Data_loading 
#this is the original downloaded data.it is not used here you can also try this

mnist=MNIST('./Mnist1/')
mnist.gz= True
x_train,y_train=mnist.load_training()
x_val,y_val=mnist.load_testing()

#Data_spliting 

x_train_1 = np.asarray(x_train).astype(np.float32)
y_train_1 = np.asarray(y_train).astype(np.int32)
x_val_1= np.asarray(x_val).astype(np.float32)
y_val_1= np.asarray(y_val).astype(np.int32)

#here the data loaded for this program 
x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
y = to_categorical(y)

#data_splitted 
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)
start_index=0


class DeepNeuralNetwork():
  #Intialization of the class with input values 
    def __init__(self, sizes, epochs=30, l_rate=0.001,B_size=64):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.B_size=B_size

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        hidden_3=self.sizes[3]
        output_layer=self.sizes[4]
        #np.random.seed(43)

        params = {
            
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. /hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(hidden_3,hidden_2) * np.sqrt(1./hidden_3),
            'W4':np.random.randn(output_layer, hidden_3) * np.sqrt(1. / output_layer)
            
            }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])
        
        #hidden layer 2 to  hidden layer 3
        params['Z3']=np.dot(params["W3"],params["A2"])
        params["A3"]=self.sigmoid(params['Z3'])

        # hidden layer 3 to output layer
        params['Z4'] = np.dot(params["W4"], params['A3'])
        params['A4'] = self.softmax(params['Z4'])

        return params['A4']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.
                  
                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}

        # Calculate W4 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z4'], derivative=True)
        change_w['W4'] = np.outer(error, params['A3'])
        
        #claculate W3 update 
        error=np.dot(params["W4"].T,error)*self.sigmoid(params['Z3'])
        change_w["W3"]=np.outer(error, params["A2"])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''
        
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred== np.argmax(y))
            
        
        return np.mean(predictions)
    
    
    @staticmethod
    def update_batchSize(x_train2,Batch_size):
        global start_index
        #print(Batch_size)
        finish_index=start_index+Batch_size
        #print("here:",len(x_train2),n,s)\
        #print("here1:",finish_index,start_index)
        x_train3=x_train2[(start_index):(finish_index)]
        start_index=finish_index
        #print(len(x_train3))
        return (x_train3)
    def up():
        return 1   
    
    def  train(self, x_train, y_train, x_val, y_val):
        #update_batchSize(x_train,64)
        start_time = time.time()
        Batch=self.B_size 
        #print(Batch)
        for iteration in range(self.epochs):
            x_train1=dnn.update_batchSize(x_train,Batch)
            y_train1=dnn.update_batchSize(y_train,Batch)
            x_val1=dnn.update_batchSize(x_val,Batch)
            y_val1=dnn.update_batchSize(y_val,Batch)
            
            for x,y in zip(x_train1, y_train1):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            
                       
            train_accuracy=self.compute_accuracy(x_train1,y_train1)
            print('Epoch:{0} , Time:{1:.2f}s,Train_Accuracy:{2:.2f}%'.format(
                iteration+1,time.time()- start_time, train_accuracy*100 ))
            
            test_accuracy = self.compute_accuracy(x_val1, y_val1)
            print('Epoch: {0}, Time Spent: {1:.2f}s,Test_Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, test_accuracy * 100
            ))
            
#update_batchSize(x_train,ytrain)
#for x, y in zip(x_val, y_val):
    #output = self.forward_pass(x)
    #print(output)
            
dnn = DeepNeuralNetwork(sizes=[784, 64, 35,16, 10])
#dnn.update_batchSize(x_train,64)
dnn.train(x_train, y_train, x_val, y_val)

