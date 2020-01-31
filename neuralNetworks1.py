# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Any results you write to the current directory are saved as output.

# %% [code]
## start coding here
class Neural_Net():
    def __init__(self):
        np.random.seed(1) #make random actually random
        #make a [3,1] array filled with random numbers between [-1,1]
        self.weights = 2 * np.random.random((3,1)) - 1 
        
    #sigmoid function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x)) 
    
    #The derivative of sigmoid, uses chain rule
    def sigmoid_derivative(self, x):
        return x * (1-x)
    
    #The trining function
    def train(self, training_input, training_output, training_iteration):
        for _ in range(training_iteration):
            output = self.think(training_input)
            error = training_output - output
            # T - Transposes
            #np.dot finds the dot product of two matrices
            adjustments = np.dot(training_input.T, error * self.sigmoid_derivative(output)) #Get dot product
            self.weights += adjustments
            
    #prediction function
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output
    
# Python's main function
if __name__ == "__main__":
    neural_network = Neural_Net()

    training_input = np.array([[0,0,1],
                               [1,1,1],
                               [1,0,0],
                               [0,1,0]])

    training_output = np.array([[0,1,1,0]]).T

    #Iterate to train
    neural_network.train(training_input, training_output, 10000)

    #Get input for testing
    print("Enter 0 or 1")
    A = input("First Number: ")
    B = input("Second Number: ")
    C = input("Third Number: ")        

    print("New input is: ", A, B, C)
    print("Predicted output is: ")
    print(neural_network.think(np.array([A,B,C])))


        