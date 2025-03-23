import numpy as np 
import random 
import math 
from activation import get_activation_function

class Layer: 
    def __init__(self, neurons): 
        self.built = False 
        self.neurons = neurons

    def create(self, inputWeight: np.array): 
        self.inputWeight = inputWeight
        self.built = True

    def call(self, inputs, *args, **kwargs): 
        return inputs

    def get_neuron(self): 
        return self.neurons
    
    def random_weight(self, shape): 
        weights = [] 
        for i in range(shape[0] + 1): 
            weights.append([])
            for j in range(shape[1]): 
                weights[i].append(random.uniform(-math.sqrt(6)/math.sqrt(shape[0] + shape[1]), math.sqrt(6)/math.sqrt(shape[0] + shape[1])))
        return weights


class Dense(Layer): 
    def __init__(self, neurons, activation=None, inputShape=None, inputWeight:np.array=None):
        super().__init__(neurons)
        self.neurons = neurons
        self.activation = get_activation_function(activation)
        self.inputShape = inputShape
        self.inputWeight = inputWeight
        self.create()
    
    def __repr__(self):
        return ''.join([
            'Layer: \n',
            f'activation = {self.activation.__name__}\n',
            f'weights \n = {self.weights}\n',
            f'bias = {self.bias}\n'
        ])
    
    def create(self):
        if self.inputShape is not None: 
            weigthsArr = np.array(self.random_weight(shape=(self.inputShape[0], self.neurons)))
        else:
            weigthsArr = np.array(self.inputWeight)
        
        self.weights = weigthsArr[1:]
        self.bias = weigthsArr[0]
        super().create(weigthsArr)

    
    def call(self, inputs: np.array):
        self.inputs = inputs 
        multiplySum = np.dot(inputs, self.weights) + self.bias
        multiplySum += self.bias
        self.multiplySum = multiplySum
        self.output = self.activation(multiplySum)
        return self.output

    

"""
input dengan random
"""
# Sample input: one data point with 3 features
inputs = np.array([[0.5, 0.2, -0.1]])

# Create a Dense layer with 4 neurons, 'relu' activation, and input shape (3,)
dense_layer = Dense(neurons=4, activation='relu', inputShape=(3,))

# Perform feed forward propagation: compute the layer's output
outputs = dense_layer.call(inputs)
print("Layer hidden 1 output:", outputs)

# Create a Dense layer with 2 neurons, 'sigmoid' activation, and input shape (4,)
dense_layer = Dense(neurons=2, activation='sigmoid', inputShape=(4,))
outputs = dense_layer.call(outputs)
print("Layer output:", outputs)



"""
input manual 
"""

