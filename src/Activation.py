import math
from enum import Enum

class ActivationType(Enum): 
    LINEAR = 1 
    RELU = 2 
    SIGMOID = 3 
    TANH = 4 

class Activation: 
    def __init__(self, activation: ActivationType): 
        self.activation = activation
    
    def getActivation(self):
        return self.activation
    
    def compute(self, x): 
        if self.activation == ActivationType.LINEAR: 
            return Activation.LinearActivation(x)
        elif self.activation == ActivationType.RELU: 
            return Activation.ReLUActivation(x)
        elif self.activation == ActivationType.SIGMOID: 
            return Activation.SigmoidActivation(x)
        elif self.activation == ActivationType.TANH: 
            return Activation.TanhActivation(x)
        else: 
            raise ValueError("Activation function not found")

    @staticmethod 
    def LinearActivation(x): 
        """
        input: x
        nilai yang direturn selalu nilai itu sendiri
        """
        return x

    @staticmethod
    def ReLUActivation(x): 
        """
        input: x
        if x > 0, return x
        else, return 0
        """
        return max(0, x)

    @staticmethod
    def SigmoidActivation(x): 
        """
        input: x
        """
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def TanhActivation(x): 
        """
        input: x 
        """
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

"""
how to use: 
ActivationType(x): used for choosing activation function type 
1. Linear
2. ReLU
3. Sigmoid
4. Tanh

Or you can use something like this instead 
ActivationType.X replace X with (LINEAR, RELU, SIGMOID, TANH)
example: ActivationType.LINEAR -> this equal to ActivationType(1)

method compute is what you want to use to get the result of the activation function 
"""
activate = Activation(ActivationType(4)) 
print(activate.compute(1))