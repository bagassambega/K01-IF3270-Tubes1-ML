from typing import List
from PerceptronException import PerceptronException

class Perceptron:
    def __init__(self, activation_function: function, inputs: List[any], weights: List[float]):
        self.activation_function = activation_function
        self.inputs = inputs
        self.weights = weights
    
    def net(self) -> float:
        sum: float = 0
        if (len(self.inputs) != len(self.weights)):
            raise PerceptronException("Number of inputs are not match with number of weights")
        elif (len(self.inputs) == 0):
            raise PerceptronException("No input found")
        elif (len(self.weights) == 0):
            raise PerceptronException("No weight found")
        else:
            for i in range(len(self.inputs)):
                sum += self.inputs[i] * self.weights[i]
        
        return sum