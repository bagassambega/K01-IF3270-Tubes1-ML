from typing import Callable, Optional
import numpy as np

class FeedForwardLayer:
    def __init__(self, num_perceptron: int, layer_activation_function: Callable[[float], float], perceptrons: Optional[np.array] = None):
        self.num_perceptron = num_perceptron
        self.layer_activation_function = layer_activation_function
        self.perceptrons = perceptrons if perceptrons is not None else np.array([])

    def set_layer_activation_function(self, layer_activation_function: Callable[[float], float]):
        self.layer_activation_function = layer_activation_function

    def set_perceptrons(self, perceptrons: np.array):
        self.perceptrons = perceptrons