from typing import Callable, Optional
import numpy as np


class FeedForwardLayer:
    """
    Implementation of a single layer in neural network
    """
    def __init__(
        self,
        num_perceptron: int,
        layer_activation_function: Callable[[float], float],
        perceptrons: Optional[np.array] = None,
    ):
        self.num_perceptron = num_perceptron
        self.layer_activation_function = layer_activation_function
        self.perceptrons = perceptrons if perceptrons is not None else np.array([])

    def set_layer_activation_function(
        self, layer_activation_function: Callable[[float], float]
    ):
        """
        Set activation function for the layer (applied to all perceptron)

        Args:
            layer_activation_function (Callable[[float], float]): function
        """
        self.layer_activation_function = layer_activation_function

    def set_perceptrons(self, perceptrons: np.array):
        """
        Set the perceptrons in the layer

        Args:
            perceptrons (np.array): _description_
        """
        self.perceptrons = perceptrons
        self.num_perceptron = len(perceptrons)
