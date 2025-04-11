from typing import Optional
import numpy as np

from NeuralNetwork.WeightGenerator import (
    zero_initialization,
    random_uniform_distribution,
    normal_distribution,
    xavier_initialization,
    he_initialization,
    one_initialization
)


class Layer:
    """
    Representation of a single layer
    """
    def __init__(self, input_dim: int, output_dim: int, activation: str, weight_method: str = "uniform", seed: Optional[int] = None, mean: Optional[float] = None, variance: Optional[float] = 0
                 , lower_bound: Optional[float] = None, upper_bound: Optional[float] = None):
        """
        Initialize a single layer of the neural network.
        Args:
            data_length (int): Number of rows in the epochs
            input_dim (int): Number of input neurons.
            output_dim (int): Number of output neurons.
            activation (str): Activation function for the layer.
            weight_method (str): Weight initialization method.
            seed (int, optional): Seed for random weight initialization.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weights, self.bias = self.initialize_weights(weight_method, seed)
        self.mean = mean
        self.variance = variance
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def initialize_weights(self, method: str, seed: Optional[int]):
        """
        Initialize weights and biases based on the specified method.
        """
        if method == "zero":
            return zero_initialization(row_dim=self.output_dim, col_dim=self.input_dim)
        elif method == "one":
            return one_initialization(row_dim=self.output_dim, col_dim=self.input_dim)
        elif method == "normal":
            return normal_distribution(mean=self.mean, variance=self.variance, row_dim=self.output_dim, col_dim=self.input_dim, seed=seed)
        elif method == "xavier":
            return xavier_initialization(row_dim=self.output_dim, col_dim=self.input_dim, seed=seed)
        elif method == "he":
            return he_initialization(row_dim=self.output_dim, col_dim=self.input_dim, seed=seed)
        else:  # Default to uniform distribution
            return random_uniform_distribution(lower_bound=-1, upper_bound=1, row_dim=self.output_dim, col_dim=self.input_dim, seed=seed)


    def activate(self, val):
        """
        Apply activation function element-wise to a numpy array of Scalars.
        """
        if self.activation == "relu":
            # Apply ReLU to each Scalar in the array
            return np.vectorize(lambda v: v.relu())(val)
        elif self.activation == "sigmoid":
            return np.vectorize(lambda v: v.sigmoid())(val)
        elif self.activation == "tanh":
            return np.vectorize(lambda v: v.tanh())(val)
        else:  # Linear activation
            return val  # No change needed for linear

    def forward(self, x):
        """Compute net input and activated output for a given input x."""
        net = np.dot(self.weights, x) + self.bias
        activated = self.activate(net)
        return activated
