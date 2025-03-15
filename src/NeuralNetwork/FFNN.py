from typing import List, Callable
import numpy as np
from PerceptronException import PerceptronException

class FFNN:
    """
    Implementation of Feed Forward Neural Netrowk
    """
    def __init__(self, x, y, layers: List[int], activations: List[Callable], loss_function: Callable = "mse", batch_size: int = 1, epochs: int = 10 ):
        """
        Create instance of Feed Forward Neural Network

        Args:
            x (np.ndarray/np.array): dataset
            y (np.array/np.ndarray): target
            layers (List[int]): list of number of layers (hidden layers and output layer)
            activations (List[Callable]): activation function for each hidden layers and output layer
            loss_function (Callable, optional): Loss function. Defaults to "mse".
            batch_size (int, optional): batch size. Defaults to 1.
            epochs (int, optional): number of epochs. Defaults to 10.

        Raises:
            PerceptronException: Check if length for array activations and layers are the same
        """
        self.x = x # Input of layer 0 (input layer)
        self.y = y
        self.layers = layers

        self.activations = activations
        if len(activations) != len(layers):
            raise PerceptronException("Number of activations must be the same with number of hidden layers + 1 output layer")

        self.weights = [[] for _ in range(len(self.layers))]
        self.loss_function = loss_function
        self.epochs = epochs
        self.batch_size = batch_size


    def initialize_weights(self, method: str = "random"): 
        """
        Initialize weights

        Args:
            method (str, optional): weights initialization. Defaults to "random".
        """
        # TODO: implement initialize weight
        self.weights = [[] for _ in range(len(self.layers))]
        for i, _ in enumerate(self.layers):
            # Banyak weight antara layer (n-1) ke layer n = ((n-1)th layer + 1 (bias)) * layer n
            self.weights[i] = np.random.rand(self.x.shape[1] + 1, self.layers[i]) # + 1 for bias, bias is the first row


    def net(self, inputs, weight):
        """
        Calculate net of layer n

        Args:
            inputs (array of input): inputs
            weight (matrix of weight, size of ((n-1)-th layer + 1) * n-th layer): matrix of weights. Bias included as w0

        Returns:
            _type_: _description_
        """
        return weight[0] + np.dot(inputs, weight[1:])

    def activate(self, activation: Callable, val):
        """
        Activation function

        Args:
            activation (Callable): activation method function (function)
            val (_type_): value

        Returns:
            _type_: _description_
        """
        return activation(val)

    def forward(self):
        pass

    def backward(self):
        pass

    def train(self):
        # Forward propagation
        # TODO: implement mini-batch (multiple row of dataset) and epoch
        # Layer input merupakan satu row, dan di NN layer input berupa x1 x2 dst merupakan fitur-fiturnya
        for _ in range(self.epochs):
            for _ in range(0, self.x.shape[0], self.batch_size):
                self.forward()
                self.backward()

# Contoh
def relu(val):
    return np.maximum(0, val)

# x(2), h1(2), o(1)
if __name__ == "__main__":
    x = np.array([[0.05, 0.1],
        [0.8, 0.7]])
    y = np.array([0, 1])

    ffnn = FFNN(x, y, [2, 1], [relu, relu])
    ffnn.initialize_weights()

    # misalkan untuk data pertama dulu (batch 1 jika batch_size = 1)

    # h1
    net_h1 = ffnn.net(x[0], ffnn.weights[0])
    print(f"Net h1: {net_h1}")
    h1 = ffnn.activate(relu, net_h1)
    print(f"H1: {h1}")

    # o
    net_o = ffnn.net(h1, ffnn.weights[1])
    print(f"Net O: {net_o}")
    o = ffnn.activate(relu, net_o)
    print(f"O: {o}")