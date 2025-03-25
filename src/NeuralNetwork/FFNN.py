from typing import List, Optional
import numpy as np
from NeuralNetwork.WeightGenerator import (
    zero_initialization,
    random_uniform_distribution,
    normal_distribution,
    xavier_initialization,
    he_initialization,
)
from NeuralNetwork.Autograd import Scalar

class FFNN:
    """
    Implementation of Feed Forward Neural Netrowk
    """

    def __init__(
        self,
        x,
        y,
        layers: List[int],
        activations: List[str] | str = "relu",
        weight_method: str = "uniform",
        loss_function: str = "mse",
        batch_size: int = 1,
        epochs: int = 5,
        learning_rate: float = 0.01,
        mean: Optional[int | float] = None,
        variance: Optional[int | float] = None,
        lower_bound: Optional[int | float] = None,
        upper_bound: Optional[int | float] = None,
        seed: Optional[int | float] = None,
    ):
        """
        Create instance of Feed Forward Neural Network. Diasumsikan bahwa input layer dan output
        layer sebenarnya adalah hidden layer pertama dan terakhir (yang bertugas melakukan
        transformasi dari dataset ke NN dan dari NN ke kelas target)

        Args:
            x (np.ndarray/np.array): dataset
            y (np.array/np.ndarray): target
            layers (List[int]): number of neurons in each layers (include input & output layer)
            activations (List[str]): activation function for each hidden layers and output layer
            loss_function (str, optional): Loss function. Defaults to "mse".
            batch_size (int, optional): batch size. Defaults to 1.
            epochs (int, optional): number of epochs. Defaults to 5.
            learning_rate (float): learning rate
            mean, variance (float, optional): used in normal weight distribution
            lower_bound, upper_bound (float, optional): used in uniform weight distribution
            seed (float, optional): seed for normal and uniform distribution
        Raises:
            PerceptronException: Check if length for array activations and layers are the same
        """
        # Initialize the data
        assert x.shape[0] == y.shape[0], "Number of x row should be same with number of row in y"
        self.x = np.array([[Scalar(v) for v in row] for  row in x])
        self.y = np.array([Scalar(v) for v in y])

        # Check on layers
        for i, _ in enumerate(layers):
            assert layers[i] > 0, f"Number of neurons {i} must be bigger than 0 ({layers[i]})"
        self.layers = layers # All layers: input layer, hidden layer, output layer

        if isinstance(activations, list):
            # Misal ada 3 layer (termasuk input & output)
            # Activation akan ada di hidden layer 1 dan output layer saja
            assert len(activations) == len(layers), "Number of activations must be the same \
                with number of layers"
            self.activations = activations
        else:
            self.activations = [activations] * len(layers)

        # Initialize weights
        if weight_method == "normal":
            assert mean is not None and variance is not None, "Jika weight menggunakan metode \
                normal, mean dan variance harus dimasukkan. Seed dianjurkan dimasukkan"
        elif weight_method == "uniform":
            assert lower_bound is not None and upper_bound is not None, "Jika weight menggunakan \
                metode uniform, lower dan upper bound harus dimasukkan. Seed dianjurkan"
            assert upper_bound >= lower_bound, "Upper bound must be higher than lower bound"

        self.mean = mean
        self.variance = variance
        self.seed = seed
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.weights: list[list[Scalar]] = [[] for _ in range(len(self.layers))]
        self.bias: list[list[Scalar]] = [[] for _ in range(len(self.layers))]
        self.initialize_weights(weight_method)

        # Parameters
        self.loss_function = loss_function
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


    def initialize_weights(self, method: str = "random"):
        """
        Initialize weights and biases

        Args:
            method (str, optional): weights initialization. Defaults to "random".
        """
        if method == "zero":  # Zero initialization
            for i, _ in enumerate(self.layers):
                self.weights[i], self.bias[i] = zero_initialization(
                    row_dim=self.x.shape[1], col_dim=self.layers[i]
                )

        elif method == "normal":  # Normal distribution
            for i, _ in enumerate(self.layers):
                self.weights[i], self.bias[i] = normal_distribution(
                    mean=self.mean,
                    variance=self.variance,
                    row_dim=self.x.shape[1],
                    col_dim=self.layers[i],
                    seed=self.seed,
                )

        elif method == "xavier":  # Xavier initialization
            for i, _ in enumerate(self.layers):
                self.weights[i], self.bias[i] = xavier_initialization(
                    row_dim=self.x.shape[1], col_dim=self.layers[i], seed=self.seed
                )

        elif method == "he":  # He initialization
            for i, _ in enumerate(self.layers):
                self.weights[i], self.bias[i] = he_initialization(
                    row_dim=self.x.shape[1], col_dim=self.layers[i], seed=self.seed
                )

        else:  # Uniform distribution (default)
            for i, _ in enumerate(self.layers):
                self.weights[i], self.bias[i] = random_uniform_distribution(
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                    row_dim=self.x.shape[1],
                    col_dim=self.layers[i],
                    seed=self.seed,
                )


    def net(self, inputs, weight, i):
        """
        Calculate net of layer n

        Args:
            inputs (array of input): inputs
            weight (matrix of weight, size of (n-1)-th layer * n-th layer): matrix of weights

        Returns:
            _type_: _description_
        """
        return self.bias[i] + np.dot(inputs[i], weight[i])


    def activate(self, activation: str, val: Scalar) -> Scalar:
        """
        Activation function

        Args:
            activation (Callable): activation method function (function)
            val (_type_): value

        Returns:
            _type_: _description_
        """
        if activation == "relu":
            return val.relu()
        elif activation == "sigmoid":
            return val.sigmoid()
        elif activation == "tanh":
            return val.tanh()
        else:
            return val.linear()


    def forward(self):
        """
        Do forward propagation
        """


    def backward(self):
        """
        Do backward propagation
        """

    def fit(self):
        """
        Train model
        """
        for _ in range(self.epochs):
            for _ in range(0, self.x.shape[0], self.batch_size):
                self.forward()
                self.backward()


# x(2), h1(2), o(1)
# if __name__ == "__main__":
#     x = np.array([[0.05, 0.1], [0.8, 0.7]])
#     y = np.array([0, 1])

#     ffnn = FFNN(x, y, [2, 1], [relu, relu])
#     ffnn.initialize_weights()

#     # misalkan untuk data pertama dulu (batch 1 jika batch_size = 1)

#     # h1
#     net_h1 = ffnn.net(x[0], ffnn.weights[0])
#     print(f"Net h1: {net_h1}")
#     h1 = ffnn.activate(relu, net_h1)
#     print(f"H1: {h1}")

#     # o
#     net_o = ffnn.net(h1, ffnn.weights[1])
#     print(f"Net O: {net_o}")
#     o = ffnn.activate(relu, net_o)
#     print(f"O: {o}")
