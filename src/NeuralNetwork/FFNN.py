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
        x: np.ndarray,
        y: np.ndarray,
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
        transformasi dari dataset ke NN dan dari NN ke kelas target). Jadi layer yang dimasukkan
        tidak termasuk ketika input fitur dan output target

        Args:
            x (np.ndarray): dataset
            y (np.ndarray): target
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
        assert x.shape[0] == y.shape[0], f"Number of x row ({x.shape[0]}) should be same with\
            number of row in y ({y.shape[0]})"
        self.x = np.array([[Scalar(v) for v in row] for row in x])
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

        self.weights = [[[Scalar(0)] for _ in range(layers[i])] for i in range(len(layers))]
        self.bias = [[Scalar(0)] for _ in range(len(self.layers))]
        self.initialize_weights(weight_method)

        # Parameters
        self.loss_function = loss_function
        assert epochs >= 1, "Number of epochs must be greater than 1"
        self.epochs = epochs
        assert batch_size >= 1, "Number of batch must be greater than 1"
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Initiate network
        self.layer_values = [[Scalar(0) for _ in range(layers[j])] for j in range(len(layers))]


    def initialize_weights(self, method: str = "random"):
        """
        Initialize weights and biases

        Args:
            method (str, optional): weights initialization. Defaults to "random".
        """
        if method == "zero":  # Zero initialization
            for i, _ in enumerate(self.layers):
                if i == 0:
                    self.weights[i], self.bias[i] = zero_initialization(
                        row_dim=self.layers[0], col_dim=self.x.shape[0]
                    )
                else:
                    self.weights[i], self.bias[i] = zero_initialization(
                        row_dim=self.layers[i], col_dim=self.layers[i-1]
                    )

        elif method == "normal":  # Normal distribution
            for i, _ in enumerate(self.layers):
                if i == 0:
                    self.weights[i], self.bias[i] = normal_distribution(
                        mean=self.mean,
                        variance=self.variance,
                        row_dim=self.layers[0], # Hidden layer pertama
                        col_dim=self.x.shape[0], # Input dataset
                        seed=self.seed,
                    )
                else:
                    self.weights[i], self.bias[i] = normal_distribution(
                        mean=self.mean,
                        variance=self.variance,
                        row_dim=self.layers[i],
                        col_dim=self.layers[i-1],
                        seed=self.seed,
                    )


        elif method == "xavier":  # Xavier initialization
            for i, _ in enumerate(self.layers):
                if i == 0:
                    self.weights[i], self.bias[i] = xavier_initialization(
                        row_dim=self.layers[0], col_dim=self.x.shape[0], seed=self.seed
                    )
                else:
                    self.weights[i], self.bias[i] = xavier_initialization(
                        row_dim=self.layers[i], col_dim=self.layers[i-1], seed=self.seed
                    )

        elif method == "he":  # He initialization
            for i, _ in enumerate(self.layers):
                if i == 0:
                    self.weights[i], self.bias[i] = he_initialization(
                        row_dim=self.layers[0], col_dim=self.x.shape[0], seed=self.seed
                    )
                else:
                    self.weights[i], self.bias[i] = he_initialization(
                        row_dim=self.layers[1], col_dim=self.layers[i-1], seed=self.seed
                    )

        else:  # Uniform distribution (default)
            for i, _ in enumerate(self.layers):
                if i == 0:
                    self.weights[i], self.bias[i] = random_uniform_distribution(
                        lower_bound=self.lower_bound,
                        upper_bound=self.upper_bound,
                        row_dim=self.x.shape[0],
                        col_dim=self.x.shape[0],
                        seed=self.seed,
                    )
                else:
                    self.weights[i], self.bias[i] = random_uniform_distribution(
                        lower_bound=self.lower_bound,
                        upper_bound=self.upper_bound,
                        row_dim=self.layers[1],
                        col_dim=self.layers[i-1],
                        seed=self.seed,
                    )


    def net(self, weights: np.ndarray, inputs: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """
        Calculate net of layer n

        Args:
            inputs (array of input): inputs
            weight (matrix of weight, size of n-th layer * (n-1)-th layer): matrix of weights
            bias (array of bias), biases for layer between n and n-1
        Returns:
            _type_: _description_
        """
        return np.dot(weights, inputs) + bias


    def activate(self, activation: str, val) -> Scalar:
        """
        Activation function

        Args:
            activation (Callable): activation method function (function)
            val (Scalar): value

        Returns:
            Scalar: _description_
        """
        if not isinstance(val, Scalar):
            val = val[0] # Ambil nilai ke-0 karena hasil perkalian matriks return-nya 2D
            for i, v in enumerate(val):
                if activation == "relu":
                    val[i] = v.relu()
                elif activation == "sigmoid":
                    val[i] =  v.sigmoid()
                elif activation == "tanh":
                    val[i] = v.tanh()
                else:
                    val[i] = v.linear()
            return val
        else:
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
        for i, _ in enumerate(self.x):
            for j in range(len(self.layers)):
                if j == 0:
                    self.layer_values[j] = self.net(self.weights[0], [self.x[i]], self.bias[j])
                else:
                    self.layer_values[j] = self.net(self.weights[j], self.layer_values[j - 1], self.bias[j - 1])
                self.layer_values[j] = self.activate(self.activations[j], self.layer_values[j])


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

