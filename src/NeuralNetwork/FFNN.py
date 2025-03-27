from typing import List, Optional
from tqdm import tqdm
import numpy as np
from NeuralNetwork.WeightGenerator import (
    zero_initialization,
    random_uniform_distribution,
    normal_distribution,
    xavier_initialization,
    he_initialization,
    one_initialization
)
from NeuralNetwork.Autograd import Scalar
from NeuralNetwork.LossFunction import binary_cross_entropy, mse, categorical_cross_entropy

class FFNN:
    """
    Implementation of Feed Forward Neural Netrowk
    """


    def __init__(
        self,
        x: np.ndarray | List,
        y: np.ndarray | List,
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
        verbose: Optional[bool] = False,
        randomize: Optional[bool] = False
    ):

        """
        Create instance of Feed Forward Neural Network. Diasumsikan bahwa input layer dan output
        layer sebenarnya adalah hidden layer pertama dan terakhir (yang bertugas melakukan
        transformasi dari dataset ke NN dan dari NN ke kelas target). Jadi layer yang dimasukkan
        tidak termasuk ketika input fitur dan output target, namun nantinya akan ditambahkan output
        layer secara hardcoded di dalam kode

        Args:
            x (np.ndarray): dataset
            y (np.ndarray): target
            layers (List[int]): number of neurons in each layers (hidden layer)
            activations (List[str]) | str: activation function for each hidden layers + output layer
            loss_function (str, optional): Loss function. Defaults to "mse".
            batch_size (int, optional): batch size. Defaults to 1.
            epochs (int, optional): number of epochs. Defaults to 5.
            learning_rate (float): learning rate
            mean, variance (float, optional): used in normal weight distribution
            lower_bound, upper_bound (float, optional): used in uniform weight distribution
            seed (float, optional): seed for normal and uniform distribution
            verbose (bool, optional): See logs of processes. Default False
            randomize (bool, optional): randomize rows while training. Default False
        Raises:
            PerceptronException: Check if length for array activations and layers are the same
        """

        # Initialize the data (input layers)
        assert x.shape[0] == y.shape[0], f"Number of x row ({x.shape}) should be same with\
            number of row in y ({y.shape})"

        self.x = np.array([[Scalar(v) for v in row] for row in x])
        self.y = np.array([Scalar(v) for v in y])

        # Check on layers
        for i, _ in enumerate(layers):
            assert layers[i] > 0, f"Number of neurons {i} must be bigger than 0 ({layers[i]})"
        layers.append(1) # From last hidden layer to output layer. Output layer must be 1
        self.layers = layers # All layers: input layer, hidden layer, output layer

        if isinstance(activations, List):
            # Misal ada 3 layer (termasuk input & output)
            # Activation akan ada di hidden layer 1 dan output layer saja
            assert len(activations) == len(layers), "Number of activations must be the same \
                with number of layers"
            for act in activations:
                assert act in ["relu", "tanh", "sigmoid", "linear"], f"No activation {act} found"
            self.activations = activations
        else:
            self.activations = [activations] * len(layers)
            assert activations in ["relu", "tanh", "sigmoid", "linear"], f"No activation \
                {activations} found"

        # Initialize weights
        assert weight_method in ["normal", "uniform", "zero", "xavier", "he", "one"], f"No \
            weighting method found for {weight_method}"
        if weight_method == "normal":
            assert mean is not None and variance is not None, "Jika weight menggunakan metode \
                normal, mean dan variance harus dimasukkan. Seed dianjurkan dimasukkan"
        elif weight_method == "uniform":
            assert lower_bound is not None and upper_bound is not None, "Jika weight menggunakan \
                metode uniform, lower dan upper bound harus dimasukkan. Seed dianjurkan"
            assert upper_bound >= lower_bound, "Upper bound must be higher than lower bound"

        # Parameter for weights
        self.mean = mean
        print(self.mean)
        self.variance = variance
        self.seed = seed
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Initialize weights
        self.weights = [[[Scalar(0)] for _ in range(layers[i])] for i in range(len(layers))]
        self.bias = [[Scalar(0)] for _ in range(len(self.layers))]
        self.initialize_weights(weight_method)

        # Parameters
        assert loss_function in ["mse", "binary_cross_entropy", "categorical_cross_entropy"], f"\
            No loss function found for {loss_function}"
        self.loss_function = loss_function
        assert epochs >= 1, "Number of epochs must be greater than 1"
        self.epochs = epochs
        assert batch_size >= 1, "Number of batch must be greater than 1"
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.randomize = randomize

        # Initiate network
        self.layer_net = [[[Scalar(0) for _ in range(layers[j])] for j in range(len(layers))] for _ in range(len(self.x))]
        self.layer_output = [[[Scalar(0) for _ in range(layers[j])] for j in range(len(layers))] for _ in range(len(self.x))]

        # Loss value for each row of dataset
        self.loss_values: List[Scalar] = [Scalar(0) for _ in range(x.shape[0])]

        # Verbose
        self.verbose = verbose


    def initialize_weights(self, method: str = "uniform"):
        """
        Initialize weights and biases

        Args:
            method (str, optional): weights initialization. Defaults to "random".
        """
        if method == "zero":  # Zero initialization
            for i, _ in enumerate(self.layers):
                if i == 0:
                    self.weights[i], self.bias[i] = zero_initialization(
                        row_dim=self.layers[0], col_dim=self.x.shape[1]
                    )
                else:
                    self.weights[i], self.bias[i] = zero_initialization(
                        row_dim=self.layers[i], col_dim=self.layers[i-1]
                    )

        elif method == "one":  # Zero initialization
            for i, _ in enumerate(self.layers):
                if i == 0:
                    self.weights[i], self.bias[i] = one_initialization(
                        row_dim=self.layers[0], col_dim=self.x.shape[1]
                    )
                else:
                    self.weights[i], self.bias[i] = one_initialization(
                        row_dim=self.layers[i], col_dim=self.layers[i-1]
                    )

        elif method == "normal":  # Normal distribution
            for i, _ in enumerate(self.layers):
                if i == 0:
                    self.weights[i], self.bias[i] = normal_distribution(
                        mean=self.mean,
                        variance=self.variance,
                        row_dim=self.layers[0], # Hidden layer pertama
                        col_dim=self.x.shape[1], # Input dataset
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
                        row_dim=self.layers[0], col_dim=self.x.shape[1], seed=self.seed
                    )
                else:
                    self.weights[i], self.bias[i] = xavier_initialization(
                        row_dim=self.layers[i], col_dim=self.layers[i-1], seed=self.seed
                    )

        elif method == "he":  # He initialization
            for i, _ in enumerate(self.layers):
                if i == 0:
                    self.weights[i], self.bias[i] = he_initialization(
                        row_dim=self.layers[0], col_dim=self.x.shape[1], seed=self.seed
                    )
                else:
                    self.weights[i], self.bias[i] = he_initialization(
                        row_dim=self.layers[i], col_dim=self.layers[i-1], seed=self.seed
                    )

        else:  # Uniform distribution (default)
            for i, _ in enumerate(self.layers):
                if i == 0:
                    self.weights[i], self.bias[i] = random_uniform_distribution(
                        lower_bound=self.lower_bound,
                        upper_bound=self.upper_bound,
                        row_dim=self.layers[0],
                        col_dim=self.x.shape[1],
                        seed=self.seed,
                    )
                else:
                    self.weights[i], self.bias[i] = random_uniform_distribution(
                        lower_bound=self.lower_bound,
                        upper_bound=self.upper_bound,
                        row_dim=self.layers[i],
                        col_dim=self.layers[i-1],
                        seed=self.seed,
                    )


    def net(self, weights: np.ndarray, inputs: np.ndarray, bias: np.ndarray, i) -> np.ndarray:
        """
        Calculate net of layer n

        Args:
            inputs (array of input): inputs
            weight (matrix of weight, size of n-th layer * (n-1)-th layer): matrix of weights
            bias (array of bias), biases for layer between n and n-1
        Returns:
            _type_: _description_
        """
        if i == 0:
            # Perlu di-transpose karena nilai yang diambil dari dataset akan sebesar 1 x num_feature
            return np.dot(weights, np.array(inputs, dtype=object).reshape(-1, 1)) + bias
        else:
            return np.dot(weights, inputs) + bias



    def activate(self, activation: str, val) -> Scalar | np.ndarray | List:
        """
        Activation function

        Args:
            activation (str): activation method function (function)
            val (): value

        Returns:
            Scalar: _description_
        """
        if not isinstance(val, Scalar):
            for i, v in enumerate(val):
                # Menggunakan indeks 0 karena matriks nantinya berukuran n x 1,
                # dengan 1 adalah array juga
                if activation == "relu":
                    val[i] = v[0].relu()
                elif activation == "sigmoid":
                    val[i] =  v[0].sigmoid()
                elif activation == "tanh":
                    val[i] = v[0].tanh()
                else:
                    val[i] = v[0].linear()
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


    def loss(self, loss_method: str, y_true: List[Scalar], y_pred: List[Scalar]) -> Scalar:
        """
        Calculate loss/error

        Args:
            error_method (str): _description_
            y_true (List[Scalar]): _description_
            y_pred (List[Scalar]): _description_

        Returns:
            Scalar: _description_
        """
        if isinstance(y_true, Scalar) and isinstance(y_pred, Scalar):
            if loss_method == "categorical_cross_entropy":
                return categorical_cross_entropy(y_pred=[y_pred], y_true=[y_true])
            elif loss_method == "binary_cross_entropy":
                return binary_cross_entropy(y_pred=[y_pred], y_true=[y_true])
            else:
                return mse(y_pred=[y_pred], y_true=[y_true])
        else:
            if loss_method == "categorical_cross_entropy":
                return categorical_cross_entropy(y_pred=y_pred, y_true=y_true)
            elif loss_method == "binary_cross_entropy":
                return binary_cross_entropy(y_pred=y_pred, y_true=y_true)
            else:
                return mse(y_pred=y_pred, y_true=y_true)


    def forward(self):
        """
        Do forward propagation
        """
        for i, _ in enumerate(self.x):
            for j, _ in enumerate(self.layers):
                # From input layer to first hidden
                if j == 0:
                    # print(self.x[i])
                    self.layer_net[i][j] = self.net(self.weights[0], [self.x[i]], self.bias[0], j)
                # Hidden layers
                else:
                    self.layer_net[i][j] = self.net(self.weights[j], self.layer_net[i][j - 1], self.bias[j], j)
                # print(i, "net:", self.layer_net[i][j].shape)
                self.layer_output[i][j] = self.activate(self.activations[j], self.layer_net[i][j])
                # print("output:", self.layer_output[i][j].shape)
                # print(f"{i} {j}:", self.layer_output[i][j], ", shape:", self.layer_output[i][j].shape)

            # Calculate the loss
            self.loss_values[i] = self.loss(self.loss_function, [self.y[i]], self.layer_output[i][-1][0])
            # print("loss:", self.loss_values[i])

            if self.verbose:
                print("Loss:", self.loss_values[i], "Predicted:", self.layer_output[i][-1][0], "Target:", self.y[i])


    def backprop(self):
        """
        Do backward propagation
        """
        for i, _ in enumerate(self.x):
            # Get the gradient
            self.loss_values[i].backward()

            # Update weights
            for j, _ in enumerate(self.weights): # Through layer
                for k in range(len(self.weights[j])): # Through baris
                    for l in range(len(self.weights[j][k])): # Through kolom
                        if j == 0:
                            temp_x = self.x[i][k]
                        else:
                            temp_x = self.layer_output[i][j - 1][l][0]
                        # print(i, j, k, l, self.weights[j][k][l])
                        if not isinstance(temp_x, (float, int, np.number, Scalar)):
                            raise TypeError(f"Wrong: {type(temp_x)}")
                        if isinstance(temp_x.value, (Scalar)):
                            raise TypeError(f"Wrong: {temp_x} {type(temp_x)}")
                        temp = self.weights[j][k][l].grad * self.learning_rate * temp_x.value
                        self.weights[j][k][l].value -= temp

            # Update bias
            for j, _ in enumerate(self.bias):
                for k in range(len(self.bias[j])):
                    if j == 0:
                        temp_x = self.x[i][k]
                    else:
                        temp_x = self.layer_output[i][j - 1][0][0]
                    self.bias[j][k][0].value -= self.bias[j][k][0].grad * self.learning_rate * temp_x.value

            self._zero_gradients()

    def _zero_gradients(self):
        """Reset all gradients to zero before processing a new batch"""
        for layer in self.weights:
            for neuron in layer:
                for w in neuron:
                    w.grad = 0

        for layer in self.bias:
            for b in layer:
                b[0].grad = 0

    def print_weight(self, epoch: int):
        """
        Print weight for debugging purposes
        """
        print(epoch)
        for i, layer in enumerate(self.weights):
            print("Layer ke-" + str(i) + ":")
            print(layer)


    def fit(self):
        """
        Train model with progress bar
        """
        num = len(self.x)
        indices = np.arange(num)
        total_loss = 0

        # Create progress bar for epochs
        epoch_pbar = tqdm(range(self.epochs), desc="Epochs", disable=not self.verbose)

        for epoch in epoch_pbar:
            # self.print_weight(epoch)
            if self.randomize:
                np.random.shuffle(indices)

            epoch_loss = 0
            batch_count = 0

            # Create progress bar for batches
            batch_pbar = tqdm(range(0, num, self.batch_size),
                            desc=f"Epoch {epoch+1}/{self.epochs} Batches",
                            leave=False,
                            disable=not self.verbose)

            for batch_start in batch_pbar:
                batch_end = min(batch_start + self.batch_size, num)
                batch_indices = indices[batch_start:batch_end]
                batch_loss = 0

                for i in batch_indices:
                    # Forward pass
                    for j in range(len(self.layers)):
                        if j == 0:
                            self.layer_net[i][j] = self.net(self.weights[0], [self.x[i]], self.bias[0], j)
                        else:
                            self.layer_net[i][j] = self.net(self.weights[j], self.layer_net[i][j-1], self.bias[j], j)
                        self.layer_output[i][j] = self.activate(self.activations[j], self.layer_net[i][j])

                    # Calculate loss
                    self.loss_values[i] = self.loss(self.loss_function, [self.y[i]], self.layer_output[i][-1][0])
                    batch_loss += self.loss_values[i].value
                    self.loss_values[i].backward()

                # Update weights and biases
                for j, _ in enumerate(self.weights):
                    for k in range(len(self.weights[j])):
                        for l in range(len(self.weights[j][k])):
                            self.weights[j][k][l].value -= self.weights[j][k][l].grad * self.learning_rate / len(batch_indices)
                            print("Update:", self.weights[j][k][l])

                for j, _ in enumerate(self.bias):
                    for k in range(len(self.bias[j])):
                        self.bias[j][k][0].value -= self.bias[j][k][0].grad * self.learning_rate / len(batch_indices)

                epoch_loss += batch_loss
                batch_count += 1
                batch_pbar.set_postfix({"Batch Loss": batch_loss/len(batch_indices)})

                self._zero_gradients()

            avg_epoch_loss = epoch_loss / num
            epoch_pbar.set_postfix({"Epoch Loss": avg_epoch_loss})
            total_loss += epoch_loss

        if self.verbose:
            print(f"\nFinal Average Loss: {total_loss/(num*self.epochs):.4f}")



    def predict_single(self, x):
        """
        Predict target of inputted single data

        Args:
            x (np.array, list): misalkan x adalah array berukuran n fitur
        """
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim == 2:
            if x.shape[1] != 1:
                x = x.reshape(-1, 1)
        layer_result = [[Scalar(0)] for _ in range(len(self.layers))]
        for j, _ in enumerate(self.layers):
            # From input layer to first hidden
            if j == 0:
                layer_result[j] = np.dot(self.weights[0], x)
            # Hidden layers
            else:
                layer_result[j] = self.net(self.weights[j], layer_result[j - 1], self.bias[j], j)
            self.layer_output[j] = self.activate(self.activations[j], layer_result[j])

        return layer_result[-1][0][0].value

    def predict(self, x):
        """
        Predict datas

        Args:
            x (np.array): _description_
        """
        if isinstance(x, list):
            x = np.array(x)

        if x.ndim == 1:
            return self.predict_single(x)
        res = []
        for row in x:
            res.append(self.predict_single(row))
        return res
