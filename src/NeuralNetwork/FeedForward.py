from typing import Optional, List, Callable
from sklearn.datasets import fetch_openml


class FFNN:
    """
    Implementation of Feed Forward Neural Network
    """

    def __init__(
        self,
        num_layers: int,
        activations: List[Callable],
        num_batch: Optional[int] = 0,
        learning_rate: Optional[float] = 0.5,
        epoch: Optional[float] = 1,
        verbose: Optional[int] = 0,
    ):
        """
        Create new FFNN

        :param num_layers: Number of layers including input and output layers.
        :param layer_structure: List defining the number of neurons in each layer.
        :param activations: List of activation functions for each layer (except input).
        :param num_batch: Number of batch.
        :param learning_rate: Learning rate value.
        :param epoch: Number of epoch.
        :param verbose: 1 for printing log, 0 for not. By default set to 0.
        """

        self.num_layers = num_layers
        self.activations = activations
        self.num_batch = num_batch
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.verbose = verbose
        self.networks = []
        self.X = None
        self.y = None

        self.initialize_network()

    def initialize_network(
        self,
        weight_initialization: Optional[Callable] = None,
        input_initialization: Optional[Callable] = None,
    ):
        """
        Initialize network layers and perceptrons with random weights.
        """

        # TODO: create weight and input initialization
        if weight_initialization:
            weight_initialization()

        if input_initialization:
            input_initialization()


    def load_dataset(self, dataset: str = "mnist_784"):
        """
        Load dataset from OpenML and save it into X (data) and y (target class).
        """
        self.X, self.y = fetch_openml(dataset, version=1, return_X_y=True)

    def set_parameter(
        self,
        num_batch: Optional[int],
        learning_rate: Optional[float],
        epoch: Optional[float],
        verbose: Optional[int],
    ):
        """
        Change the parameter of number of batch, number of epoch, learning rate value, and/or verbose.
        """
        self.num_batch = num_batch
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.verbose = verbose
