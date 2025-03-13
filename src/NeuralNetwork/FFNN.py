from typing import List, Callable, Optional
import numpy as np
from sklearn.datasets import fetch_openml


class FFNN:
    """
    Implementation of Feed Forward Neural Network
    """
    def __init__(
        self,
        num_layers: int,
        num_perceptrons: List[int],
        activation_function: List[Callable],
        loss_function: Callable,
        dataset: str,
        weights: List[np.ndarray[np.dtype[float], np.dtype[float]]],
    ):
        """
        Initialization of FFNN struct

        Args:
            num_layers (int): number of layers (including the input and output layer)
            num_perceptrons (List[int]): list of number of perceptrons in each layer (not including the input layer)
            activation_function (List[Callable[[float], float]]): list of activation function in each layer (not including the input layer)
            dataset (str): name of dataset
            loss_function (Callable): loss function to calculate the error in the end of forward propagation
            weights (List[np.ndarray[np.dtype[float], np.dtype[float]]]): list of weight and bias for each layer (for example there's total 5 layer including input and output layer, there should be 4 np.ndarray of weights)
        """
        # Load the dataset
        self.X = None
        self.y = None
        self.load_dataset(dataset=dataset)

        # Check for layers and number of perceptrons
        assert len(num_perceptrons) == num_layers - 1 # Input layer not counted
        assert len(activation_function) == num_layers - 1 # Input layer not counted
        self.num_layers: int = num_layers
        self.num_perceptrons: List[int] = num_perceptrons
        self.activation_function: List[Callable] = activation_function
        self.loss = loss_function

        # Check weights
        assert len(weights) == num_layers - 1
        self.weights = weights
        for i in range(num_layers):
            assert len(self.weights[i]) == (num_perceptrons[i] + 1) # Check for bias as w0 too, so num_perceptrons in each layer + 1

        for i in range(len(self.X)):
            assert len(self.weights[i]) == len(self.X[i]) + 1 # Check if number of features is match with number of weight + bias


    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size: Optional[int],
        learning_rate: Optional[float],
        epochs: Optional[int] = 1,
        verbose: Optional[int] = 0,
    ):
        pass


    def load_dataset(self, dataset: str):
        """
        Load dataset into X (matrix of instances, features) and y (list of target)

        Args:
            dataset (str): name of dataset to OpenML
        """
        self.X, self.y = fetch_openml(name=dataset, version=1, return_X_y=True, as_frame=False, cache=True)
