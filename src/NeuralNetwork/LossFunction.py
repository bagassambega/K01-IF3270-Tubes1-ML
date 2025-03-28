from NeuralNetwork.Autograd import Scalar
import numpy as np

def mse(y_true: list[Scalar], y_pred: list[Scalar]) -> Scalar:
    """ Compute the Mean Squared Error (MSE) loss """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    n = len(y_true)

    errors = [(yt - yp) for yt, yp in zip(y_true, y_pred)]
    loss = sum(e ** 2 for e in errors) * (1 / n)

    print(f"loss mse: {loss}")
    return loss


def binary_cross_entropy(y_true: list[Scalar], y_pred: list[Scalar]) -> Scalar:
    """ Calculate Binary Cross Entropy (BCE) loss """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"

    epsilon = 1e-9  # To prevent log(0)
    n = len(y_true)

    loss_terms = [
        yt * (yp + epsilon).log() + (1 - yt) * ((1 - yp) + epsilon).log()
        for yt, yp in zip(y_true, y_pred)
    ]
    loss = sum(loss_terms) * (-1 / n)

    return loss

def categorical_cross_entropy(y_true, y_pred):
    """ Compute Categorical Cross Entropy (CCE) loss """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    
    epsilon = 1e-9  # To prevent log(0)
    n = len(y_true)

    # Extract values from Scalar objects
    y_pred_values = np.array([s[0].value for s in y_pred])

    # Clip predictions for numerical stability
    y_pred_values = np.clip(y_pred_values, epsilon, 1.0 - epsilon)

    # Compute categorical cross-entropy loss
    loss_value = -np.sum(y_true * np.log(y_pred_values)) / n
    loss_value = [Scalar(loss_value)]

    return loss_value
