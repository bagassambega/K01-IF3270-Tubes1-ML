import numpy as np

def mse(y_true, y_pred, derivative=False):
    if derivative:
        return 2 * (y_pred - y_true) / y_true.size
    return np.mean(np.square(y_true - y_pred))

def binary_crossentropy(y_true, y_pred, derivative=False):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    if derivative:
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.size
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true, y_pred, derivative=False):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    if derivative:
        return -y_true / y_pred / y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def get_loss_function(name):
    if name == 'mse':
        return mse
    elif name == 'binary_crossentropy':
        return binary_crossentropy
    elif name == 'categorical_crossentropy':
        return categorical_crossentropy
    else:
        raise ValueError(f"Loss function '{name}' not supported")

def get_loss_derivative(name):
    loss_func = get_loss_function(name)
    return lambda y_true, y_pred: loss_func(y_true, y_pred, derivative=True)