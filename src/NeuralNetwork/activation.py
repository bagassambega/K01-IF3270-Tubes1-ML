import numpy as np

def relu(x, derivative=False):
    x = np.array(x, dtype=float)
    if derivative:
        return (x > 0).astype(float)
    return np.maximum(0, x)

# Fungsi aktivasi sigmoid
def sigmoid(x, derivative=False):
    x = np.array(x, dtype=float)
    if derivative:
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)
    return 1 / (1 + np.exp(-x))

# Fungsi aktivasi linear
def linear(x, derivative=False):
    x = np.array(x, dtype=float)
    if derivative:
        return np.ones_like(x)
    return x

def get_activation_function(function_name):
    if function_name == "linear":
        return linear
    elif function_name == "relu":
        return relu
    elif function_name == "sigmoid":
        return sigmoid
    else:
        raise Exception("Activation function not found.")