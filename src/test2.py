from sklearn.datasets import fetch_openml

import os
import joblib  # For saving and loading the dataset
from NeuralNetwork.Autograd import Scalar
from NeuralNetwork.Visualize import draw_dot
import numpy as np
from NeuralNetwork.WeightGenerator import normal_distribution
from NeuralNetwork.FFNN import *
def get_dataset(name: str = 'mnist_784'):
    """Get dataset from OpenML, checking local data folder first.

    Args:
        name (str): Name of the dataset on OpenML.
    """
    cwd: str = os.getcwd()
    data_dir = os.path.join(cwd, 'data')
    dataset_path = os.path.join(data_dir, f"{name}.joblib")  # Path to save/load dataset

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(dataset_path):
        print(f"Loading dataset '{name}' from local file...")
        X, y = joblib.load(dataset_path)
        return X, y
    else:
        print(f"Fetching dataset '{name}' from OpenML...")
        try:
            dataset = fetch_openml(name, version=1, return_X_y=True, data_home=data_dir)
            X, y = dataset
            joblib.dump((X, y), dataset_path)  # Save the dataset
            return X, y
        except Exception as e:
            return False

X, y = get_dataset()
if X is not None:
    print(f"Dataset {X.shape}, {y.shape} loaded.")
    X = np.array(X)
    y = np.array(y)

test = X[-20:]
real = y[-20:]


model = FFNN.load_model("model1.pkl")
print(model.x)
print(model.y)
print(model.activations)
print(model.total_layers)
print(model.epochs)
print(model.weight_method)
print(model.learning_rate)
print(model.loss_function)
print(model.layers)

print(f"FFNN accuracy: {model.accuracy(test, real, 'accuracy')}")

# print(model.learning_rate)
