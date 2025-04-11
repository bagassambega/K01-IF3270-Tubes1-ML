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

# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         X[i][j] = (X[i][j] - 0)/10

X = np.array(X, dtype=np.float32) / 255.0
y = np.array([float(y[i]) for i in range(len(y))])

temp_x = X[0:1000]
temp_y = y[0:1000]
temp_x_val = X[100:200]
temp_y_val = y[100:200]

ffnn = FFNN(x=temp_x, y=temp_y, x_val=temp_x_val, y_val=temp_y_val, total_layers=[5], loss_function="mse", weight_method="xa", learning_rate=0.1, activations=["relu", "softmax"], verbose=True, epochs=2, seed=42)

ffnn.fit()
# for i, weights in enumerate(ffnn.weights):
#     print(f"Layer {i}:", weights)
test = X[-20:]
real = y[-20:]


result = ffnn.predict(test)
print("Prediction: ", result)
print("Real: ", real)
print(f"FFNN accuracy: {ffnn.accuracy(test, real, 'accuracy')}")
ffnn.save_model("model1.pkl")