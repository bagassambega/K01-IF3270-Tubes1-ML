from sklearn.datasets import fetch_openml
import os
import joblib  # For saving and loading the dataset
from NeuralNetwork.Autograd import Scalar
from NeuralNetwork.Visualize import draw_dot
import numpy as np
from NeuralNetwork.WeightGenerator import norm


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

# test case 1
# width = 5, depth = 2
X, y = get_dataset()
if X is not None:
    print(f"Dataset {X.shape}, {y.shape} loaded.")
    X = np.array(X)
    y = np.array(y)


X = np.array(X, dtype=np.float32) / 255.0
y = np.array([float(y[i]) for i in range(len(y))])

temp_x = X[0:10000]
temp_y = y[0:10000]

ffnn = FFNN(x=temp_x, y=temp_y, layers=[5], loss_function="mse", weight_method="xavier", learning_rate=0.01, activations=["relu", "softmax"], verbose=True, epochs=2, seed=42)

ffnn.fit()

test = X[-20:]
real = y[-20:]
print(type(real))

print("Accuracy: ", ffnn.accuracy(test, real, "f1"))

ffnn.save_model("case1-depth5-neuron2")