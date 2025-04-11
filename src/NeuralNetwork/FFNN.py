from typing import List, Optional
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork.Accuracy import (
    accuracy,
    f1_score,
    log_loss,
)
from NeuralNetwork.Layer import Layer
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
        x_val: Optional[np.ndarray | List],
        y_val: Optional[np.ndarray | List],
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
        randomize: Optional[bool] = False,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0
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
        layers.append(10) # From last hidden layer to output layer. Output layer must be 1
        self.num_layers = layers # All layers: hidden layer + output layer

        if isinstance(activations, List):
            # Misal ada 3 layer (termasuk input & output)
            # Activation akan ada di hidden layer 1 dan output layer saja
            assert len(activations) == len(layers), "Number of activations must be the same \
                with number of layers"
            for i, act in enumerate(activations):
                if act == "softmax" and i != len(layers) - 1:
                    raise ValueError("Can't use softmax except in last layer")
            for act in activations:
                assert act in ["relu", "tanh", "sigmoid", "linear", "softmax"], f"No activation {act} found"
            self.activations = activations
        else:
            assert activations == "softmax", "Cannot using softmax in all layers. Use in last layer only"
            assert activations in ["relu", "tanh", "sigmoid", "linear"], f"No activation \
                {activations} found"
            self.activations = [activations] * len(layers)

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
        self.variance = variance
        self.seed = seed
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.x_val = x_val
        self.y_val = y_val

        # Initialize Regularization
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

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
        self.layers = [Layer(input_dim=x.shape[1], output_dim=layers[0], activation=self.activations[0], weight_method=weight_method, seed=seed)]
        for i in range(1, len(layers)):
            self.layers.append(Layer(input_dim=layers[i-1], output_dim=layers[i], activation=self.activations[i], weight_method=weight_method, seed=seed)) 
        # Loss value for each row of dataset

        # Verbose
        self.verbose: bool = verbose



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
        # if i == 0:
        #     Perlu di-transpose karena nilai yang diambil dari dataset akan sebesar 1 x num_feature
        #     return np.dot(weights, np.array(inputs, dtype=object).reshape(-1, 1)) + bias
        # else:
        return np.dot(weights, inputs) + bias


    def loss(self, loss_method: str, y_true: List[Scalar], y_pred: List[Scalar]) -> Scalar:
        """
        Calculate loss/error with regularization
        """
        # Original loss calculation
        if isinstance(y_true, Scalar) and isinstance(y_pred, Scalar):
            if loss_method == "categorical_cross_entropy":
                loss = categorical_cross_entropy(y_pred=[y_pred], y_true=[y_true])
            elif loss_method == "binary_cross_entropy":
                loss = binary_cross_entropy(y_pred=[y_pred], y_true=[y_true])
            else:
                loss = mse(y_pred=[y_pred], y_true=[y_true])
        else:
            if loss_method == "categorical_cross_entropy":
                loss = categorical_cross_entropy(y_pred=y_pred, y_true=y_true)
            elif loss_method == "binary_cross_entropy":
                loss = binary_cross_entropy(y_pred=y_pred, y_true=y_true)
            else:
                loss = mse(y_pred=y_pred, y_true=y_true)

        # Add L1 regularization
        if self.l1_lambda > 0:
            l1_loss = sum(abs(w) for layer in self.layers for neuron in layer.weights  for w in neuron)
            loss += self.l1_lambda * l1_loss

        # Add L2 regularization
        if self.l2_lambda > 0:
            l2_loss = sum(w**2 for layer in self.layers for neuron in layer.weights for w in neuron)
            loss += self.l2_lambda * l2_loss

        return loss

    def _zero_gradients(self):
        """Reset all gradients to zero before processing a new batch"""
        for layer in self.layers:
            for neuron in layer.weights:
                for w in neuron:
                    w.grad = 0

        for layer in self.layers:
            for b in layer.bias:
                b[0].grad = 0

    def print_weight(self, epoch: int):
        """
        Print weight for debugging purposes
        """
        print(epoch)
        for layer, i in enumerate(self.layers):
            print(f"Layer-{i}:")
            print(layer)

    def softmax(self, logit_array):
        """_summary_

        Args:
            logit_array (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Extract scalar values
        logits = np.array([scalar.value for scalar in logit_array.flatten()])

        # Compute softmax
        exp_logits = np.exp(logits - np.max(logits))  # Stabilized exponentiation
        softmax_probs = exp_logits / np.sum(exp_logits)  # Normalize

        return softmax_probs


    def fit(self):
        """
        Train model with progress bar
        """
        num = len(self.x)
        indices = np.arange(num)
        total_loss = 0

        self.training_losses = []
        self.validation_losses = []  # Only if you are calculating validation loss

        # One-hot encode the labels
        one_hot_y = np.zeros((num, 10))

        for idx, val in enumerate(self.y):
            one_hot_y[idx][val.value.astype(int)] = 1

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
                batch_size = len(batch_indices)
                batch_loss = 0
                total_grade = []

                self._zero_gradients()

                # current_input = self.x[batch_indices]
                # batch_y = one_hot_y[batch_indices]

                # for i, x in enumerate(current_input):
                #     current_input[i] = x.reshape(-1, 1)
                
                # for layer in self.layers:
                #     net, output = layer.forward(current_input)
                #     layer_net_inputs.append(net)
                #     layer_outputs.append(output)
                #     current_input = output


                for i in batch_indices:
                    x_i = self.x[i]
                    y_true = one_hot_y[i]

                    layer_outputs = None

                    current_input = x_i.reshape(-1, 1)

                    for layer in self.layers:
                        output = layer.forward(current_input)
                        layer_outputs = output
                        current_input = output

                    # Apply softmax to the output layer
                    softmax_probs = self.softmax(layer_outputs)
                    for idx, val in enumerate(layer_outputs):
                        val[0].value = softmax_probs[idx]

                    # Calculate loss
                    loss = self.loss(self.loss_function, y_true, layer_outputs)
                    for l in loss:
                        batch_loss += l.value
                        
                    loss[-1].backward()

                # Average gradients over the batch
                batch_size = len(batch_indices)
                for layer in self.layers:
                    for neuron in layer.weights:
                        for w in neuron:
                            w.grad /= batch_size
                    for b in layer.bias:
                        b[0].grad /= batch_size

                # Apply regularization gradients
                if self.l1_lambda > 0 or self.l2_lambda > 0:
                    for layer in self.layers:
                        for neuron in layer.weights:
                            for w in neuron:
                                if self.l1_lambda > 0:
                                    w.grad += self.l1_lambda * np.sign(w.value) / batch_size
                                if self.l2_lambda > 0:
                                    w.grad += self.l2_lambda * w.value / batch_size

                # Update weights and biases
                for layer in self.layers:
                    # Update weights
                    for neuron in layer.weights:
                        for w in neuron:
                            w.value -= self.learning_rate * w.grad
                    # Update biases
                    for b in layer.bias:
                        b[0].value -= self.learning_rate * b[0].grad

                    layer.weights_history.append(layer.weights)
                    

                # Calculate average batch loss
                avg_batch_loss = batch_loss / batch_size
                epoch_loss += avg_batch_loss
                batch_pbar.set_postfix({"Batch Loss": avg_batch_loss})

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / (num / self.batch_size)
            self.training_losses.append(avg_epoch_loss)

            #calculate validation loss 
            if self.x_val is not None and self.y_val is not None: 
                val_loss = self.compute_validation_loss(self.x_val, self.y_val)
                self.validation_losses.append(val_loss)
                epoch_pbar.set_postfix({"Validation Loss": val_loss})

            epoch_pbar.set_postfix({"Epoch Loss": avg_epoch_loss})

        self.plot_loss()
        self.plot_weights()

    def compute_validation_loss(self, X_val, y_val):
        result = []
        y_true = []
        for i in range(len(X_val)): 
            result.append(Scalar(self.predict_single(X_val[i]))) 
            y_true.append(Scalar(y_val[i]))
        return self.loss(self.loss_function, y_true, result).value

    def plot_weights(self):
        # Iterate over each layer and create a separate plot (window) for each layer's weight distribution.
        for idx, layer in enumerate(self.layers):
            # Flatten the weights: from list of list of lists to a single list.
            flattened = []
            for data in layer.weights_history:
                for neuron in data:
                    for weights in neuron:
                        if isinstance(weights, list):
                            for w in weights:
                                flattened.append(w.value)
                        else:
                            flattened.append(weights.value)
            
            # Create a new figure for each layer.
            plt.figure(figsize=(8, 6))
            
            # Since flattened is just one list of numbers, we can create a boxplot out of it.
            # We wrap flattened in a list to satisfy plt.boxplot's input format.
            plt.boxplot(flattened, patch_artist=True)
            
            plt.title(f"Weight Distribution for Layer {idx + 1}")
            plt.xlabel("Weight Distribution")
            plt.ylabel("Weight Value")
            plt.grid(True)
            
            # Display the plot; this will open a new window if you run it in an interactive environment.
            plt.show()

    def plot_loss(self):
        """
        Creating loss plot
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.training_losses) + 1), self.training_losses, label="Training Loss", marker='o')
            
        if self.validation_losses:
            plt.plot(range(1, len(self.validation_losses) + 1), self.validation_losses, label="Validation Loss", marker='s')

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.show()

    def predict_single(self, x):
        """
        Predict target of a single input.
        """
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        layer_result = x  # Start with input
        for j in range(len(self.num_layers)):
            if j == 0:
                layer_result = np.dot(self.layers[0].weights, layer_result) + self.layers[0].bias
            else:
                layer_result = np.dot(self.layers[j].weights, layer_result) + self.layers[j].bias
            layer_result = self.layers[j].activate(layer_result)

        ### DEBUGING
        # print(f"layer_result:\n{layer_result}")

        # Convert Scalar objects to float values if necessary
        logits = np.array([val.value if isinstance(val, Scalar) else val for val in layer_result.flatten()])

        # Apply softmax
        softmax_probs = np.exp(logits - np.max(logits))  # Prevent overflow
        softmax_probs /= np.sum(softmax_probs)

        # Return class with highest probability
        return np.argmax(softmax_probs)

    def predict(self, x):
        """
        Predict class labels for multiple inputs.
        """
        x = np.array(x)

        if x.ndim == 1:
            return self.predict_single(x)

        return np.array([self.predict_single(row) for row in x])

    def save_model(self, filepath: str):
        """
        Save the neural network model to a file using pickle.
        Args:
            filepath (str): Path to the file where the model will be saved
        """
        model_data = {
            'x_train': self.x_train,
            'y_train': self.y_train,
            'layers': self.layers,
            'activations': self.activations,
            'weights': self.weights,
            'bias': self.bias,
            'loss_function': self.loss_function,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'mean': self.mean,
            'variance': self.variance,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'seed': self.seed,
            'verbose': self.verbose,
            'randomize': self.randomize
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a neural network model from a file using pickle.
        Args:
            filepath (str): Path to the file containing the saved model
        Returns:
            FFNN: Reconstructed neural network model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

    
        x = model_data['x']
        y = model_data['y']
        layers = model_data['layers']
        activations = model_data['activations']
        
      
        model = cls(
            x_train=np.array([[v.value for v in row] for row in x]), 
            y_train=np.array([v.value for v in y]),
            layers=layers[:-1],  
            activations=activations,
            batch_size=model_data['batch_size'],
            epochs=model_data['epochs'],
            learning_rate=model_data['learning_rate'],
            weight_method='uniform',
            loss_function=model_data['loss_function'],
            mean=model_data['mean'],
            variance=model_data['variance'],
            lower_bound=model_data['lower_bound'],
            upper_bound=model_data['upper_bound'],
            seed=model_data['seed'],
            verbose=model_data['verbose'],
            randomize=model_data['randomize']
        )
        model.weights = model_data['weights']
        model.bias = model_data['bias']

        return model

    def accuracy(self, x, y_true, acc_method:str, verbose: bool = True):
        """
        X is the data to be predicted, y_true is the true data

        Args:
            x (_type_): _description_
            y_true (_type_): _description_
            acc_method (str): accuracy, f1, logloss
        """
        y_pred = self.predict(x)
        assert len(y_pred) == len(y_true), "Length of x and y_true is not the same"
        score = []
        for i in range(len(y_pred)):
            if verbose:
                print(f"y_pred: {y_pred[i]}, y_true: {y_true[i]}")
            if acc_method == "accuracy":
                score.append(accuracy(y_pred=y_pred[i], y_true=y_true[i]))
            elif acc_method == "f1":
                score.append(f1_score(y_pred=y_pred[i], y_true=y_true[i]))
            else:
                score.append(log_loss(y_pred=y_pred[i], y_true=y_true[i]))

        avg_score = sum(score) / len(score)
        return avg_score