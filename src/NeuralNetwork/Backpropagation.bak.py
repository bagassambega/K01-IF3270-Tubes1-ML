import numpy as np
from activation import get_activation_derivative

class Backpropagation:
    def __init__(self, learning_rate=0.01, momentum=0.0, regularization_lambda=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization_lambda = regularization_lambda
        self.prevWeightUpdates = []
        self.prevBiasUpdates = []

    def compile(self, layers):
        self.layers = layers
        self.prevWeightUpdates = []
        self.prevBiasUpdates = []
        for layer in layers:
            if hasattr(layer, 'weights'):
                self.prevWeightUpdates.append(np.zeros_like(layer.weights))
                self.prevBiasUpdates.append(np.zeros_like(layer.bias))
        # print(self.prevWeightUpdates)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.call(output)
        return output

    def compute_gradients(self, y_true, loss_derivative):
        num_layers = len(self.layers)
        gradients = []
        # Get output from the final layer
        output_layer = self.layers[-1]
        y_pred = output_layer.output
        delta = loss_derivative(y_true, y_pred)

        # Backpropagate through layers in reverse order
        for i in range(num_layers - 1, -1, -1):
            layer = self.layers[i]
            if not hasattr(layer, 'weights'):
                continue
            # Use the activation derivative from activation.py
            activation_deriv_func = get_activation_derivative(layer.activation.__name__)
            activation_deriv = activation_deriv_func(layer.multiplySum)
            delta = delta * activation_deriv
            dW = np.outer(layer.inputs, delta)
            db = delta
            gradients.insert(0, {'dW': dW, 'db': db})
            # Propagate delta to previous layer if exists
            if i > 0 and hasattr(self.layers[i-1], 'weights'):
                delta = np.dot(delta, layer.weights.T)
        return gradients

    def update_weights(self, gradients):
        grad_index = 0
        for i, layer in enumerate(self.layers):
            if not hasattr(layer, 'weights'):
                continue
            dW = gradients[grad_index]['dW']
            db = gradients[grad_index]['db']
            reg_term = self.regularization_lambda * layer.weights if self.regularization_lambda > 0 else 0
            weight_update = self.learning_rate * dW + self.momentum * self.prevWeightUpdates[grad_index]
            bias_update = self.learning_rate * db + self.momentum * self.prevBiasUpdates[grad_index]
            layer.weights -= (weight_update + reg_term)
            layer.bias -= bias_update
            self.prevWeightUpdates[grad_index] = weight_update
            self.prevBiasUpdates[grad_index] = bias_update
            grad_index += 1

    def train_step(self, x, y, loss_derivative):
        prediction = self.forward(x)
        gradients = self.compute_gradients(y, loss_derivative)
        self.update_weights(gradients)
        return prediction, gradients

    def train(self, x_train, y_train, epochs, batch_size, loss_function, loss_derivative, verbose=1):
        history = {'loss': []}
        n_samples = len(x_train)
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                x_batch = x_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                batch_loss = 0
                grad_accum = None
                for j in range(len(x_batch)):
                    self.forward(x_batch[j])
                    y_pred = self.layers[-1].output
                    sample_loss = loss_function(y_batch[j], y_pred)
                    batch_loss += sample_loss
                    grads = self.compute_gradients(y_batch[j], loss_derivative)
                    if grad_accum is None:
                        grad_accum = grads
                    else:
                        for k in range(len(grads)):
                            grad_accum[k]['dW'] += grads[k]['dW']
                            grad_accum[k]['db'] += grads[k]['db']
                for k in range(len(grad_accum)):
                    grad_accum[k]['dW'] /= len(x_batch)
                    grad_accum[k]['db'] /= len(x_batch)
                self.update_weights(grad_accum)
                batch_loss /= len(x_batch)
                epoch_loss += batch_loss * len(x_batch)
            epoch_loss /= n_samples
            history['loss'].append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")
        return history

    def print_parameters(self):
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                print(f"Layer {idx+1} parameters:")
                print("Weights:\n", layer.weights)
                print("Bias:\n", layer.bias)
                print("-" * 40)

    def predict(self, x_data):
        # If x_data is 2D (batch) then predict for each sample.
        # Otherwise, assume single sample.
        if x_data.ndim == 1:
            return self.forward(x_data)
        else:
            predictions = []
            for sample in x_data:
                predictions.append(self.forward(sample))
            return np.array(predictions)
        

