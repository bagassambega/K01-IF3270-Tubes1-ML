import numpy as np
from activation import get_activation_derivative
from Lossfunction import get_loss_derivative, get_loss_function

class Backpropagation:
    def __init__(self, learning_rate: float= 0.01, loss: str='mse', batch_size: int = None, epochs: int = 10, verbose=False):
        self.learning_rate = learning_rate
        self.loss_name = loss
        self.loss_function = get_loss_function(loss)
        self.loss_derivative = get_loss_derivative(loss)
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

    def compile(self, layers):
        self.layers = layers

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.call(output)
            # print(output)
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
        for layer in self.layers:
            if not hasattr(layer, 'weights'):
                continue
                
            dW = gradients[grad_index]['dW']
            db = gradients[grad_index]['db']
            
            layer.weights -= self.learning_rate * dW
            layer.bias -= self.learning_rate * db
            
            grad_index += 1

    def train_step(self, x, y, loss_derivative):
        prediction = self.forward(x)
        gradients = self.compute_gradients(y, loss_derivative)
        self.update_weights(gradients)
        return prediction, gradients

    def train(self, x_train, y_train):
        history = {'loss': []}
        n_samples = len(x_train)
        
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            epoch_loss = 0

            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                x_batch = x_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                batch_loss = 0
                grad_accum = None

                for j in range(len(x_batch)):
                    self.forward(x_batch[j])
                    y_pred = self.layers[-1].output
                    sample_loss = self.loss_function(y_batch[j], y_pred)
                    batch_loss += sample_loss
                    grads = self.compute_gradients(y_batch[j], self.loss_derivative)
                    
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
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - loss: {epoch_loss:.4f}")
                
        return history

    def print_parameters(self):
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                print(f"Layer {idx+1} parameters:")
                print("Weights:\n", layer.weights)
                print("Bias:\n", layer.bias)
                print("-" * 40)

    def predict(self, x_data):
        # print(x_data)
        return self.forward(x_data)
        # if x_data.ndim == 1:
        #     return self.forward(x_data)
        # else:
        #     return np.array([self.forward(sample) for sample in x_data])