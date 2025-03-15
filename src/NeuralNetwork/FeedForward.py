import numpy as np 

class KelasFeedforwardNetwork:   # 3 Layer
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_range: tuple[int]=(-0.1, 0.1)):
        """
        input_size (int): number of input features 
        hidden_size (int): number of hidden layer neurons 
        output_size (int): number of output neurons 
        weight_range (tuple): range of random weights 

        """
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.output_size = output_size 
        self.weight_range = weight_range 

        self.weights_input_hidden = np.array([[1, 1], [1, 1]])
        self.bias_hidden = np.array([0, -1])
        self.weight_output = np.array([1, -2])
        self.bias_output = np.array([0])

        # self.debug()
        

    def sigmoid(self, x):
        """Sigmoid activation function [[7]]"""
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def forward(self, X):
        """
        Forward propagation through the network
        :param X: Input batch (numpy array of shape (batch_size, input_size))
        :return: Network output (numpy array of shape (batch_size, output_size))
        """
        # Hidden layer computation
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)
        
        # Output layer computation
        output_input = np.dot(hidden_output, self.weight_output) + self.bias_output
        output = self.relu(output_input)

        print(output) 
        return output
    
    def debug(self): 
        print(self.weights_input_hidden)
        print(self.bias_hidden)
        print()
        print(self.weight_output)
        print(self.bias_output)



class FeedforwardNetwork: 
    def __init__(self, input_size, hidden_sizes, output_size, weight_range=(-0.1, 0.1)):
        """
        :param input_size: Number of input features
        :param hidden_sizes: List of sizes for hidden layers (e.g., [10, 20, 30])
        :param output_size: Number of output neurons
        :param weight_range: Tuple (start, end) for weight initialization
        """
        layers = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        # Create weight matrices and bias vectors for all layers [[5]][[6]]
        for i in range(len(layers)-1):
            w = np.random.uniform(weight_range[0], weight_range[1], 
                                (layers[i], layers[i+1]))
            b = np.random.uniform(weight_range[0], weight_range[1], 
                                (1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        self.debug()

    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x): 
        return np.maximum(0, x)
    
    def forward(self, X):
        """Forward propagation through all layers"""
        activations = X
        for i in range(len(self.weights)-1):
            z = np.dot(activations, self.weights[i]) + self.biases[i]
            activations = self.relu(z)

        # last one 
        z = np.dot(activations, self.weights[-1]) + self.biases[-1]
        output = self.sigmoid(z)

        return output.squeeze()
    
    def debug(self): 
        print(self.weights)
        print(self.biases)
    
