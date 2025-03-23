import numpy as np 
import random 
import math 
from activation import get_activation_function

class Layer: 
    def __init__(self, neurons): 
        self.built = False 
        self.neurons = neurons

    def create(self, inputWeight: np.array): 
        self.inputWeight = inputWeight
        self.built = True

    def call(self, inputs, *args, **kwargs): 
        return inputs

    def get_neuron(self): 
        return self.neurons
    
    # def random_weight(self, shape): 
    #     weights = [] 
    #     for i in range(shape[0] + 1): 
    #         weights.append([])
    #         for j in range(shape[1]): 
    #             weights[i].append(random.uniform(-math.sqrt(6)/math.sqrt(shape[0] + shape[1]), math.sqrt(6)/math.sqrt(shape[0] + shape[1])))
    #     return weights

    def random_weight(self, shape, initMethod = 'default', **kwargs): 
        """
        Menghasilkan bobot (termasuk bias) dengan bentuk (input_dim+1, neurons).
        
        Parameter:
          - shape: tuple, (input_dim, neurons)
          - init_method: string, opsi 'zero', 'uniform', 'normal', atau default ('xavier')
          - kwargs: parameter tambahan tergantung metode:
             * uniform: lower_bound, upper_bound, seed
             * normal: mean, variance, seed
        """
        rows = shape[0] + 1
        cols = shape[1] 

        # Zero initialization
        if initMethod == "zero": 
            return np.zeros((rows, cols))

        # Random dengan distribusi uniform.
        # Menerima parameter lower bound (batas minimal) dan upper bound (batas maksimal)
        # Menerima parameter seed untuk reproducibility
        elif initMethod == "uniform": 
            lowerBound = kwargs.get('lower_bound', -np.sqrt(6)/np.sqrt(shape[0] + shape[1]))
            upperBound = kwargs.get('upper_bound', np.sqrt(6)/np.sqrt(shape[0] + shape[1]))
            seed = kwargs.get('seed', None)

            if seed is not None: 
                random.seed(seed)
            weights = [] 
            for i in range(rows):
                row = []
                for j in range(cols): 
                    row.append(random.uniform(lowerBound, upperBound))
                weights.append(row)
            return np.array(weights)

        # Random dengan distribusi normal.
        # Menerima parameter mean dan variance
        # Menerima parameter seed untuk reproducibility
        elif initMethod == 'normal':
            mean = kwargs.get('mean', 0)
            variance = kwargs.get('variance', 1)
            seed = kwargs.get('seed', None)
            if seed is not None:
                np.random.seed(seed)
            std = math.sqrt(variance)
            return np.random.normal(mean, std, size=(rows, cols))
        
        # Xavier/Glorot initialization
        else:  
            weights = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    row.append(random.uniform(-np.sqrt(6)/np.sqrt(shape[0] + shape[1]),
                                                np.sqrt(6)/np.sqrt(shape[0] + shape[1])))
                weights.append(row)
            return np.array(weights)

class Dense(Layer): 
    def __init__(
            self, 
            neurons, 
            activation=None, 
            inputShape=None, 
            inputWeight:np.array=None,
            weightInit = 'default', 
            weightInitParams=None
            ):
        super().__init__(neurons)
        self.neurons = neurons
        self.activation = get_activation_function(activation)
        self.inputShape = inputShape
        self.inputWeight = inputWeight
        self.weightInit = weightInit
        self.weightInitParams = weightInitParams if weightInitParams is not None else {}
        self.create()
    
    def __repr__(self):
        return ''.join([
            'Layer: \n',
            f'activation = {self.activation.__name__}\n',
            f'weights \n = {self.weights}\n',
            f'bias = {self.bias}\n'
        ])
    
    def create(self):
        if self.inputShape is not None: 
            weightsArr = self.random_weight(shape=(self.inputShape[0], self.neurons),
                                            initMethod=self.weightInit,
                                            **self.weightInitParams)
        else:
            weightsArr = np.array(self.inputWeight)

        # The first row is the bias and the remaining rows are the weights.
        self.bias = weightsArr[0]
        self.weights = weightsArr[1:]
        super().create(weightsArr)


    
    def call(self, inputs: np.array):
        self.inputs = inputs 
        multiplySum = np.dot(inputs, self.weights) + self.bias
        multiplySum += self.bias
        self.multiplySum = multiplySum
        self.output = self.activation(multiplySum)
        return self.output

    


