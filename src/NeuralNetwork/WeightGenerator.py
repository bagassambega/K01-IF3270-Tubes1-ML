import random
import math

#rowDim = jumlah input node (di layer n-1, ga termasuk bias)
#colDim = jumlah output node (di layer n)
def zeroInitialization(rowDim, colDim):
    weightMatrix = [[0 for _ in range(colDim)] for _ in range(rowDim)]
    biasMatrix = [[0 for _ in range(1)] for _ in range(rowDim)]

    return weightMatrix, biasMatrix

def random_uniform_distribution(lower_bound, upper_bound, rowDim, colDim, seed=None):
    weightMatrix = zeroInitialization(rowDim, colDim)

    if seed is not None:
        random.seed(seed)
    
    for i in range(rowDim):
        for j in range(colDim):
            weightMatrix[i][j] = round(random.uniform(lower_bound, upper_bound), 4)
    
    #generate bias
    bias = round(random.uniform(lower_bound, upper_bound), 4)
    biasMatrix = [[bias for _ in range(1)] for _ in range(rowDim)]
            
    return weightMatrix, biasMatrix

def normal_distribution(mean, variance, rowDim, colDim, seed=None):
    weightMatrix = zeroInitialization(rowDim, colDim)

    if seed is not None:
        random.seed(seed)
    
    for i in range(rowDim):
        for j in range(colDim):
            weightMatrix[i][j] = round(random.normalvariate(mean, math.sqrt(variance)), 4)

    #generate bias
    bias = round(random.normalvariate(mean, math.sqrt(variance)), 4)
    biasMatrix = [[bias for _ in range(1)] for _ in range(rowDim)]

    return weightMatrix, biasMatrix

# referensi bonus: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
def xavier_initialization(rowDim, colDim, seed=None):
    lower_bound = -1 * (math.sqrt(6) / math.sqrt(rowDim + colDim))
    upper_bound = math.sqrt(6) / math.sqrt(rowDim + colDim)

    weightMatrix, biasMatrix = random_uniform_distribution(lower_bound, upper_bound, rowDim, colDim, seed)
    return weightMatrix, biasMatrix

def he_initialization(rowDim, colDim, seed=None):
    weightMatrix, biasMatrix = normal_distribution(0, 2/rowDim, rowDim, colDim, seed)
    return weightMatrix, biasMatrix
