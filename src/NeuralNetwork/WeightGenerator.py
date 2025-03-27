import random
import math
from typing import Optional
from NeuralNetwork.Autograd import Scalar


def zero_initialization(row_dim: int, col_dim: int):
    """
    Isi semua weight dengan 0

    Args:
        row_dim (int): jumlah output node (di layer n, ga termasuk bias)
        col_dim (int): jumlah input node (di layer n-1)

    Returns:
        tuple of matrix weight and bias
    """
    weight_matrix = [[Scalar(0) for _ in range(col_dim)] for _ in range(row_dim)]
    bias_matrix = [[Scalar(0) for _ in range(1)] for _ in range(row_dim)]

    return weight_matrix, bias_matrix

def one_initialization(row_dim: int, col_dim: int):
    """
    Isi semua weight dengan 1

    Args:
        row_dim (int): jumlah output node (di layer n, ga termasuk bias)
        col_dim (int): jumlah input node (di layer n-1)

    Returns:
        tuple of matrix weight and bias
    """
    weight_matrix = [[Scalar(1) for _ in range(col_dim)] for _ in range(row_dim)]
    bias_matrix = [[Scalar(1) for _ in range(1)] for _ in range(row_dim)]

    return weight_matrix, bias_matrix


def random_uniform_distribution(
    lower_bound: int | float,
    upper_bound: int | float,
    row_dim: int,
    col_dim: int,
    seed: Optional[int | float] = None,
):
    """
    Generate weight dan bias menggunakan random uniform distribution

    Args:
        lower_bound (int | float): batas bawah
        upper_bound (int | float): batas atas
        row_dim (int): jumlah output node
        col_dim (int): jumlah input node
        seed (Optional[int | float], optional): Seed. Defaults to None.

    Returns:
        tuple of matrix weight and bias
    """
    weight_matrix, _ = zero_initialization(row_dim, col_dim)

    if seed is not None:
        random.seed(seed)

    for i in range(row_dim):
        for j in range(col_dim):
            weight_matrix[i][j] = Scalar(
                round(random.uniform(lower_bound, upper_bound), 4)
            )

    # generate bias
    bias = round(random.uniform(lower_bound, upper_bound), 4)
    bias_matrix = [[Scalar(bias) for _ in range(1)] for _ in range(row_dim)]

    return weight_matrix, bias_matrix


def normal_distribution(
    mean: int | float,
    variance: int | float,
    row_dim: int,
    col_dim: int,
    seed: Optional[int | float] = None,
):
    """
    Generate weight dan bias menggunakan distribusi normal

    Args:
        mean (int | float): rata-rata
        variance (int | float): varians
        row_dim (int): jumlah output node
        col_dim (int): jumlah input node
        seed (Optional[int | float], optional): seed. Defaults to None.

    Returns:
        tuple of matrix weight and bias
    """
    weight_matrix, _ = zero_initialization(row_dim, col_dim)

    if seed is not None:
        random.seed(seed)

    for i in range(row_dim):
        for j in range(col_dim):
            weight_matrix[i][j] = Scalar(
                round(random.normalvariate(mean, math.sqrt(variance)), 4)
            )

    # generate bias
    bias = round(random.normalvariate(mean, math.sqrt(variance)), 4)
    bias_matrix = [[Scalar(bias) for _ in range(1)] for _ in range(row_dim)]

    return weight_matrix, bias_matrix

# referensi bonus: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
def xavier_initialization(row_dim: int, col_dim: int, seed=None):
    """
    Generate weight dan bias memakai Xavier initialization

    Args:
        row_dim (int): jumlah output node
        col_dim (int): jumlah input node
        seed (float, optional): seed. Defaults to None.

    Returns:
        tuple of matrix weight and bias
    """
    lower_bound = -1 * (math.sqrt(6) / math.sqrt(row_dim + col_dim))
    upper_bound = math.sqrt(6) / math.sqrt(row_dim + col_dim)

    weight_matrix, bias_matrix = random_uniform_distribution(
        lower_bound, upper_bound, row_dim, col_dim, seed
    )
    return weight_matrix, bias_matrix


def he_initialization(row_dim: int, col_dim: int, seed=None):
    """
    Generate weight dan bias memakai He initialization

    Args:
        row_dim (int): jumlah output node
        col_dim (int): jumlah input node
        seed (float, optional): seed. Defaults to None.

    Returns:
        tuple of matrix weight and bias
    """
    weight_matrix, bias_matrix = normal_distribution(
        0, 2 / row_dim, row_dim, col_dim, seed
    )
    return weight_matrix, bias_matrix
