import numpy as np

def accuracy(y_true, y_pred) -> float:
    """ Compute accuracy between y_true and y_pred """
    return int(round(y_pred)) == y_true

def f1_score(y_true, y_pred) -> float:
    """ Compute F1 score between y_true and y_pred """
    tp = int(round(y_pred)) == 1 and y_true == 1
    fp = int(round(y_pred)) == 1 and y_true == 0
    fn = int(round(y_pred)) == 0 and y_true == 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return f1

def log_loss(y_true, y_pred):
    """ Compute Log Loss between y_true and y_pred """
    epsilon = 1e-9  # To prevent log(0)
    loss = -(y_true * (y_pred + epsilon).log() + (1 - y_true) * ((1 - y_pred) + epsilon).log())
    return loss
