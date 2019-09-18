import numpy as np


def accuracy(y_true, y_predict):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_predict[i]:
            correct += 1
    return correct/float(len(y_true)) * 100
