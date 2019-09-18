import numpy as np


class Activation(object):

    @staticmethod
    def sigmoid(z):
        signal = np.clip(z, -500, 500)
        return np.divide(1.0, np.add(1.0, np.exp(-signal)))
