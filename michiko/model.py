import numpy as np

from michiko.activations.activation import Activation


class NeuralNetwork(object):

    def __init__(self, **kwargs):

        # user customisable inputs
        self.num_layers = kwargs.get('num_layers', 3)
        self.learning_rate = kwargs.get('learning_rate', 0.01)

        self.num_nodes = 3
        self.num_hidden_layers = 1 if self.num_layers == 3 else 0
        self.size = [self.num_nodes] * self.num_layers
        self.weights = {}
        self.bias = {}
        self.activation = {}
        self.hypothesis = {}

        for i in range(self.num_hidden_layers + 1):
            self.weights[i + 1] = np.random.randn(self.size[i], self.size[i + 1])
            self.bias[i + 1] = np.zeros((1, self.size[i + 1]))

    def forward_pass(self, x):
        self.hypothesis[0] = x.reshape(1, -1)
        for i in range(self.num_hidden_layers):
            self.activation[i+1] = np.matmul(self.hypothesis[i], self.weights[i+1]) + self.bias[i+1]
            self.hypothesis[i+1] = Activation.sigmoid(self.activation[i+1])

        self.activation[self.num_hidden_layers + 1] = np.matmul(
            self.hypothesis[self.num_hidden_layers],
            self.weights[self.num_hidden_layers + 1]
        ) + self.bias[self.num_hidden_layers + 1]

        self.hypothesis[self.num_hidden_layers + 1] = Activation.sigmoid(self.activation[self.num_hidden_layers+1])
        return self.hypothesis[self.num_hidden_layers + 1]

    def predict(self, x):
        return np.array(
            [self.forward_pass(i) for i in x]
        ).squeeze()

    def gradient_sigmoid(self, x):
        return np.multiply(x, np.subtract(1, x))

    def gradient(self, x, y):
        self.forward_pass(x)
        self.delta_w = {}
        self.delta_b = {}
        self.delta_h = {}
        self.delta_a = {}

        dim = self.num_hidden_layers + 1
        self.delta_a[dim] = self.hypothesis[dim] - y
        for k in range(dim, 0, -1):
            self.delta_w[k] = np.matmul(self.hypothesis[k - 1].T, self.delta_a[k])
            self.delta_b[k] = self.delta_a[k]
            self.delta_h[k - 1] = np.matmul(self.delta_a[k], self.weights[k].T)
            self.delta_a[k - 1] = np.multiply(self.delta_h[k - 1], self.gradient_sigmoid(self.hypothesis[k - 1]))

    def mean_squared_error(self, y, y_pred):
        return np.square(np.subtract(y, y_pred)).mean()

    def fit(self, x, y, epoches=500):

        loss = dict()

        for epoch in range(epoches):
            delta_weight = {}
            delta_bias = {}

            for i in range(self.num_hidden_layers + 1):
                delta_weight[i + 1] = np.zeros((self.size[i], self.size[i + 1]))
                delta_bias[i + 1] = np.zeros((1, self.size[i + 1]))

            dict_x_y = zip(x, y)
            for x_i, y_i in dict_x_y:
                self.gradient(x_i, y_i)
                for i in range(self.num_hidden_layers + 1):
                    delta_weight[i + 1] += self.delta_w[i + 1]
                    delta_bias[i + 1] += self.delta_b[i + 1]

            m = x.shape[1]
            for i in range(self.num_hidden_layers + 1):
                self.weights[i + 1] -= self.learning_rate * (delta_weight[i + 1] / m)
                self.bias[i + 1] -= self.learning_rate * (delta_bias[i + 1] / m)

            y_prediction = self.predict(x)
            loss[epoch] = self.mean_squared_error(y, y_prediction)
            if epoch % 20 == 0:
                print (f"Epoch: {epoch}")
                print (f"Loss: {loss[epoch]}")

