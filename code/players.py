import numpy as np
import util

class Player:
    def __init__(self, model, data_train, label_train, config):
        self.model = model
        self.C = config['C']
        self.m = config['batch_size']
        r = config['r']
        splitable_size = self.m * r
        self.data_train = np.split(data_train[:splitable_size], r)
        self.label_train = np.split(label_train[:splitable_size], r)
        self.mC = self.m * self.C

    def update_gradient(self, gradient):
        self.model.apply_gradient(gradient)

    def calculate_gradient(self, i):
        """Does a single gradient descent iteration"""
        gradient = self.model.gradient_step(self.data_train[i], self.label_train[i])

        assert len(gradient) == 10
        share_rand = [np.random.uniform(-self.mC, self.mC, len(arr)) for arr in gradient]
        rand_gradient = [gradient[j] + share_rand[j] for j in range(len(gradient))]
        return rand_gradient, share_rand

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        return (predictions == y_test).mean()

class Host:
    def __init__(self, config):
        self.gradients = None
        self.mC = config['C'] * config['batch_size']
        self.players = config['players']

    def feed_gradient(self, gradient):
        if self.gradients is None:
            self.gradients = gradient
        else:
            assert len(gradient) == len(self.gradients)
            self.gradients = [self.gradients[i] + gradient[i] for i in range(len(gradient))]

    def calculate_gradient(self):
        sum_gradient = [util.smod(self.gradients[i], self.mC) for i in range(len(self.gradients))]
        self.gradients = None
        return sum_gradient
