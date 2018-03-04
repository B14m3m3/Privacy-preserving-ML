import numpy as np
import logistic_regression as log_reg
from util import CollabModel

class LogisticClassifierTwoClass(CollabModel):

    def __init__(self, lr, C, order):
        self.w = None
        self.lr = lr
        self.C = C
        self.order = order

    def apply_gradient(self, gradient):
        self.w = self.w - self.lr * gradient

    def gradient_step(self, X_, y, reg=1e-4):
        """ Takes a gradient step on a input
        Args:
          X_: np.array shape (n,d) float - n data points d dimens
          y: array of int shape (n,) - Label
          reg: scalar - regularization parameter
        """
        X = np.c_[np.ones(X_.shape[0]), X_] # adding bias
        if self.w is None:
            self.w = np.zeros(X.shape[1])
        gradient = log_reg.log_cost(X, y, self.w, self.C, self.order, reg=reg)
        return gradient

    def visualize_model(self, ax):
        """ visualize the model by plotting the weight vector as an image """
        ax.imshow(self.w[1:].reshape(28, -1, order='F').T, cmap='bone')


    def predict(self, X_):
        """ Predict class for each data point in X with this model
        For all data points if the output probability is < 0.5 return class 0, else return class 1
        Args:
         X: np.array  shape (n,d)

        Returns:
         predictions: np.array shape (n,) dtype int64
        """
        X = np.c_[np.ones(X_.shape[0]), X_]
        predictions = np.zeros(X.shape[0])

        predictions = np.rint(self.probability(X_))

        assert predictions.shape == (X.shape[0],)
        return predictions.astype('int64')

    def probability(self, X_):
        """ Return the probability for class 1, for point x that is sigmoid(w^interval x)
        for each point x in the input. Needed for all-vs-one

        Args:
         X_: np.array  shape (n,d)
        Returns:
         probs: np.array shape (n,)
        """
        X = np.c_[np.ones(X_.shape[0]), X_] # Add one for bias to the first columns
        probs = np.zeros(X.shape[0])

        probs = X @ self.w

        assert probs.shape == (X.shape[0],)
        return probs

class LogisticClassifier(CollabModel):
    """ Logistic Regression model for more than two classes using one-vs-all """

    def __init__(self, lr, C, order):
        self.classes = 10
        self.models = [LogisticClassifierTwoClass(lr, C, order) for i in range(self.classes)]

    def visualize_model(self, ax):
        """ Plots the model weights on the given axes """
        tr = np.c_[[model.w[1:] for model in self.models]].T
        tr2 = tr.reshape(28, 28, 10, order='F')
        tr3 = np.transpose(tr2, axes=[1, 0, 2])
        ax.imshow(tr3.reshape(28, -1, order='F'), cmap='bone')

    def apply_gradient(self, gradient):
        for idx, model in enumerate(self.models):
            model.apply_gradient(gradient[idx])

    def gradient_step(self, X, y, reg=1e-4):
        """
        Args:
            X = np.array shape(n,d) - n data points with d dimensions
            y: np.array shape(n,) - Labels
            reg: scalar - regularization parameter
        """
        permutation = np.random.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        size = len(X) // 10
        gradient_list = []
        for idx, model in enumerate(self.models):
            y_iter = y[size*idx:size*(idx+1)]
            X_iter = X[size*idx:size*(idx+1), :]
            assert len(X_iter) == len(y_iter)
            y_clasi = [1 if val == idx else 0 for val in y_iter]
            gradient = model.gradient_step(X_iter, y_clasi, reg=reg)
            gradient_list.append(gradient)
        return gradient_list

    def predict(self, X):
        """ Predict class for each data point in X with this model
        Args:
         X: numpy array, shape (n,d) each row is a data point

        Returns:
         predictions: numpy array shape (n,) int, prediction on each input point
        """

        pred = np.zeros(X.shape[0])
        class_probabilities = []

        for value in range(self.classes):
            class_probabilities.append(self.models[value].probability(X))
        pred = np.argmax(class_probabilities, axis=0)

        assert pred.shape == (X.shape[0],)
        return pred
