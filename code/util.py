import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata


def fetch_mnist_data():
    """Loads the minst data"""
    custom_data_home = "./data"
    mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
    data = mnist.data
    label = mnist.target
    return data, label

def train_and_validation_split(data, label, n_samples=70000, val_size=10000):
    """Randomizes and splits the data for train and validation, default all, and 10000 validation"""
    rand_data, rand_label = shuffle(data, label, n_samples=n_samples, random_state=0)

    #test used for validation
    data_val = np.array(rand_data[0:val_size, :])
    label_val = np.array(rand_label[0:val_size])

    #used for training
    data_train = np.array(rand_data[val_size:, :])
    label_train = np.array(rand_label[val_size:])
    return data_train, label_train, data_val, label_val

def split_train_data(data, label, players):
    shares_data_points = (len(data)//players) * players
    data_split = np.split(data[:shares_data_points], players)
    label_split = np.split(label[:shares_data_points], players)
    return data_split, label_split

def export_dataframe(name, dataframe):
    """ Trivial helper function for exporting pandas data frames to the right folder """
    result_path = 'results'
    my_path = os.path.join(result_path, name)
    dataframe.to_csv(my_path, index=False)

def export_fig(name, fig):
    """ Trivial helper function for exporting figures to the right folder """
    result_path = 'results'
    my_path = os.path.join(result_path, name)
    fig.savefig(my_path)

def model_accuracy(model, data, label):
    """Tests a models accuracy"""
    pred = model.predict(data)
    acc = np.mean(pred == label)
    return acc

def visualizeData(data, labels, filt=None):
    # Visualize the first 16 two's from training set
    if filt != None:
        idx1 = (labels == filter)
        img2s = data[idx1, :]
    else:
        img2s = data

    x2 = img2s[0:16, :].reshape(-1, 28, 28)
    x2 = x2.transpose(1, 0, 2)
    plt.imshow(x2.reshape(28, -1), cmap='bone')
    plt.yticks([])
    plt.xticks([])
    plt.show()

def clip(C, gradient, order):
    if order == 3:
        order = np.inf
    if order == 0:
        order = 1
    clipped_gradient = gradient * np.minimum(1, C/np.linalg.norm(gradient, ord=order))
    return clipped_gradient

def randomness(size, b, order=0):
    if order == 1 or order == 3:
        return np.random.laplace(0, b, size=size)
    elif order == 2:
        return np.random.normal(0, b, size=size)
    else:
        return 0

def calculate_b(C, epsilon, delta, dimens, order=0):
    if order == 1:
        #(epsilon, 0)-DP
        return C/epsilon
    elif order == 2:
        #(epsilon, delta)-DP
        return C * np.sqrt(2 * np.log(1.25/delta)) / epsilon
    elif order == 3:
        #(epsilon, delta)-DP
        return (C * np.sqrt(dimens) * (np.sqrt(np.log10(1/delta))+np.sqrt(np.log10(1/delta) + 2 * epsilon)))/(epsilon * np.sqrt(2))

def smod(x, C):
    assert type(x).__module__ == np.__name__
    return ((x + C) % (2*C)) - C

class CollabModel:
    def gradient_step(self, X, y, reg):
        raise NotImplementedError

    def apply_gradient(self, gradient):
        raise NotImplementedError

    def visualize_model(self, ax):
        raise NotImplementedError

    def predict(self, X):
        """ Compute the predictions on data X
        Args:
        X: np.array shape (n,d)
        Returns:
        pred: np.array  shape (n,) dtype = int64
        """
        raise NotImplementedError
