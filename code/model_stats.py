import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def export_dataframe(name, df):
    result_path = 'results'
    my_path = os.path.join(result_path, name)
    df.to_csv(my_path, index=False)

def export_fig(name, fig):
    result_path = 'results'
    my_path = os.path.join(result_path, name)
    fig.savefig(my_path)

def visualize_model(name, model):
    """ Visualize a model after training for at short time """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    model.visualize_model(ax)
    export_fig(name, fig)

def confusion_mat(name, model, X_eval, y_eval):
    pred_test = model.predict(X_eval)
    confusion = confusion_matrix(y_eval, pred_test)
    df_confusion = pd.DataFrame(confusion)
    export_dataframe(name, df_confusion)
