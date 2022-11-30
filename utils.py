import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# METRICS

def calculate_r2(predictions, true_labels):
    # corr_matrix = np.corrcoef(true_labels, predictions)
    # corr = corr_matrix[0,1]
    # return corr**2
    return metrics.r2_score(true_labels, predictions)

def calculate_pearson(predictions, true_labels):
    preds = predictions - predictions.mean()
    labels = true_labels - true_labels.mean()
    nom = np.sum(preds * labels)
    denom = math.sqrt(np.sum(np.square(preds)) * np.sum(np.square(labels)))
    return nom / denom

def compute_confusion_matrix(predictions, labels):
    matrix = np.zeros((2, 2))
    matrix[0][0] = sum(predictions[labels==1])
    matrix[0][1] = sum(predictions[labels==0])
    matrix[1][0] = len(labels[labels==1]) - matrix[0][0]
    matrix[1][1] = len(labels[labels==0]) - matrix[0][1]
    return matrix

# LOSS PLOTS

def plot_train_val_loss(hyperparameter_name, hyperparameters, train_loss, val_loss, experiment_no):
    for i in range(len(hyperparameters)):
        plt.plot([x for x in range(len(train_loss[i]))], [loss.data for loss in train_loss[i]], label = f"{hyperparameter_name}: {hyperparameters[i]}")
        plt.plot([x for x in range(len(val_loss[i]))], [loss.data for loss in val_loss[i]], label = f"{hyperparameter_name}: {hyperparameters[i]}", linestyle="--")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f"experiment_results/experiment_{experiment_no}.jpg")
