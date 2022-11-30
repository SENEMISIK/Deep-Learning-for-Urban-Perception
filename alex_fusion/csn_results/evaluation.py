import numpy as np
import pandas as pd 
import requests
import numpy as np
import math
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv("training_results.csv")
    predictions = df['csn_pred'].to_numpy()
    labels = df['labels'].to_numpy()
    matrix = np.zeros((2, 2))
    matrix[0][0] = sum(predictions[labels==1])
    matrix[0][1] = sum(predictions[labels==0])
    matrix[1][0] = len(labels[labels==1]) - matrix[0][0]
    matrix[1][1] = len(labels[labels==0]) - matrix[0][1]
    print(matrix)
    print("precision: " + str(matrix[0][0]/ (matrix[0][0] + matrix[0][1])))
    print("recall: " + str(matrix[0][0] / (matrix[0][0] + matrix[1][0])))