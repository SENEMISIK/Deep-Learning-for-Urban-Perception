import numpy as np
import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt



if __name__ == '__main__':
    df = pd.read_csv("training_results.csv")
    preds = df['predictions'].to_numpy() - df['predictions'].mean()
    labels = df['labels'].to_numpy() - df['labels'].mean()
    nom = np.sum(preds * labels)
    denom = math.sqrt(np.sum(np.square(preds)) * np.sum(np.square(labels)))
    print("Pearson correlation: ")  # 0.24446020170402202 or 0.2643483172286415
    print(nom / denom)

