import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import utils 
from utils import plot_train_val_loss

from alexnet.alexnet import PandasDataset
from alexnet.alexnet import LitModel as Alexnet
from alexnet.alexnet import TestModel

from alex_fusion.csn import VotesDataset
from alex_fusion.csn import LitModel as CSN

image_name_list = os.listdir('images')


def finetune_alexnet(batch_size=40, dropout=0.4, weight_decay=15, callback=True, plot=False):
    image_list = ["images/" + x for x in image_name_list]
    actual_labels = [] 
    df = pd.read_csv("true_skill_labels.csv")
    for name in image_name_list:
        row = df.loc[df['id'] == name[:-4]]
        actual_labels.append(float(row["rating"])) 
    torch.manual_seed(69)
    transformations = transforms.ToTensor()

    full_dataset = PandasDataset(image_list, actual_labels, transformations)

    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [test_size//2, test_size - test_size//2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if callback:
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", check_on_train_epoch_end=True)
        trainer = pl.Trainer(max_epochs=10, log_every_n_steps=5, callbacks=[early_stopping])
    else:
        trainer = pl.Trainer(max_epochs=4, log_every_n_steps=5)

    model = Alexnet(dropout=dropout, weight_decay=weight_decay)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    predictions = np.zeros((test_size - test_size//2, 1))
    true_labels = np.zeros((test_size - test_size//2, 1))
    ifrom = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        n = len(inputs)
        ito = ifrom + len(inputs)
        outputs = model.model(inputs)
        predictions[ifrom:ito] = outputs.data.numpy()
        true_labels[ifrom:ito] = np.expand_dims(np.array(labels), 1)
        ifrom = ito

    train_avg_loss = model.training_avg_losses
    val_avg_loss = model.validation_avg_losses

    torch.save(model.model.state_dict(), f"models/alexnet_{batch_size}_{dropout}_{weight_decay}.pth")

    r2_score = utils.calculate_r2(predictions, true_labels)
    pearson_correlation = utils.calculate_pearson(predictions, true_labels)

    if plot:
        train_loss = model.training_losses 
        val_loss = model.validation_losses
        return train_loss, train_avg_loss, val_loss, val_avg_loss, true_labels, predictions, r2_score, pearson_correlation
    
    return train_avg_loss, val_avg_loss, r2_score, pearson_correlation

def train_csn(training_size=0.7, weight_decay=1e-5):
    batch_size = 32
    torch.manual_seed(69)
    image_set = set(image_name_list)

    votes = pd.read_csv("Boston_NY_Data.csv")
    transformations = transforms.ToTensor()
    full_dataset = VotesDataset(votes, image_set, transformations)

    train_size = int(training_size * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [test_size//2, test_size - test_size//2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)
    
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", check_on_train_epoch_end=True)
    trainer = pl.Trainer(max_epochs=10, log_every_n_steps=5, callbacks=[early_stopping])
    model = CSN(weight_decay=weight_decay)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    torch.save(model.model.state_dict(), f"models/csn_model_{training_size}_{weight_decay}.pth")

    train_loss = model.training_avg_losses
    val_loss = model.validation_avg_losses

    predictions = np.zeros((test_size - test_size//2, 2))
    labels = np.zeros((test_size - test_size//2, 2))
    for i, data in enumerate(test_loader):
        input, label = data
        output = model.model(input)
        output = torch.squeeze(output)
        if output[0] > output[1]:
            predictions[i] = 0
        else:
            predictions[i] = 1
        labels[i] = label

    confusion_matrix = utils.compute_confusion_matrix(predictions, labels)

    return train_loss, val_loss, confusion_matrix

def text_alexnet(path):
    torch.manual_seed(69)
    image_set = set(image_name_list)

    votes = pd.read_csv("Boston_NY_Data.csv")
    transformations = transforms.ToTensor()
    full_dataset = VotesDataset(votes, image_set, transformations)

    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, val_test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    _, test_dataset = torch.utils.data.random_split(val_test_dataset, [test_size//2, test_size - test_size//2])
    test_loader = DataLoader(test_dataset, shuffle=False)
    
    model = TestModel(dropout=0.5, path=path)
    model.model.load_state_dict(torch.load(path))
    
    predictions = np.zeros((test_size - test_size//2, 2))
    labels = np.zeros((test_size - test_size//2, 2))

    for i, data in enumerate(test_loader):
        input, label = data
        predictions[i] = model.model(input)
        labels[i] = label

    confusion_matrix = utils.compute_confusion_matrix(predictions, labels)

    return predictions, labels, confusion_matrix


if __name__ == '__main__':

    # # Experiment 1
    # print("Experiment 0")
    # weight_decays = [1, 5, 10, 20]
    # training_loss = []
    # validation_loss = []
    # r2_scores = []
    # pearson_correlations = []
    # for i, l2_regulization in enumerate(weight_decays):
    #     print(i)
    #     t_loss, v_loss, r2, pearson = finetune_alexnet(weight_decay=l2_regulization, callback=False)
    #     training_loss.append(t_loss)
    #     validation_loss.append(v_loss)
    #     r2_scores.append(r2)
    #     pearson_correlations.append(pearson)
    # new_df = pd.DataFrame(weight_decays, columns=['weight_decay'])
    # new_df["r2_score"] = r2_scores
    # new_df["pearson_correlation"] = pearson_correlations
    # new_df.to_csv("experiment_results/experiment_0.csv")
    # plot_train_val_loss("l2 regularization", weight_decays, training_loss, validation_loss, 0)

    # # Experiment 2
    # print("Experiment 2")
    # batch_sizes = [1, 5, 40]
    # training_loss = []
    # validation_loss = []
    # r2_scores = []
    # pearson_correlations = []
    # for i, size in enumerate(batch_sizes):
    #     print(i)
    #     t_loss, v_loss, r2, pearson = finetune_alexnet(batch_size=size)
    #     training_loss.append(t_loss)
    #     validation_loss.append(v_loss)
    #     r2_scores.append(r2)
    #     pearson_correlations.append(pearson)
    # new_df = pd.DataFrame(batch_sizes, columns=['batch_size'])
    # new_df["r2_score"] = r2_scores
    # new_df["pearson_correlation"] = pearson_correlations
    # new_df.to_csv("experiment_results/experiment_2.csv")
    # plot_train_val_loss("batch size", batch_sizes, training_loss, validation_loss, 2)

    # # Experiment 3
    # dropouts = [0.3, 0.5, 0.7]
    # training_loss = []
    # validation_loss = []
    # r2_scores = []
    # pearson_correlations = []
    # for i, dropout in enumerate(dropouts):
    #     print(i)
    #     t_loss, v_loss, r2, pearson = finetune_alexnet(dropout=dropout)
    #     training_loss.append(t_loss)
    #     validation_loss.append(v_loss)
    #     r2_scores.append(r2)
    #     pearson_correlations.append(pearson)
    # new_df = pd.DataFrame(dropouts, columns=['dropout'])
    # new_df["r2_score"] = r2_scores
    # new_df["pearson_correlation"] = pearson_correlations
    # new_df.to_csv("experiment_results/experiment_3.csv")
    # plot_train_val_loss("dropout", dropouts, training_loss, validation_loss, 3)

    # # Experiment 4
    # training_sizes = [0.4, 0.5, 0.6, 0.7]
    # training_loss = []
    # validation_loss = []
    # precisions = []
    # recalls = []
    # for size in training_sizes:
    #     t_loss, v_loss, confusion_matrix = train_csn(training_size=size)
    #     training_loss.append(t_loss)
    #     validation_loss.append(v_loss)
    #     precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    #     recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    #     precisions.append(precision)
    #     recalls.append(recall)
    # new_df = pd.DataFrame(training_sizes, columns=['training_size'])
    # new_df["precision"] = precisions
    # new_df["recall"] = recalls
    # new_df.to_csv("experiment_results/experiment_4.csv")
    # plot_train_val_loss("training size", training_sizes, training_loss, validation_loss, 4)

    # # Experiment 5
    # path = "models/alexnet_40_0.4_10.pth"
    # predictions, labels, confusion_matrix = text_alexnet(path)
    # precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    # recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    # new_df = pd.DataFrame([precision], columns=['precision'])
    # new_df["recall"] = [recall]
    # new_df.to_csv("experiment_results/experiment_5.csv")

    # # Experiment 6
    # batch_size = 40
    # dropout = 0.4
    # weight_decay = 10
    # train_loss, train_avg_loss, val_loss, val_avg_loss, true_labels, predictions, r2_score, pearson_correlation = finetune_alexnet(batch_size=batch_size, dropout=dropout, weight_decay=weight_decay, callback=True, plot=True)
    # plt.figure(1)
    # plt.scatter([x for x in true_labels], [y for y in predictions])
    # plt.ylabel('Predictions')
    # plt.xlabel('Labels')
    # plt.savefig(f"experiment_results/best_model_predictions_vs_labels.jpg")

    # plt.figure(2)
    # plt.plot([x for x in range(len(train_loss))], [loss.data for loss in train_loss])
    # plt.plot([x for x in range(len(val_loss))], [loss.data for loss in val_loss], linestyle="--")
    # plt.ylabel('Loss')
    # plt.xlabel('Steps')
    # plt.savefig(f"experiment_results/best_model_loss_by_step.jpg")

    # plt.figure(3)
    # plt.plot([x for x in range(len(train_avg_loss))], [loss.data for loss in train_avg_loss])
    # plt.plot([x for x in range(len(val_avg_loss))], [loss.data for loss in val_avg_loss], linestyle="--")
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.savefig(f"experiment_results/best_model_loss_by_epoch.jpg")

    # new_df = pd.DataFrame([r2_score], columns=['r2_score'])
    # new_df["pearson"] = [pearson_correlation]
    # new_df.to_csv("experiment_results/best_model_performance.csv")

    # Experiment 7
    weight_decays = [1e-3, 1, 5, 10]
    training_loss = []
    validation_loss = []
    precisions = []
    recalls = []
    for i, l2_regulization in enumerate(weight_decays):
        t_loss, v_loss, confusion_matrix = train_csn(training_size=0.7, weight_decay=l2_regulization)
        training_loss.append(t_loss)
        validation_loss.append(v_loss)
        precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
        precisions.append(precision)
        recalls.append(recall)
    new_df = pd.DataFrame(weight_decays, columns=['weight_decay'])
    new_df["precision"] = precisions
    new_df["recall"] = recalls
    new_df.to_csv("experiment_results/experiment_7.csv")
    plot_train_val_loss("weight decay", weight_decays, training_loss, validation_loss, 7)

