import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib.pyplot import imshow
import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CSNModel(nn.Module):

    def __init__(self):
        super(CSNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, groups=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, groups=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, groups=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, groups=2, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fusion_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, groups=1, padding=1)
        self.fusion_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, groups=1, padding=1)
        self.fusion_3 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, stride=1, groups=1, padding=1)
        self.fusion_pred = nn.Conv2d(2, 2, (6, 6))

        nn.init.kaiming_normal_(self.fusion_1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(self.fusion_2.bias, 0.01)
        nn.init.kaiming_normal_(self.fusion_2.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(self.fusion_2.bias, 0.01)
        nn.init.kaiming_normal_(self.fusion_3.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(self.fusion_3.bias, 0.01)
        nn.init.xavier_normal_(self.fusion_pred.weight)
        torch.nn.init.constant_(self.fusion_pred.bias, 0.01)

    def forward(self, x):
        conv1_0           = self.conv1(x[0])
        relu1_0           = self.relu(conv1_0)
        pool1_0           = self.maxpool(relu1_0)
        norm1_0           = self.LRN(size = 5, alpha=0.0001, beta=0.75)(pool1_0)
        conv2_0           = self.conv2(norm1_0)
        relu2_0           = self.relu(conv2_0)
        pool2_0           = self.maxpool(relu2_0)
        norm2_0           = self.LRN(size = 5, alpha=0.0001, beta=0.75)(pool2_0)
        conv3_0           = self.conv3(norm2_0)
        relu3_0           = self.relu(conv3_0)
        conv4_0           = self.conv4(relu3_0)
        relu4_0           = self.relu(conv4_0)
        conv5_0           = self.conv5(relu4_0)
        relu5_0           = self.relu(conv5_0)
        pool5_0           = self.maxpool(relu5_0)

        conv1_1           = self.conv1(x[1])
        relu1_1           = self.relu(conv1_1)
        pool1_1           = self.maxpool(relu1_1)
        norm1_1           = self.LRN(size = 5, alpha=0.0001, beta=0.75)(pool1_1)
        conv2_1           = self.conv2(norm1_1)
        relu2_1           = self.relu(conv2_1)
        pool2_1           = self.maxpool(relu2_1)
        norm2_1           = self.LRN(size = 5, alpha=0.0001, beta=0.75)(pool2_1)
        conv3_1           = self.conv3(norm2_1)
        relu3_1           = self.relu(conv3_1)
        conv4_1           = self.conv4(relu3_1)
        relu4_1           = self.relu(conv4_1)
        conv5_1           = self.conv5(relu4_1)
        relu5_1           = self.relu(conv5_1)
        pool5_1           = self.maxpool(relu5_1)

        fusion_concat     = torch.concat((pool5_0, pool5_1), 1)

        fusion_conv_1     = self.fusion_1(fusion_concat)
        fusion_relu_1     = self.relu(fusion_conv_1)
        fusion_conv_2     = self.fusion_2(fusion_relu_1)
        fusion_relu_2     = self.relu(fusion_conv_2)
        fusion_conv_3     = self.fusion_3(fusion_relu_2)
        fusion_relu_3     = self.relu(fusion_conv_3)
        
        fusion_pred       = self.fusion_pred(fusion_relu_3)

        return fusion_pred
    
    class LRN(nn.Module):
        def __init__(self, size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
            super(CSNModel.LRN, self).__init__()
            self.ACROSS_CHANNELS = ACROSS_CHANNELS
            if self.ACROSS_CHANNELS:
                self.average=nn.AvgPool3d(kernel_size=(size, 1, 1),
                        stride=1,
                        padding=(int((size-1.0)/2), 0, 0))
            else:
                self.average=nn.AvgPool2d(kernel_size=size,
                        stride=1,
                        padding=int((size-1.0)/2))
            self.alpha = alpha
            self.beta = beta

        def forward(self, x):
            if self.ACROSS_CHANNELS:
                div = x.pow(2).unsqueeze(1)
                div = self.average(div).squeeze(1)
                div = div.mul(self.alpha).add(1.0).pow(self.beta)
            else:
                div = x.pow(2)
                div = self.average(div)
                div = div.mul(self.alpha).add(1.0).pow(self.beta)
            x = x.div(div)
            return x


class VotesDataset(Dataset):
    def __init__(self, votes, image_set, transform=None):
        pairs = []
        targets = []
        for _, place in votes.iterrows():
            if place["left"] + ".png" in image_set and place["right"] + ".png" in image_set:
                left = "images/" + place["left"] + ".png" 
                right = "images/" + place["right"] + ".png" 
                choice = place['choice']
                if choice == "right":
                    pairs.append((left, right))
                    pairs.append((right, left))
                    targets.append(1)
                    targets.append(0)
                elif choice == "left":
                    pairs.append((left, right))
                    pairs.append((right, left))
                    targets.append(0)
                    targets.append(1)
        self.pairs = pairs
        self.list_targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        image_0 = Image.open(self.pairs[idx][0]).convert('RGB')
        image_0 = image_0.resize((227,227), Image.BILINEAR) 
        image_0 = np.array(image_0, dtype='f4')
        # Convert RGB to BGR 
        image_0 = image_0[:, :, ::-1]
        
        image_0 = image_0.astype('float32')
        
        # add transforms
        if self.transform:
            image_0 = self.transform(image_0)

        image_1 = Image.open(self.pairs[idx][1]).convert('RGB')
        image_1 = image_1.resize((227,227), Image.BILINEAR) 
        image_1 = np.array(image_1, dtype='f4')
        # Convert RGB to BGR 
        image_1 = image_1[:, :, ::-1]
        
        image_1 = image_1.astype('float32')
        
        # add transforms
        if self.transform:
            image_1 = self.transform(image_1)
            
        return (image_0, image_1), self.list_targets[idx]

    def __len__(self):
        return len(self.pairs)


class LitModel(pl.LightningModule):
    def __init__(self, weight_decay):
        super().__init__()
        self.model = CSNModel()
        pretrained_dict = torch.load('generated_files/pytorch_state.npy') # change the location of generated_files directory accordingly
        model_dict = self.model.state_dict()
        for k, _ in model_dict.items():
            # 1. filter out unnecessary keys
            if k in pretrained_dict:
                # 2. overwrite entries in the existing state dict
                model_dict[k] = pretrained_dict[k]
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)
        self.training_losses = []
        self.validation_losses = []
        self.training_avg_losses = []
        self.validation_avg_losses = []
        self.weight_decay = weight_decay

    def training_step(self, batch, i):
        x, y = batch
        y_hat = self.model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(torch.squeeze(y_hat), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.training_losses.append(loss)
        return loss

    def validation_step(self, batch, i):
        x, y = batch
        y_hat = self.model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(torch.squeeze(y_hat), y)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.validation_losses.append(loss)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.validation_avg_losses.append(loss)
    
    def training_epoch_end(self, outputs):
        # OPTIONAL
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.training_avg_losses.append(loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=self.weight_decay)


if __name__ == '__main__':

    batch_size = 32
    torch.manual_seed(69)
    image_name_list = os.listdir('images')      # adjust the directory to have the images folder
    image_set = set(image_name_list)

    votes = pd.read_csv("Boston_NY_Data.csv")       # adjust the directory to include the data
    transformations = transforms.ToTensor()
    full_dataset = VotesDataset(votes, image_set, transformations)

    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [test_size//2, test_size - test_size//2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)
    
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", check_on_train_epoch_end=True)
    trainer = pl.Trainer(max_epochs=10, log_every_n_steps=5, callbacks=[early_stopping])
    model = LitModel()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    t_loss = model.training_losses
    v_loss = model.validation_losses
    t_avg_loss = model.training_avg_losses
    v_avg_loss = model.validation_avg_losses

    plt.figure(1)
    plt.plot([x for x in range(len(v_loss))], [loss.data for loss in v_loss])
    plt.ylabel('Validation Loss')
    plt.xlabel('Steps')
    plt.savefig("csn_results/validation_loss.jpg")

    plt.figure(2)
    plt.plot([x for x in range(len(t_loss))], [loss.data for loss in t_loss])
    plt.ylabel('Training loss')
    plt.xlabel('Steps')
    plt.savefig("csn_results/training_loss.jpg")

    plt.figure(3)
    plt.plot([x for x in range(len(t_avg_loss))], [loss.data for loss in t_avg_loss], color='green')
    plt.plot([x for x in range(len(v_avg_loss))], [loss.data for loss in v_avg_loss], color='red')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig("csn_results/train_vs_val_loss.jpg")

    predictions = np.zeros((test_size - test_size//2, 1))
    labels = np.zeros((test_size - test_size//2, 1))
    ifrom = 0
    for i, data in enumerate(test_loader):
        input, label = data
        output = model.model(input)
        output = torch.squeeze(output)
        if output[0] > output[1]:
            predictions[i] = 0
        else:
            predictions[i] = 1
        labels[i] = label

    new_df = pd.DataFrame(labels, columns=['labels'])
    new_df["csn_pred"] = predictions
    new_df.to_csv("csn_results/training_results.csv")

    



    