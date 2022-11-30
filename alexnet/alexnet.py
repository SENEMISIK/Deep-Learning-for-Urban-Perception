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

class KitModel(nn.Module):

    def __init__(self, dropout):
        super(KitModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, groups=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, groups=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, groups=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, groups=2, padding=1)
        self.fc6_1 = nn.Linear(in_features = 9216, out_features = 4096)
        self.fc7_1 = nn.Linear(in_features = 4096, out_features = 4096)
        self.ip_1 = nn.Linear(in_features = 4096, out_features = 1)
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        conv1           = self.conv1(x)
        relu1           = self.relu(conv1)
        pool1           = self.maxpool(relu1)
        norm1           = self.LRN(size = 5, alpha=0.0001, beta=0.75)(pool1)
        
        conv2           = self.conv2(norm1)
        relu2           = self.relu(conv2)
        pool2           = self.maxpool(relu2)
        norm2           = self.LRN(size = 5, alpha=0.0001, beta=0.75)(pool2)
        
        conv3           = self.conv3(norm2)
        relu3           = self.relu(conv3)
        conv4           = self.conv4(relu3)
        relu4           = self.relu(conv4)
        conv5           = self.conv5(relu4)
        relu5           = self.relu(conv5)
        pool5           = self.maxpool(relu5)
        
        fc6_0           = pool5.view(pool5.size(0), -1)
        
        fc6_1           = self.fc6_1(fc6_0)
        relu6           = self.relu(fc6_1)
        drop6           = self.drop(relu6)
        fc7_1           = self.fc7_1(drop6)
        relu7           = self.relu(fc7_1)
        ip_0            = self.drop(relu7)
        ip_1            = self.ip_1(ip_0)
        
        return ip_1
    
    class LRN(nn.Module):
        def __init__(self, size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
            super(KitModel.LRN, self).__init__()
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

class KitModel_Test(nn.Module):
    
    def __init__(self, dropout):
        super(KitModel_Test, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, groups=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, groups=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, groups=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, groups=2, padding=1)
        self.fc6_1 = nn.Linear(in_features = 9216, out_features = 4096)
        self.fc7_1 = nn.Linear(in_features = 4096, out_features = 4096)
        self.ip_1 = nn.Linear(in_features = 4096, out_features = 1)
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward_helper(self, x):
        conv1           = self.conv1(x)
        relu1           = self.relu(conv1)
        pool1           = self.maxpool(relu1)
        norm1           = self.LRN(size = 5, alpha=0.0001, beta=0.75)(pool1)
        
        conv2           = self.conv2(norm1)
        relu2           = self.relu(conv2)
        pool2           = self.maxpool(relu2)
        norm2           = self.LRN(size = 5, alpha=0.0001, beta=0.75)(pool2)
        
        conv3           = self.conv3(norm2)
        relu3           = self.relu(conv3)
        conv4           = self.conv4(relu3)
        relu4           = self.relu(conv4)
        conv5           = self.conv5(relu4)
        relu5           = self.relu(conv5)
        pool5           = self.maxpool(relu5)
        
        fc6_0           = pool5.view(pool5.size(0), -1)
        
        fc6_1           = self.fc6_1(fc6_0)
        relu6           = self.relu(fc6_1)
        drop6           = self.drop(relu6)
        fc7_1           = self.fc7_1(drop6)
        relu7           = self.relu(fc7_1)
        ip_0            = self.drop(relu7)
        ip_1            = self.ip_1(ip_0)
        
        return ip_1
    
    def forward(self, x):
        if self.forward_helper(x[0]) > self.forward_helper(x[1]):
            return 0
        else:
            return 1
    
    class LRN(nn.Module):
        def __init__(self, size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
            super(KitModel_Test.LRN, self).__init__()
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


class PandasDataset(Dataset):
    def __init__(self, list_images, list_targets, transform=None):
        self.list_images = list_images
        self.list_targets = list_targets
        # add transforms as well
        self.transform = transform

    def __getitem__(self, idx):
        
        image = Image.open(self.list_images[idx]).convert('RGB')
        image = image.resize((227,227), Image.BILINEAR) 
        image = np.array(image, dtype='f4')
        # Convert RGB to BGR 
        image = image[:, :, ::-1]
        
        image = image.astype('float32')
        
        # add transforms
        if self.transform:
            image = self.transform(image)
            
        return image, self.list_targets[idx]

    def __len__(self):
        return len(self.list_images)


class TestModel(pl.LightningModule):
    def __init__(self, dropout, path):
        super().__init__()
        self.model = KitModel_Test(dropout)
        self.model.load_state_dict(torch.load(path))     # change the location of generated_files directory accordingly
       
class LitModel(pl.LightningModule):
    def __init__(self, dropout, weight_decay):
        super().__init__()
        self.model = KitModel(dropout)
        self.model.load_state_dict(torch.load('generated_files/pytorch_state.npy'))     # change the location of generated_files directory accordingly
        self.training_losses = []
        self.validation_losses = []
        self.training_avg_losses = []
        self.validation_avg_losses = []
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(torch.squeeze(y_hat, 1), y.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.training_losses.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(torch.squeeze(y_hat, 1), y.float())
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
    batch_size = 5

    image_name_list = os.listdir('images')      # adjust directory to have the images folder
    image_list = ["images/" + x for x in image_name_list]
    actual_labels = [] 
    df = pd.read_csv("true_skill_labels.csv")      # adjust directory to have the true labels
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
    plt.savefig("alexnet_results/validation_loss.jpg")

    plt.figure(2)
    plt.plot([x for x in range(len(t_loss))], [loss.data for loss in t_loss])
    plt.ylabel('Training loss')
    plt.xlabel('Steps')
    plt.savefig("alexnet_results/training_loss.jpg")

    plt.figure(3)
    plt.plot([x for x in range(len(t_avg_loss))], [loss.data for loss in t_avg_loss], color='green')
    plt.plot([x for x in range(len(v_avg_loss))], [loss.data for loss in v_avg_loss], color='red')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig("alexnet_results/training_vs_testing_loss.jpg")

    predictions = np.zeros((test_size - test_size//2, 1))
    true_labels = np.zeros((test_size - test_size//2, 1))
    ifrom = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        n = len(inputs)
        ito = ifrom + len(inputs)
        inputs = Variable(inputs)
        outputs = model.model(inputs)
        
        predictions[ifrom:ito] = outputs.data.numpy()
        true_labels[ifrom:ito] = np.expand_dims(np.array(labels), 1)

        ifrom = ito
    
    new_df = pd.DataFrame(predictions, columns=['predictions'])
    new_df["labels"] = true_labels
    new_df.to_csv("training_results.csv")

    plt.figure(4)
    plt.scatter(true_labels, predictions)
    plt.ylabel('Predictions')
    plt.xlabel('Labels')
    plt.savefig("alexnet_results/finetuned_evaluation.jpg")