from cgi import test
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np


class exampleNet(pl.LightningModule):
    def __init__(self, hparams, input_size=3*32*32, num_classes=10):
        """A simple CNN for Cifar10 classification.

        Args:
            input_size (_type_, optional): _description_. Defaults to 3*32*32.
            output_size (int, optional): _description_. Defaults to 10.
        """
        super().__init__()
        
        self.hparams = hparams

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(.25),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(.25),
            nn.Linear(84, num_classes),
            nn.Softmax()
        )


    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        return optimizer

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'loss': loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        return {"val_loss": val_loss}
    
    def validation_end(self, outputs):
        avg_loss = self.general_end(outputs, "val")
        #print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        return {"test_loss": test_loss}

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred

    def prepare_data(self):

        # create dataset
        CIFAR_ROOT_TRAIN = "../datasets/cifar-10-batches-py/train"
        CIFAR_ROOT_TEST = "../datasets/cifar-10-batches-py/test"

        mean = [0.5, 0.5, 0.5]
        std = [0.229, 0.224, 0.225]

        train_val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        cifar_train_val = torchvision.datasets.ImageFolder(root=CIFAR_ROOT_TRAIN, transform=train_val_transform)
        cifar_test = torchvision.datasets.ImageFolder(root=CIFAR_ROOT_TEST, transform=train_val_transform)

        N = len(cifar_train_val)        
        num_train = int(N*0.8)
        np.random.seed(0)
        indices = np.random.permutation(N)
        train_idx, val_idx = indices[:num_train], indices[num_train:]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        M = len(cifar_test)
        test_indices = np.random.permutation(M)
        test_sampler = SubsetRandomSampler(test_indices)

        self.sampler = {"train": train_sampler, "val": val_sampler, "test": test_sampler}


        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = cifar_train_val, cifar_train_val, cifar_test

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.hparams["batch_size"], sampler=self.sampler["train"])

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"], sampler=self.sampler["val"])
    
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"], sampler=self.sampler["test"])

    def getTestAcc(self, loader = None):

        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc

    def testModel(self):
        _, val_acc = self.getTestAcc(self.val_dataloader())
        print("Validation-Accuracy: {}%".format(val_acc*100))
       
        _, test_acc = self.getTestAcc()
        print("Test-Accuracy: {}%".format(test_acc*100))