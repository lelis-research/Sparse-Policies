import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.net_y1 import NetY1
from models.net_y2 import NetY2

class Options:
    def __init__(self, input_size, output_size_y1, hidden_size, learning_rate, l1_lambda, batch_size, num_epochs):
        self.model_y1 = NetY1(input_size=input_size, output_size=output_size_y1, hidden_size=hidden_size)
        self.model_y2 = NetY2(input_size=input_size, hidden_size=hidden_size)
        self.optimizer_y1 = optim.Adam(self.model_y1.parameters(), lr=learning_rate)
        self.optimizer_y2 = optim.Adam(self.model_y2.parameters(), lr=learning_rate)
        self.criterion_y1 = nn.CrossEntropyLoss()
        self.criterion_y2 = nn.BCELoss()
        self.l1_lambda = l1_lambda
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train_y1(self, dataset_y1):
        dataloader_y1 = DataLoader(dataset_y1, batch_size=self.batch_size, shuffle=False)
        for epoch in range(self.num_epochs):
            for observations, y1 in dataloader_y1:
                self.optimizer_y1.zero_grad()
                outputs = self.model_y1(observations)
                y1 = torch.argmax(y1, dim=1)  # CrossEntropy expects class indices
                loss = self.criterion_y1(outputs, y1) + self.model_y1.l1_norm(self.l1_lambda)
                loss.backward()
                self.optimizer_y1.step()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss Y1: {loss.item():.4f}')

    def train_y2(self, dataset_y2):
        dataloader_y2 = DataLoader(dataset_y2, batch_size=self.batch_size, shuffle=False)
        for epoch in range(self.num_epochs):
            for observations, y2 in dataloader_y2:
                self.optimizer_y2.zero_grad()
                outputs = self.model_y2(observations).view_as(y2)
                loss = self.criterion_y2(outputs, y2) + self.model_y2.l1_norm(self.l1_lambda)
                loss.backward()
                self.optimizer_y2.step()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss Y2: {loss.item():.4f}')