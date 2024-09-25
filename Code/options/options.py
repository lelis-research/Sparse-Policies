import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.net_y1 import NetY1
from models.net_y2 import NetY2
from torch.optim import lr_scheduler
import torch.nn as nn
from utils import plot_loss

class Option:
    def __init__(self, problem, trajectory, sequence, input_size, output_size_y1, hidden_size, learning_rate, l1_lambda, batch_size, num_epochs):
        self.problem = problem
        self.sequence = sequence
        self.default_trajectory = trajectory
        self.window_size = len(sequence)
        self.hidden_size = hidden_size

        self.model_y1 = NetY1(input_size=input_size, output_size=output_size_y1, hidden_size=hidden_size)
        self.model_y2 = NetY2(input_size=input_size, hidden_size=hidden_size)

        self.criterion_y1 = nn.CrossEntropyLoss()
        self.criterion_y2 = nn.BCELoss()

        self.optimizer_y1 = optim.SGD(self.model_y1.parameters(), lr=learning_rate, momentum=0.9)
        self.optimizer_y2 = optim.SGD(self.model_y2.parameters(), lr=learning_rate, momentum=0.9)

        self.scheduler_y1 = lr_scheduler.StepLR(self.optimizer_y1, step_size=1000, gamma=0.1)
        self.scheduler_y2 = lr_scheduler.StepLR(self.optimizer_y2, step_size=1000, gamma=0.1)

        self.l1_lambda = l1_lambda
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train_y1(self, dataset_y1):
        dataloader_y1 = DataLoader(dataset_y1, batch_size=self.batch_size, shuffle=False)
        loss_values = []  # List to store loss values for plotting
        for epoch in range(self.num_epochs):
            loss = 0
            epoch_loss = 0  # Accumulate loss for the current epoch
            # counter = 0
            for observations, y1 in dataloader_y1:
                # if counter == 3:
                #     break
                # counter += 1

                # Zero gradients
                self.optimizer_y1.zero_grad()

                # Forward pass
                outputs = self.model_y1(observations)
                y1 = torch.argmax(y1, dim=1)  # CrossEntropy expects class indices
                loss += self.criterion_y1(outputs, y1) + self.model_y1.l1_norm(self.l1_lambda)
                epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            self.optimizer_y1.step()
            self.scheduler_y1.step()

            average_loss = epoch_loss / len(dataloader_y1)
            loss_values.append(average_loss)

            current_lr = self.optimizer_y1.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss Y1: {loss.item():.4f}, LR: {current_lr}')

        title = f'Training Loss Over Epochs - Y1 - hidden size: {self.hidden_size} - sequence: {self.sequence}'
        plot_loss(loss_values, title=title, save_path=f'plots/{title}.png')

    def train_y2(self, dataset_y2):
        dataloader_y2 = DataLoader(dataset_y2, batch_size=self.batch_size, shuffle=False)
        loss_values = []  # List to store loss values for plotting
        for epoch in range(self.num_epochs):
            loss = 0
            epoch_loss = 0  # Accumulate loss for the current epoch
            # counter = 0
            for observations, y2 in dataloader_y2:
                # if counter == 3:
                #     break
                # counter += 1

                # Zero gradients
                self.optimizer_y2.zero_grad()

                # Forward pass
                outputs = self.model_y2(observations).view_as(y2)
                loss += self.criterion_y2(outputs, y2) + self.model_y2.l1_norm(self.l1_lambda)
                epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            self.optimizer_y2.step()
            self.scheduler_y2.step()

            average_loss = epoch_loss / len(dataloader_y2)
            loss_values.append(average_loss)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss Y2: {loss.item():.4f}')

        title = f'Training Loss Over Epochs - Y2 - hidden size: {self.hidden_size} - sequence: {self.sequence}'
        plot_loss(loss_values, title=title, save_path=f'plots/{title}.png')

    def truncate_weights(self, model, threshold):
        """
        Function to truncate small weights in the model to zero
        """
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.where(torch.abs(param.data) < threshold, torch.tensor(0.0), param.data)

    def truncate_all_weights(self, threshold=0):
        """
        Truncate weights for both models (y1 and y2) based on a threshold
        """
        self.truncate_weights(self.model_y1, threshold)
        self.truncate_weights(self.model_y2, threshold)

    def print_model_weights(self):
        """
        Function to print the model weights
        """
        for name, param in self.model_y1.state_dict().items():
            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")

        for name, param in self.model_y2.state_dict().items():
            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")