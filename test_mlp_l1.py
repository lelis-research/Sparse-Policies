import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from combo import Game

class NetY1(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(NetY1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
    
    def l1_norm(self, lambda_l1):
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return lambda_l1 * l1_norm

class NetY2(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(NetY2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)
    
    def l1_norm(self, lambda_l1):
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return lambda_l1 * l1_norm

class CustomDataset(Dataset):
    def __init__(self, observations, labels):
        self.observations = observations
        self.labels = labels

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = torch.tensor(self.observations[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return observation, label
    

# Load the dataset from the pickle file
with open('binary/dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Separate observations, y1 labels, and y2 labels
observations = []
y1_labels = []
y2_labels = []

for obs, y1, y2 in data:
    observations.append(obs)
    y1_labels.append(y1)
    y2_labels.append(y2)

# Create the datasets
dataset_y1 = CustomDataset(observations, y1_labels)
dataset_y2 = CustomDataset(observations, y2_labels)

# Using PyTorch's DataLoader to create data loaders for training
batch_size = 1

dataloader_y1 = DataLoader(dataset_y1, batch_size=batch_size, shuffle=False)
dataloader_y2 = DataLoader(dataset_y2, batch_size=batch_size, shuffle=False)

input_size = len(observations[0])  # Size of the observation vector
output_size_y1 = 3  # 3 possible actions for y1
hidden_size = 64  # Hidden layer size
learning_rate = 0.01

# Initialize networks
model_y1 = NetY1(input_size=input_size, output_size=output_size_y1, hidden_size=hidden_size)
model_y2 = NetY2(input_size=input_size, hidden_size=hidden_size)

# Define loss functions
criterion_y1 = nn.CrossEntropyLoss()
criterion_y2 = nn.BCELoss()

# Define optimizers
optimizer_y1 = optim.Adam(model_y1.parameters(), lr=learning_rate)
optimizer_y2 = optim.Adam(model_y2.parameters(), lr=learning_rate)

num_epochs = 100
l1_lambda = 0.0001
# l1_lambda = 0

# Training loop for model_y1
for epoch in range(num_epochs):
    counter = 0
    for observations, y1 in dataloader_y1:
        if counter == 3:
            break
        counter += 1
        # Zero the parameter gradients
        optimizer_y1.zero_grad()

        # Forward pass
        outputs = model_y1(observations)
        y1 = torch.argmax(y1, dim=1)  # CrossEntropyLoss expects class indices, not one-hot
        loss = criterion_y1(outputs, y1) + model_y1.l1_norm(l1_lambda)

        print(observations, y1, outputs)

        # Backward pass and optimization
        loss.backward()
        optimizer_y1.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss Y1: {loss.item():.4f}')


num_epochs = 100

# Training loop for model_y2
for epoch in range(num_epochs):
    for observations, y2 in dataloader_y2:
        # Zero the parameter gradients
        optimizer_y2.zero_grad()

        # print(observations, y2)

        # Forward pass
        outputs = model_y2(observations).view_as(y2) 
        loss = criterion_y2(outputs, y2) + model_y2.l1_norm(l1_lambda)
        # loss += criterion_y2(outputs, y2)

        # Backward pass and optimization
        loss.backward()
        optimizer_y2.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss Y2: {loss.item():.4f}')


for name, param in model_y1.state_dict().items():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param}")

for name, param in model_y2.state_dict().items():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param}")


# problem = "TL-BR"
# problem = "TR-BL"
problem = "BR-TL"
# problem = "BL-TR"
env = Game(3, 3, problem)

for i in range(3):
    for j in range(3):
        env._matrix_unit = np.zeros((3, 3))
        env._matrix_unit[i][j] = 1

        for _ in range(3):
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            prob_actions = model_y1(x_tensor)
            stopping_probability = model_y2(x_tensor)
            a = torch.argmax(prob_actions).item()
            print(a, stopping_probability)
            # print(a)
            env.apply_action(a)