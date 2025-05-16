import torch
from torch.utils.data import Dataset

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
    

# Demonstratio dataset for the DAGGER approach
class DemonstrationDataset(Dataset):
    def __init__(self):
        self.observations = []
        self.actions = []

    def add(self, obs, action):
        self.observations.append(obs)
        self.actions.append(action)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        return obs, action