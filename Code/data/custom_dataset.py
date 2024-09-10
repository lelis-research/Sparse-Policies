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