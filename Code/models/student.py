import torch


class StudentPolicy(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=6):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x