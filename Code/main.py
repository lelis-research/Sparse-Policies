import pickle
import torch
from data.custom_dataset import CustomDataset
from options.options import Option
from utils import setup_environment, run_environment

# Load the dataset
with open('binary/dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Prepare data
observations, y1_labels, y2_labels = [], [], []
for obs, y1, y2 in data:
    observations.append(obs)
    y1_labels.append(y1)
    y2_labels.append(y2)

# Create datasets for y1 and y2
dataset_y1 = CustomDataset(observations, y1_labels)
dataset_y2 = CustomDataset(observations, y2_labels)

# Model parameters
input_size = len(observations[0])  # Size of the observation vector
output_size_y1 = 3  # 3 possible actions for y1
hidden_size = 6  # Hidden layer size
learning_rate = 0.1
num_epochs = 5000
l1_lambda = 0.005
batch_size = 1

# Initialize the Options class with the new hyperparameters
option = Option(input_size, output_size_y1, hidden_size, learning_rate, l1_lambda, batch_size, num_epochs)

# Train models
option.train_y1(dataset_y1)
option.train_y2(dataset_y2)

# Truncate weights with the given threshold (optional, set to 0)
option.truncate_all_weights(threshold=0)

# Print model weights
option.print_model_weights()

# Set up the environment
problem = "TL-BR"  # You can change this to other problem types like "TR-BL", "BR-TL", etc.
env = setup_environment(problem)

# Run the environment using the trained models
run_environment(env, option.model_y1, option.model_y2)