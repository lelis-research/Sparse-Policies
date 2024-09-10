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

# Dimension of the environment
dim = 3
trajectory_length = 2 * (dim-1) * 3     # 2 = 2-D grid, (dim-1) = number of steps in one direction, 3 = size of the behavior

# Create an array of Options objects with different window sizes
options_list = []

# Loop through different window sizes (from 2 to the length of the trajectory)
for window_size in range(2, trajectory_length):
    # Initialize the Options class with the window size
    option = Option(input_size, output_size_y1, hidden_size, learning_rate, l1_lambda, batch_size, num_epochs, window_size)
    
    # Train the models
    option.train_y1(dataset_y1)
    option.train_y2(dataset_y2)
    
    # Truncate weights with the given threshold (optional, set to 0)
    option.truncate_all_weights(threshold=0)
    
    # Store the options for later evaluation
    options_list.append(option)

# Print model weights for the last trained option as an example (for evaluation purposes)
options_list[-1].print_model_weights()

# Set up the environment
problem = "TL-BR"  # You can change this to other problem types like "TR-BL", "BR-TL", etc.
env = setup_environment(problem, dim)

# Run the environment using the trained models from the last options object
run_environment(env, options_list[-1].model_y1, options_list[-1].model_y2)