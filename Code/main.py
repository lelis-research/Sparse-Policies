import pickle
import torch
from data.custom_dataset import CustomDataset
from models.model import CustomRelu
from options.options import Option
from utils import setup_environment, run_environment, load_trajectories, update_uniq_seq_dict, generate_labels, create_trajectory

# Load the dataset
with open('binary/dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Prepare data
observations, y1_labels, y2_labels = [], [], []
for obs, y1, y2 in data:
    observations.append(obs)
    y1_labels.append(y1)
    y2_labels.append(y2)

# # Create datasets for y1 and y2
# dataset_y1 = CustomDataset(observations, y1_labels)
# dataset_y2 = CustomDataset(observations, y2_labels)

# Model parameters
input_size = len(observations[0])  # Size of the observation vector
output_size_y1 = 3  # 3 possible actions for y1
hidden_size = 6  # Hidden layer size
learning_rate = 0.1
num_epochs = 5000
l1_lambda = 0.005
batch_size = 1

# Environment
game_width = 3
problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
hidden_size_custom_relu = 6

# Create an array of Options objects with different window sizes
options_list = []

rnn = CustomRelu(game_width**2 * 2 + 9, hidden_size_custom_relu, 3)
models = []
uniq_seq_dict = {}

# Keeping track of sequences of actions and the models that generate them
trajectories = load_trajectories(problems, hidden_size_custom_relu, game_width)
for problem, trajectory in trajectories.items():
    rnn = CustomRelu(game_width**2 * 2 + 9, hidden_size_custom_relu, 3)
    rnn.load_state_dict(torch.load('binary/game-width' + str(game_width) + '-' + problem + '-relu-' + str(hidden_size_custom_relu) + '-model.pth'))
    models.append(rnn)

    print("Problem:", problem)
    print("actions: ", trajectory.get_action_sequence(), " \n")

    window_sizes = list(range(2, len(trajectory.get_trajectory())))

    # Loop through different window sizes (from 2 to the length of the trajectory)
    for ws in window_sizes:
        uniq_seq_dict = update_uniq_seq_dict(trajectory, rnn, problem, ws, seq_dict=uniq_seq_dict)
    

# loop through sequeces and create options
for seq, (problem, model, states) in uniq_seq_dict.items():

    # Initialize the Option object with the window size
    option = Option(problem, trajectory, seq, input_size, output_size_y1, hidden_size, learning_rate, l1_lambda, batch_size, num_epochs)

    # Each option has different dataset
    observations = [state.get_observation() for state in states] 
    y1_labels, y2_labels = generate_labels(create_trajectory(seq, states))
    
    dataset_y1 = CustomDataset(observations, y1_labels)
    dataset_y2 = CustomDataset(observations, y2_labels)
    
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
env = setup_environment(problem, game_width)

# Run the environment using the trained models from the last options object
run_environment(env, options_list[-1].model_y1, options_list[-1].model_y2)