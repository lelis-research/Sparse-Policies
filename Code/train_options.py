import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import torch
from models.model import CustomRelu
from utils import setup_environment, run_environment, load_trajectories, update_uniq_seq_dict, generate_labels, process_option


# Environment
game_width = 3
problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
hidden_size_custom_relu = 32

# Model parameters
# input_size = len(observations[0])  # Size of the observation vector
input_size = game_width**2 * 2 + 9 # (3*3) for agent position + (3*3) for goal position + (3*3) for 3 actions of size 3
output_size_y1 = 3  # 3 possible actions for y1
learning_rate = 0.1
num_epochs = 5000
l1_lambda = 0.005
batch_size = 1
multi_problem = True

# Create an array of Options objects with different window sizes
options_list = []
models = []
uniq_seq_dict = {}

# Keeping track of sequences of actions and the models that generate them
trajectories = load_trajectories(problems, hidden_size_custom_relu, game_width)
for problem, trajectory in trajectories.items():
    rnn = CustomRelu(game_width**2 * 2 + 9, hidden_size_custom_relu, 3)
    rnn.load_state_dict(torch.load('binary/game-width' + str(game_width) + '-' + problem + '-relu-' + str(hidden_size_custom_relu) + '-lr-' + str(l1_lambda) + '-model.pth'))
    models.append(rnn)

    print("Problem:", problem)
    print("actions: ", trajectory.get_action_sequence(), " \n")

    # window_sizes = list(range(2, len(trajectory.get_trajectory())))
    window_sizes = [3]

    # Loop through different window sizes (from 2 to the length of the trajectory)
    for ws in window_sizes:
        uniq_seq_dict = update_uniq_seq_dict(trajectory, problem, ws, seq_dict=uniq_seq_dict, multi_problem=multi_problem)
    

if multi_problem:
    for seq, problem_dict in uniq_seq_dict.items():
        for problem, states in problem_dict.items():
            option = process_option(uniq_seq_dict, problem, seq, states, input_size, output_size_y1, hidden_size_custom_relu, learning_rate, l1_lambda, batch_size, num_epochs)
            options_list.append(option)
else:
    for seq, (problem, states) in uniq_seq_dict.items():
        option = process_option(uniq_seq_dict, problem, seq, states, input_size, output_size_y1, hidden_size_custom_relu, learning_rate, l1_lambda, batch_size, num_epochs)
        options_list.append(option)


# Save the options list to a file
save_path = 'binary/options_list_hidden_size_' + str(hidden_size_custom_relu) + '_game_width_' + str(game_width) + '_num_epochs_' + str(num_epochs) + '-lr-' + str(l1_lambda) + '_onlyws3.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(options_list, f)

print(f'Options list saved to {save_path}')

# Print model weights for the last trained option as an example (for evaluation purposes)
options_list[-1].print_model_weights()

# Set up the environment
problem = "TL-BR"  # You can change this to other problem types like "TR-BL", "BR-TL", etc.
env = setup_environment(problem, game_width)

# Run the environment using the trained models from the last options object
run_environment(env, options_list[-1].model_y1, options_list[-1].model_y2)