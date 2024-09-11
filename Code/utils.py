import numpy as np
import torch
from combo import Game
from agent import PolicyGuidedAgent, Trajectory
from models.model import CustomRelu

def setup_environment(problem, dim):
    """
    Set up the Game environment based on the provided problem.
    """
    return Game(dim, dim, problem)


def run_environment(env, model_y1, model_y2):
    """
    Runs the trained models in the environment and prints the chosen actions
    and stopping probabilities for a few iterations.
    """
    for i in range(3):
        for j in range(3):
            env._matrix_unit = np.zeros((3, 3))  # Reset the environment
            env._matrix_unit[i][j] = 1  # Place agent at position (i, j)

            for _ in range(3):  # Perform a few steps in the environment
                # Convert environment observation to tensor
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)

                # Get action probabilities and stopping probability
                prob_actions = model_y1(x_tensor)
                stopping_probability = model_y2(x_tensor)

                # Choose action
                a = torch.argmax(prob_actions).item()

                # Print chosen action and stopping probability
                print(f"Action: {a}, Stopping Probability: {stopping_probability.item()}")

                # Apply the chosen action to the environment
                env.apply_action(a)


def load_trajectories(problems, hidden_size, game_width):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    trajectories = {}
    for problem in problems:
        env = Game(game_width, game_width, problem)
        agent = PolicyGuidedAgent()
        rnn = CustomRelu(game_width**2 * 2 + 9, hidden_size, 3)
        
        rnn.load_state_dict(torch.load('binary/game-width' + str(game_width) + '-' + problem + '-relu-' + str(hidden_size) + '-model.pth'))

        trajectory = agent.run(env, rnn, greedy=True)
        trajectories[problem] = trajectory

    return trajectories


def update_uniq_seq_dict(trajectory, problem, window_size, stride=1, seq_dict=None):
    """
    The unique sequence dictionary is a dictionary that maps a sequence of actions to a tuple containing a model and a list of corresponding states.
    Parameters:
    - It takes a single trajectory and model as input.
    - It extracts the action sequence and state sequence from the trajectory.
    - It creates sliding windows of the action sequence with the specified window size and stride.
    - For each window, it checks if the sequence is already present in the dictionary.
    - If the sequence is not present, it adds the sequence as a key in the dictionary and associates it with the model and the corresponding state.
    - If the sequence is already present, it appends the corresponding state to the existing list of states.
    - seq_dict = {
            seq1: (problem1, [state1, state2, ...]),
            seq2: (problem2, [...]),
            ...
        }
    """
    actions = trajectory.get_action_sequence()
    states = trajectory.get_state_sequence()

    for i in range(0, len(actions) - window_size + 1, stride):
        seq = tuple(actions[i:i+window_size])
        
        # Collect the corresponding sequence of states for each action in the window
        state_tuple = tuple(states[i:i+window_size])
        
        # If the sequence is not in the dictionary, add it with the corresponding state tuple
        if seq not in seq_dict:
            seq_dict[seq] = (problem, [state_tuple])
        else:
            # If the sequence already exists, append the new state tuple to the list
            seq_dict[seq][1].append(state_tuple)

    return seq_dict


def generate_labels(uniq_seq_dict, seq):
    """
    Generate y1 and y2 labels based on the action sequence.
    
    Parameters:
    - uniq_seq_dict: The unique sequence dictionary with action sequences and corresponding states.
    - seq: The specific action sequence (tuple of actions) for which to generate labels.
    
    Returns:
    - y1_labels: A list of one-hot encoded labels for the actions.
    - y2_labels: A list of labels indicating whether the sequence is ongoing (1) or done (0).
    """
    y1_labels = []
    y2_labels = []
    sequence_length = len(seq)
    
    if seq not in uniq_seq_dict:
        raise ValueError(f"Sequence {seq} not found in uniq_seq_dict")

    # Extract the actions and states corresponding to the given sequence
    actions = list(seq)  # Convert the tuple of actions into a list
    _, state_tuples = uniq_seq_dict[seq]  # Extract the list of state tuples for this sequence
    
    # The number of state tuples determines how many times the sequence should be repeated
    repeat_count = len(state_tuples)

    # Generate y1 labels for the action sequence and repeat it across all state tuples
    for _ in range(repeat_count):
        for action in actions:
            y1 = [0, 0, 0]  # Initialize the one-hot encoding
            y1[action] = 1  # Set the action index to 1
            y1_labels.append(y1)

    # Generate y2 labels to indicate whether the sequence is ongoing or done
    for _ in range(repeat_count):
        for i in range(len(actions)):
            if (i + 1) % sequence_length == 0:
                y2 = 0  # End of the sequence
            else:
                y2 = 1  # Sequence is not done
            y2_labels.append(y2)

    return y1_labels, y2_labels


def create_trajectory(sequence_of_actions, states):
    trajectory = Trajectory()
    for action, state in zip(sequence_of_actions, states):
        trajectory.add_pair(state, action)
    return trajectory