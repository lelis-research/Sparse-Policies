import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
from environment.combo import Game
from agent import PolicyGuidedAgent, Trajectory
from models.model import CustomRelu
from data.custom_dataset import CustomDataset
from options.options import Option
import logging
import io


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


def load_trajectories(problems, args):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    trajectories = {}
    for problem in problems:
        env = Game(args.game_width, args.game_width, problem)
        agent = PolicyGuidedAgent()
        rnn = CustomRelu(args.game_width**2 * 2 + 9, args.hidden_size, 3)
        
        rnn.load_state_dict(torch.load('binary/NN-game-width' + str(args.game_width) + '-' + problem + '-relu-' + str(args.hidden_size) + '-l1-' + str(args.l1) + '-lr-' + str(args.lr) + '-model.pth'))
        # rnn.load_state_dict(torch.load('binary/game-width' + str(game_width) + '-' + problem + '-relu-' + str(hidden_size) + '-model.pth'))
        trajectory = agent.run(env, rnn, greedy=True)
        trajectories[problem] = trajectory

    return trajectories


def update_uniq_seq_dict(trajectory, problem, window_size, stride=1, seq_dict=None, multi_problem=False):
    """
    The unique sequence dictionary is a dictionary that maps a sequence of actions to a tuple containing the problem and a list of corresponding states.
    Parameters:
    - It takes a single trajectory and problem as input.
    - It extracts the action sequence and state sequence from the trajectory.
    - It creates sliding windows of the action sequence with the specified window size and stride.
    - For each window, it checks if the sequence is already present in the dictionary.
    - If the sequence is not present, it adds the sequence as a key in the dictionary and associates it with the model and the corresponding states.
    - If the sequence is already present, it appends the corresponding states to the existing list of state tuples.
    - multi_problem: If True
    - uniq_seq_dict = {seq: [(problem, state_tuple), (problem, state_tuple), ...]}
    
    Example of seq_dict structure (with multi_problem=True):
    {
        seq1: {Prob1: [(s1, s2, ...), (s1, s2, ...), ...], Prob2: [(s1, s2, ...), ...]},
        seq2: {Prob1: [...], Prob2: [...]},
        ...
    }
    """
    if seq_dict is None:
        seq_dict = {}
    
    actions = trajectory.get_action_sequence()
    states = trajectory.get_state_sequence()

    for i in range(0, len(actions) - window_size + 1, stride):
        seq = tuple(actions[i:i+window_size])
        
        # Collect the corresponding sequence of states for each action in the window
        state_tuple = tuple(states[i:i+window_size])
        
        if multi_problem:
            if seq not in seq_dict:
                # Initialize the sequence with an empty list to store tuples of (problem, state_tuple)
                seq_dict[seq] = []
            
            seq_dict[seq].append((problem, state_tuple))
        
        else:
            # If the sequence is not in the dictionary, add it with the corresponding state tuple
            if seq not in seq_dict:
                seq_dict[seq] = (problem, [state_tuple])
            else:
                # If the sequence already exists, append the new state tuple to the list
                seq_dict[seq][1].append(state_tuple)

    return seq_dict


def generate_labels(uniq_seq_dict, seq, problem, multi_problem, state_tuple=None):
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
    if multi_problem:
        state_tuples = [state_tuple]
    else:
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


def group_options_by_problem(options_list):
    """
    Groups the options in the options_list by their associated problems.
    
    Parameters:
    - options_list: List of Option objects.

    Returns:
    - problems_options: A dictionary where each key is a problem and the value is a list of options associated with that problem.
      problems_options = {
          problem1: [option1, option2, ...],
          problem2: [option1, option2, ...],
        ...
      }
    """
    problems_options = {}

    # Iterate over all the options in the list
    for option in options_list:
        problem = option.problem  # Extract the problem associated with the option

        # If the problem doesn't exist in the dictionary, initialize an empty list
        if problem not in problems_options:
            problems_options[problem] = []

        # Append the option to the list for this problem
        problems_options[problem].append(option)

    return problems_options


def process_option(uniq_seq_dict, problem, seq, states, input_size, output_size_y1, hidden_size_custom_relu, learning_rate, l1_lambda, batch_size, num_epochs, multi_problem, print_loss):
    """
    This function contains the common logic for processing each sequence and problem.
    """
    # Initialize the Option object with the window size
    option = Option(problem, seq, input_size, output_size_y1, hidden_size_custom_relu, learning_rate, l1_lambda, batch_size, num_epochs)

    # Each option has different dataset
    observations = []

    if multi_problem:
        # Loop over each individual state in the tuple
        for state in states:    # here states is a tuple of states
            # Get the observations for the current state
            observations.append(state.get_observation())
    else:
        # Loop over each tuple of states in the list of state tuples
        for state_tuple in states:
            # Loop over each individual state in the tuple
            for state in state_tuple:
                # Get the observations for the current state
                observations.append(state.get_observation())
    
    if multi_problem:
        y1_labels, y2_labels = generate_labels(uniq_seq_dict, seq, problem, multi_problem, states)
    else:
        y1_labels, y2_labels = generate_labels(uniq_seq_dict, seq, problem, multi_problem)


    dataset_y1 = CustomDataset(observations, y1_labels)
    dataset_y2 = CustomDataset(observations, y2_labels)
    
    # Train the models
    option.train_y1(dataset_y1, print_loss=print_loss)
    option.train_y2(dataset_y2, print_loss=print_loss)

    return option


def extract_base_behaviors(problems_options):
    """
    This functions returns the model_y1 and model_y2 for the base behaviors (up, down, ledt, right).
    """
    base_behaviors = {"up": [], "down": [], "left": [], "right": []}
    for problem, options in problems_options.items():
        for option in options:
            if option.sequence == (0, 0, 1):
                base_behaviors["up"].append(option)
            elif option.sequence == (0, 1, 2):
                base_behaviors["down"].append(option)
            elif option.sequence == (2, 1, 0):
                base_behaviors["left"].append(option)
            elif option.sequence == (1, 0, 2):
                base_behaviors["right"].append(option)
    return base_behaviors


# Function to capture printed output from a function that prints
def capture_printed_output(func, *args, **kwargs):
    # Create a StringIO object to capture the output
    captured_output = io.StringIO()
    
    # Temporarily redirect sys.stdout to the StringIO object
    sys.stdout = captured_output
    
    try:
        # Call the function, which will print to the redirected stdout
        func(*args, **kwargs)
    finally:
        # Restore sys.stdout to its original state
        sys.stdout = sys.__stdout__
    
    # Get the printed content from StringIO and return it
    return captured_output.getvalue()


def log_weights(base_behaviors, hidden_size, game_width, l1_lambda, threshold, agent_loc, goal_loc, learning_rate):
    logging.basicConfig(
        filename=f'logs/base_behaviors_width_{game_width}_relu_{str(hidden_size)}_l1_{str(l1_lambda)}_lr_{str(learning_rate)}_thresh_{str(threshold)}_agentloc_{str(agent_loc)}_goalloc_{str(goal_loc)}_log.txt',  # Log file where the output will be saved
        filemode='w',  # 'w' for overwrite each time, 'a' for append
        level=logging.INFO,  # Log level
        format='%(message)s',  # Log format
    )

    logger = logging.getLogger()

    for behavior, options_for_behavior in base_behaviors.items():
        for option in options_for_behavior:
            logger.info(f"Behavior: {behavior} -- Sequence: {option.sequence} -- Problem: {option.problem}")
            
            option.truncate_all_weights(threshold=threshold)
            
            # Capture the printed model weights and log them
            try:
                # Ensure option.print_model_weights is valid and callable
                weights_log = capture_printed_output(option.print_model_weights, game_width, agent_loc=agent_loc, goal_loc=goal_loc)
                logger.info(weights_log)  # Log the captured output
            except Exception as e:
                logger.error(f"Error capturing weights: {e}")
        logger.info("################################################ END BEHAVIOR \n\n")


def log_evalute_behaviors_each_cell(problems_options, problems, game_width, hidden_size, l1_lambda, learning_rate):
    """
    In this function, we evaluate our base behaviors (4 sequences of actions) in each cell of the grid to see if they can perform as expected.
    """
    logging.basicConfig(
        filename=f'logs/each_cell_behavior_width_{game_width}_relu_{str(hidden_size)}_l1_{str(l1_lambda)}_lr_{str(learning_rate)}_log.txt',  # Log file where the output will be saved
        filemode='w',  # 'w' for overwrite each time, 'a' for append
        level=logging.INFO,  # Log level
        format='%(message)s',  # Log format
    )

    logger = logging.getLogger()
    base_behaviors = extract_base_behaviors(problems_options)
    
    for problem in problems:
        logger.info(f"################################################ OUTER PROBLEM: {problem} \n")
        env = Game(game_width, game_width, problem)

        for behavior, options_for_behavior in base_behaviors.items():
            for option in options_for_behavior:
                logger.info(f"Behavior: {behavior} -- Sequence: {option.sequence} -- Problem: {option.problem}")
                model_y1 = option.model_y1
                model_y2 = option.model_y2

                for i in range(game_width):
                    for j in range(game_width):
                        env._matrix_unit = np.zeros((game_width, game_width))
                        env._matrix_unit[i][j] = 1

                        logger.info(f"Cell: {i}, {j}")

                        for _ in range(3):  # 3 is the length of the sequence of actions
                            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                            prob_actions = model_y1(x_tensor)
                            stopping_probability = model_y2(x_tensor)
                            a = torch.argmax(prob_actions).item()
                            logger.info(f"{a} -- {stopping_probability}")
                            env.apply_action(a)
            logger.info("################################################ END BEHAVIOR \n\n")
        logger.info("################################################ END OUTER PROBLEM \n\n")