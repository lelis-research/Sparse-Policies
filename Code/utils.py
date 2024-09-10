import numpy as np
import torch
from combo import Game

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