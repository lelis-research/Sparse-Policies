import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from levin_loss import LevinLossMLP
from utils import setup_environment, run_environment, group_options_by_problem, load_trajectories
import copy
import pickle
from combo import Game
import numpy as np
import torch


def evaluate_all_options_for_problem(selected_options, problem, trajectories, number_actions, number_iterations, options_for_this_model):
    """
    Function that evaluates all options for a given model. It returns the best option (the one that minimizes the Levin loss)
    for the current set of selected options. It also returns the Levin loss of the best option. 
    """
    best_option = None
    best_value = None
    loss = LevinLossMLP()

    for current_option in options_for_this_model:
        
        value = loss.compute_loss_y1_y2(selected_options + [current_option], problem, trajectories, number_actions, number_iterations)
        print("Value: ", value)

        if best_option is None or value < best_value:
            best_value = value
            best_option = copy.deepcopy(current_option)
            print("Best Value: ", best_value)
                            
    return best_option, best_value


def evaluate_all_options_levin_loss(problems_options, trajectories):
    """
    This function implements the greedy approach for selecting options.
    This method evaluates all different options of a given model and adds to the pool of options the one that minimizes
    the Levin loss. This process is repeated while we can minimize the Levin loss.  
    """
    number_iterations = 3
    number_actions = 3

    previous_loss = None
    best_loss = None

    loss = LevinLossMLP()

    selected_options = []
    selected_options_problem = []

    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_option = None
        problem_option = None

        for problem, options_of_problem in problems_options.items():

            option, levin_loss = evaluate_all_options_for_problem(selected_options, problem, trajectories, number_actions, number_iterations, options_of_problem)

            if best_loss is None or levin_loss < best_loss:
                best_loss = levin_loss
                best_option = option
                problem_option = problem

                print('Best Loss so far: ', best_loss, problem)
            print("################################################ END PROBLEM")

        """
        we recompute the Levin loss after the automaton is selected so that we can use 
        the loss on all trajectories as the stopping condition for selecting automata 
        """
        selected_options.append(best_option)
        selected_options_problem.append(problem_option)
        best_loss = loss.compute_loss_y1_y2(selected_options, "", trajectories, number_actions, number_iterations)

        print("Levin loss of the current set: ", best_loss)
        print("################################################ END OPTION\n\n")


    loss = LevinLossMLP()
    loss.print_output_subpolicy_trajectory_y1y2(selected_options, selected_options_problem, trajectories, number_iterations)

    for i in range(len(selected_options)):
        print("Option", i, ":", selected_options[i].sequence)

    # Save the options list to a file
    save_path = 'binary/final_options.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(selected_options, f)
    print(f'Options list saved to {save_path}')


def extract_base_behaviors(problems_options):
    """
    This functions returns the model_y1 and model_y2 for the base behaviors (up, down, ledt, right).
    """
    base_behaviors = {}
    for problem, options in problems_options.items():
        for option in options:
            if option.sequence == (0, 0, 1):
                base_behaviors["up"] = option
            elif option.sequence == (0, 1, 2):
                base_behaviors["down"] = option
            elif option.sequence == (2, 1, 0):
                base_behaviors["left"] = option
            elif option.sequence == (1, 0, 2):
                base_behaviors["right"] = option
    return base_behaviors


def evalute_behaviors_each_cell(problems_options, problem, game_width):
    """
    In this function, we evaluate our base behaviors (4 sequences of actions) in each cell of the grid to see if they can perform as expected.
    """
    base_behaviors = extract_base_behaviors(problems_options)
    env = Game(game_width, game_width, problem)

    for behavior, option in base_behaviors.items():
        print("Behavior: ", behavior, " -- Sequence: ", option.sequence)
        model_y1 = option.model_y1
        model_y2 = option.model_y2

        for i in range(game_width):
            for j in range(game_width):
                env._matrix_unit = np.zeros((game_width, game_width))
                env._matrix_unit[i][j] = 1

                print("Cell: ", i, j)

                for _ in range(3):  # 3 is the length of the sequence of actions
                    x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                    prob_actions = model_y1(x_tensor)
                    stopping_probability = model_y2(x_tensor)
                    a = torch.argmax(prob_actions).item()
                    print(a, stopping_probability)
                    env.apply_action(a)
        print("################################################ END BEHAVIOR \n\n")


def main():

    num_epochs = 5000
    game_width = 3
    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
    hidden_size_custom_relu = 32
    l1_lambda = 0.005

    # Load options_list from the file
    save_path = 'binary/options_list_hidden_size_' + str(hidden_size_custom_relu) + '_game_width_' + str(game_width) + '_num_epochs_' + str(num_epochs) + '-lr-' + str(l1_lambda) + '_onlyws3.pkl'
    with open(save_path, 'rb') as f:
        options_list = pickle.load(f)
    print(f'Options list loaded from {save_path}')


    trajectories = load_trajectories(problems, hidden_size_custom_relu, game_width, l1_lambda)
    problems_options = group_options_by_problem(options_list)

    """
    1. Levin Loss evaluation
    """
    evaluate_all_options_levin_loss(problems_options, trajectories)

    """
    2. Evaluating base options in each cell
    """
    # evalute_behaviors_each_cell(problems_options, problem="BL-TR", game_width=game_width)

    """
    3. Analyzing the weights of the base options to see the effect of l1 regularization
    """
    # base_behaviors = extract_base_behaviors(problems_options)
    # for behavior, option in base_behaviors.items():
    #     print("Behavior: ", behavior, " -- Sequence: ", option.sequence)
    #     option.print_model_weights()
    #     print("################################################ END BEHAVIOR \n\n")



if __name__ == "__main__":
    main()