import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from levin_loss import LevinLossMLP
from utils import group_options_by_problem, load_trajectories, extract_base_behaviors, log_weights, log_evalute_behaviors_each_cell
import copy
import pickle


def evaluate_all_options_for_problem(selected_options, problem, trajectories, number_actions, number_iterations, options_for_this_model):
    """
    Function that evaluates all options for a given problem. It returns the best option (the one that minimizes the Levin loss)
    for the current set of selected options. It also returns the Levin loss of the best option. 
    """
    best_option = None
    best_value = None
    loss = LevinLossMLP()

    for current_option in options_for_this_model:
        
        value = loss.compute_loss_y1_y2(selected_options + [current_option], problem, trajectories, number_actions, number_iterations)
        print("-- Value: ", value, "for opt: ", current_option.sequence)

        if best_option is None or value < best_value:
            best_value = value
            best_option = copy.deepcopy(current_option)
            print("- Best Value: ", best_value)
                            
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
    # selected_options_problem = []

    while previous_loss is None or best_loss < previous_loss:
        print("************ NEW ITERATION ************")
        previous_loss = best_loss

        best_loss = None
        best_option = None
        # problem_option = None

        for problem, options_of_problem in problems_options.items():
            print("evaluating option for problem: ", problem)

            option, levin_loss = evaluate_all_options_for_problem(selected_options, problem, trajectories, number_actions, number_iterations, options_of_problem)
            print("Option: ", option.sequence, " - Loss: ", levin_loss)

            if best_loss is None or levin_loss < best_loss:
                best_loss = levin_loss
                best_option = option
                # problem_option = problem

                print('Best Loss so far: ', best_loss, problem)
            print("################################################ END PROBLEM")

        """
        we recompute the Levin loss after the automaton is selected so that we can use 
        the loss on all trajectories as the stopping condition for selecting automata 
        """
        print("option ", best_option.sequence, " selected")
        # if best_option.sequence in [(0, 0, 1), (0, 1, 2), (2, 1, 0), (1, 0, 2)]:
        selected_options.append(best_option)
        # selected_options_problem.append(problem_option)
        best_loss = loss.compute_loss_y1_y2(selected_options, "", trajectories, number_actions, number_iterations)

        print("selected options so far:")
        for opt in selected_options:
            print(opt.sequence)
        print("Levin loss of the current set: ", best_loss)
        print("################################################ END OPTION\n\n")

    # remove the last option added because it increased the loss
    selected_options = selected_options[:-1]
    print("final selected options:")
    for opt in selected_options:
        print(opt.sequence)

    for t in trajectories.items():
        print("Trajectory: ", t[0])
        print("Trajectory: ", t[1].get_action_sequence())
        print("################################################ END TRAJECTORY\n")
    loss = LevinLossMLP()
    loss.print_output_subpolicy_trajectory_y1y2(selected_options, trajectories, number_iterations)

    for i in range(len(selected_options)):
        print("Option", i, ":", selected_options[i].sequence, "- from Problem: ", selected_options[i].problem)

    # Save the options list to a file
    save_path = 'binary/final_options.pkl'
    # with open(save_path, 'wb') as f:
    #     pickle.dump(selected_options, f)
    # print(f'Options list saved to {save_path}')
        

def main():

    num_epochs = 5000
    game_width = 3
    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
    hidden_size_custom_relu = 32
    l1_lambda = 0.0005
    l1_base = 0.005
    learning_rate = 0.1


    # Load options_list from the file
    save_path = 'binary/options_list_hidden_size_' + str(hidden_size_custom_relu) + '_game_width_' + str(game_width) + '_num_epochs_' + str(num_epochs) + '_l1_' + str(l1_lambda) + '_lr_' + str(learning_rate) + '_onlyws3.pkl'
    with open(save_path, 'rb') as f:
        options_list = pickle.load(f)
    print(f'Options list loaded from {save_path}')


    trajectories = load_trajectories(problems, hidden_size_custom_relu, game_width, l1_base)
    problems_options = group_options_by_problem(options_list)

    """
    1. Levin Loss evaluation
    """
    evaluate_all_options_levin_loss(problems_options, trajectories)

    """
    2. Evaluating base options in each cell
    """
    # log_evalute_behaviors_each_cell(problems_options, problems, game_width, hidden_size_custom_relu, l1_lambda, learning_rate)

    """
    3. Analyzing the weights of the base options to see the effect of l1 regularization
    """
    base_behaviors = extract_base_behaviors(problems_options)

    threshold = 0.00001
    agent_loc = True
    goal_loc = True
    # log_weights(base_behaviors, hidden_size_custom_relu, game_width, l1_lambda, threshold, agent_loc, goal_loc, learning_rate)

    # for problem, trajectory in trajectories.items():
    #     print("Problem:", problem)
    #     print("actions: ", trajectory.get_action_sequence(), " \n")



if __name__ == "__main__":
    main()