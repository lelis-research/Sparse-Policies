from levin_loss import LevinLossMLP
from utils import setup_environment, run_environment, group_options_by_problem, load_trajectories
import copy
import pickle


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



def main():

    # Load options_list from the file
    save_path = 'binary/options_list.pkl'
    with open(save_path, 'rb') as f:
        options_list = pickle.load(f)
    print(f'Options list loaded from {save_path}')


    game_width = 3
    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
    hidden_size_custom_relu = 6


    trajectories = load_trajectories(problems, hidden_size_custom_relu, game_width)
    problems_options = group_options_by_problem(options_list)

    
    evaluate_all_options_levin_loss(problems_options, trajectories)



if __name__ == "__main__":
    main()