import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from levin_loss import LevinLossMLP
from utils import (group_options_by_problem, 
                   load_trajectories, 
                   extract_base_behaviors, 
                   log_weights, 
                   log_evalute_behaviors_each_cell,
                   load_trajectories_ppo,
                   extract_any_behaviors,)
import copy
import pickle
import argparse


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


def evaluate_all_options_levin_loss(problems_options, trajectories, args):
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
    if args.save_options:
        print("Saving options list")
        save_path = 'binary/selected_options_relu_' + str(args.hidden_size) + '_gw_' + str(args.game_width) + '_num_epochs_' + str(args.num_epoch) + '_l1_' + str(args.l1) + '_lr_' + str(args.lr) + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(selected_options, f)
        print(f'Options list saved to {save_path}')
    

def parse_tuples(lst):
    try:
        return [tuple(int(item) for item in s.strip().strip('()').split(',')) for s in lst]
    except ValueError:
        raise argparse.ArgumentTypeError("Each tuple must contain integers separated by commas.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_model', default="nn", type=str)
    parser.add_argument('--game_width', default=3, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--l1', default=0.001, type=float)
    parser.add_argument('--problem', default="All", type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--log_path', default="logs/", type=str)
    parser.add_argument('--num_epoch', default=5000, type=int)
    parser.add_argument('--l1_base', default=0.001, type=float)
    parser.add_argument('--print_loss', default=False, type=bool)
    parser.add_argument('--eval_method', nargs='+', default=["levinloss"], type=str, help="['levinloss', 'each_cell', 'weights', 'weights_any'] (default: levinloss)")
    parser.add_argument('--weight_thresh', default=0.00001, type=float)
    parser.add_argument('--agent_loc', default=True, type=bool)
    parser.add_argument('--goal_loc', default=True, type=bool)
    parser.add_argument('--total_timesteps', default=50000, type=int)
    parser.add_argument('--ent_coef', default=0.0, type=float)
    parser.add_argument('--clip_coef', default=0.01, type=float)
    parser.add_argument('--len3', default=False, type=bool)
    parser.add_argument('--other_behaviors', nargs='+', default=[], type=str)
    parser.add_argument('--save_options', action='store_true', help="Save the options extracted from levin loss selection")

    args = parser.parse_args()


    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]

    save_path = 'binary/' + args.base_model + '_options_list_relu_' + str(args.hidden_size) + '_game_width_' + str(args.game_width) + '_num_epochs_' + str(args.num_epoch) + '_l1_' + str(args.l1) + '_lr_' + str(args.lr) + '.pkl'
    if args.len3:   save_path = save_path.replace(".pkl", "_onlyws3.pkl")
    with open(save_path, 'rb') as f:
        options_list = pickle.load(f)
    print(f'Options list loaded from {save_path}')


    if args.base_model == "nn":
        trajectories = load_trajectories(problems, args)
    elif args.base_model == "ppo":
        trajectories = load_trajectories_ppo(problems, args)

    problems_options = group_options_by_problem(options_list)

    for problem, trajectory in trajectories.items():
        print("Problem:", problem)
        print("actions: ", trajectory.get_action_sequence(), " \n")

    print(f"Eval methods: {args.eval_method}")
    if "levinloss" in args.eval_method:
        """
        1. Levin Loss evaluation
        """
        print("Running Levin Loss evaluation")
        evaluate_all_options_levin_loss(problems_options, trajectories, args)

    if "each_cell" in args.eval_method:
        """
        2. Evaluating base options in each cell
        """
        print("Running evaluation of base options in each cell")
        log_evalute_behaviors_each_cell(problems_options, problems, args)

    if "weights" in args.eval_method:
        """
        3. Analyzing the weights of the base options to see the effect of l1 regularization
        """
        print("Analyzing weights of base options")
        base_behaviors = extract_base_behaviors(problems_options)
        log_weights(base_behaviors, args)

    if "weights_any" in args.eval_method:
        print("Analyzing weights of desired options", args.other_behaviors)
        args.other_behaviors = parse_tuples(args.other_behaviors)
        behaviors = extract_any_behaviors(problems_options, args.other_behaviors)
        log_weights(behaviors, args, is_base=False)


if __name__ == "__main__":
    main()