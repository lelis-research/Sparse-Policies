import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
from utils import (load_trajectories, 
                   update_uniq_seq_dict, 
                   process_option,
                   load_trajectories_ppo,)
import argparse


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
    parser.add_argument('--total_timesteps', default=50000, type=int)
    parser.add_argument('--ent_coef', default=0.0, type=float)
    parser.add_argument('--clip_coef', default=0.01, type=float)

    args = parser.parse_args()


    problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]

    input_size = args.game_width**2 * 2 + 9 # (3*3) for agent position + (3*3) for goal position + (3*3) for 3 actions of size 3
    output_size_y1 = 3  # 3 possible actions for y1
    batch_size = 1
    multi_problem = True

    # Create an array of Options objects with different window sizes
    options_list = []
    uniq_seq_dict = {}

    # Keeping track of sequences of actions and the models that generate them
    if args.base_model == "nn":
        trajectories = load_trajectories(problems, args)
    elif args.base_model == "ppo":
        trajectories = load_trajectories_ppo(problems, args)

    for problem, trajectory in trajectories.items():
        print("Problem:", problem)
        print("actions: ", trajectory.get_action_sequence(), " \n")

        window_sizes = list(range(2, len(trajectory.get_trajectory())))
        # window_sizes = [3]

        # Loop through different window sizes (from 2 to the length of the trajectory)
        for ws in window_sizes:
            uniq_seq_dict = update_uniq_seq_dict(trajectory, problem, ws, seq_dict=uniq_seq_dict, multi_problem=multi_problem)
        

    if multi_problem:
        for seq, problem_state_list in uniq_seq_dict.items():
            for problem, states_tuple in problem_state_list:
                print(f"Sequence: {seq}, Problem: {problem}, States: {states_tuple}")
                option = process_option(uniq_seq_dict, problem, seq, states_tuple, input_size, output_size_y1, args.hidden_size, args.lr, args.l1, batch_size, args.num_epoch, multi_problem, args.print_loss)
                options_list.append(option)


    # Save the options list to a file
    save_path = 'binary/' + str(args.base_model.upper()) + '_options_list_relu_' + str(args.hidden_size) + '_game_width_' + str(args.game_width) + '_num_epochs_' + str(args.num_epoch) + '_l1_' + str(args.l1) + '_lr_' + str(args.lr) + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(options_list, f)
    print(f'Options list saved to {save_path}')


if __name__ == "__main__":
    main()