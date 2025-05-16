import torch
import numpy as np
from environment.combo import Game
from models.model import CustomRNN, CustomRelu 
from agents import PolicyGuidedAgent
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


    args = parser.parse_args()

    if args.base_model == "nn":

        policy_agent = PolicyGuidedAgent()

        if args.problem == "All":
            problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
        else:
            problems = [args.problem]

        for problem in problems:
            print("Problem: ", problem)

            hidden_size = args.hidden_size
            game_width = args.game_width
            lambda_l1 = args.l1
            learning_rate = args.lr

            rnn = CustomRelu(game_width**2 * 2 + 9, hidden_size, 3, lambda_l1=lambda_l1)

            shortest_trajectory_length = np.inf
            best_trajectory = None

            for _ in range(150):
                for _ in range(500):
                    env = Game(game_width, game_width, problem)
                    trajectory = policy_agent.run(env, rnn, length_cap=shortest_trajectory_length, verbose=False)

                    if len(trajectory.get_trajectory()) < shortest_trajectory_length:
                        shortest_trajectory_length = len(trajectory.get_trajectory())
                        best_trajectory = trajectory

                print('Trajectory length: ', len(best_trajectory.get_trajectory()))
                for _ in range(10):
                    loss = rnn.train(best_trajectory)
                    print(loss)
                print()

            policy_agent._epsilon = 0.0
            env = Game(game_width, game_width, problem)
            policy_agent.run(env, rnn, greedy=True, length_cap=None, verbose=True)
            rnn.print_weights()

            env = Game(game_width, game_width, problem)
            policy_agent.run_with_relu_state(env, rnn)

            torch.save(rnn.state_dict(), 'binary/NN-game-width' + str(game_width) + '-' + problem + '-relu-' + str(hidden_size) + '-l1-' + str(lambda_l1) + '-lr-' + str(learning_rate) + '-model.pth')

if __name__ == "__main__":
    main()