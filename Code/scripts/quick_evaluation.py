import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
import argparse
import gymnasium as gym

from environment.karel_env.gym_envs.karel_gym import KarelGymEnv, make_karel_env
from agents import PPOAgent, GruAgent

import re


def evaluate_model_on_large_grid(model_path, args):
    # Configuration for the large grid environment
    env_config = {
        'task_name': args.task_name,
        'env_height': args.game_width_eval,
        'env_width': args.game_width_eval,
        'max_steps': args.max_steps,
        'sparse_reward': args.sparse_reward,
        'seed': args.karel_seed,
        'initial_state': None
    }

    def make_env():
        return KarelGymEnv(env_config=env_config)

    envs = gym.vector.SyncVectorEnv([make_env])
    obs_shape = envs.single_observation_space.shape
    action_space = envs.single_action_space

    envs.reset()
    # envs.envs[0].render()
    # envs.envs[0].task.state2image(envs.envs[0].get_observation(), root_dir=project_root+'/environment/').show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.ppo_type == "original":
        agent = PPOAgent(envs, hidden_size=args.hidden_size).to(device)
    elif args.ppo_type == "gru":
        agent = GruAgent(envs, h_size=args.hidden_size, feature_extractor=args.feature_extractor, greedy=True).to(device)

    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    MAX_STEPS = 1000

    obs, _ = envs.reset(seed=0)
    done = False
    total_rewards = 0
    step = 0
    

    if args.ppo_type == "original":
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()

            obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.any(terminated) or np.any(truncated)
            total_rewards += reward[0]  # Since we have only one environment
            step += 1
            if step > MAX_STEPS:
                done = True

            # envs.envs[0].render()
            envs.envs[0].task.state2image(envs.envs[0].get_observation(), root_dir=project_root+'/environment/').show()

    elif args.ppo_type == "gru":
        # Initialize hidden state(s)
        num_envs = envs.num_envs  # Should be 1
        rnn_state = torch.zeros(agent.gru.num_layers, num_envs, agent.gru.hidden_size).to(device)
        done_tensor = torch.zeros(num_envs, dtype=torch.bool).to(device)
        step = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                action, _, _, _, rnn_state = agent.get_action_and_value(obs_tensor, rnn_state, done_tensor.float())
                action = action.cpu().numpy()

            obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.any(terminated) or np.any(truncated)
            total_rewards += reward[0]  # Since we have only one environment
            step += 1
            if step > MAX_STEPS:
                done = True

            # Update the done tensor for the next step
            done_tensor = torch.tensor(terminated | truncated, dtype=torch.bool).to(device)

            # envs.envs[0].render()
            # envs.envs[0].task.state2image(envs.envs[0].get_observation(), root_dir=project_root+'/environment/').show()

    envs.close()

    return total_rewards, step


"""
    This code, gets a directory of models and evaluates them on certain grid environments.
    It is for faster evaluation of many models out of a sweep.
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--binaries_path', default="binary_sweep_stair_climber", type=str, help="Path to the directory containing the models")
    parser.add_argument('--task_name', default="stair_climber", type=str, help="[stair_climber, maze]")
    parser.add_argument('--game_width_eval', default=100, type=int)
    parser.add_argument('--max_steps', default=100, type=int)
    parser.add_argument('--sparse_reward', action='store_true')
    parser.add_argument('--karel_seeds', nargs='+', type=int, help="For testing on multiple seeds")
    parser.add_argument('--feature_extractor', action='store_true')

    args = parser.parse_args()

    base_root = os.path.abspath(os.path.join(project_root, ".."))
    args.binaries_path = os.path.join(base_root, "binary", args.binaries_path)

    # Regex patterns for extracting values
    model_pattern = re.compile(
        r'gw(?P<game_width>\d+)-gh(?P<game_height>\d+)-h(?P<hidden_size>\d+)-lr(?P<learning_rate>[0-9e.-]+)-'
        r'sd(?P<model_seed>\d+)-entcoef(?P<ent_coef>[0-9e.-]+)-clipcoef(?P<clip_coef>[0-9e.-]+)_vlr(?P<value_learning_rate>[0-9e.-]+)'
    )

    for model in os.listdir(args.binaries_path):
        match = model_pattern.search(model)
        if not match:
            raise ValueError(f"Invalid model filename: {model}")

        args.game_width = int(match.group('game_width'))
        args.game_height = int(match.group('game_height'))
        args.hidden_size = int(match.group('hidden_size'))
        args.learning_rate = float(match.group('learning_rate'))
        args.model_seed = int(match.group('model_seed'))
        args.ent_coef = float(match.group('ent_coef'))
        args.clip_coef = float(match.group('clip_coef'))
        args.value_learning_rate = float(match.group('value_learning_rate'))
        args.ppo_type = model.split("_")[-3]
        args.time = int(model.split("_")[-1].split(".")[0])

        # print(f"Extracted Parameters: "
        #       f"Game Width: {args.game_width}, Game Height: {args.game_height}, "
        #       f"Hidden Size: {args.hidden_size}, Learning Rate: {args.learning_rate}, "
        #       f"Model Seed: {args.model_seed}, Entropy Coef: {args.ent_coef}, "
        #       f"Clip Coef: {args.clip_coef}, Value LR: {args.value_learning_rate} \n")


        model_file_name = f'{args.binaries_path}/PPO-Karel_{args.task_name}-gw{args.game_width}-gh{args.game_height}-h{args.hidden_size}-lr{args.learning_rate}-sd{args.model_seed}-entcoef{args.ent_coef}-clipcoef{args.clip_coef}_vlr{args.value_learning_rate}_{args.ppo_type}_MODEL_{args.time}.pt'
        print(f"--- Model: {model}")

        for ks in args.karel_seeds:
            args.karel_seed = ks
            total_rewards, step = evaluate_model_on_large_grid(model_file_name, args)
            print(f"--- Seed: {ks}, Total Rewards: {total_rewards}, Steps: {step} \n")
