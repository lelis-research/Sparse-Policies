import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
import argparse
import gymnasium as gym
import re
from collections import defaultdict
from tqdm import tqdm

from environment.karel_env.gym_envs.karel_gym import KarelGymEnv, make_karel_env
from agents import PPOAgent, GruAgent



def evaluate_model_on_large_grid(model_path, args):
    # Configuration for the large grid environment
    env_config = {
        'task_name': args.task_name,
        'env_height': args.game_width_eval,
        'env_width': args.game_width_eval,
        'max_steps': args.max_steps,
        'sparse_reward': args.sparse_reward,
        'seed': args.karel_seed,
        'initial_state': None,
        'reward_scale': False,
    }

    def make_env():
        return KarelGymEnv(env_config=env_config)

    envs = gym.vector.SyncVectorEnv([make_env])
    envs.reset()
    # envs.envs[0].render()
    # envs.envs[0].task.state2image(envs.envs[0].get_observation(), root_dir=project_root+'/environment/').show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.ppo_type == "original":
        agent = PPOAgent(envs, hidden_size=args.hidden_size, feature_extractor=args.feature_extractor, greedy=True).to(device)
    elif args.ppo_type == "gru":
        agent = GruAgent(envs, h_size=args.hidden_size, feature_extractor=args.feature_extractor, greedy=True).to(device)

    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    MAX_STEPS = 100000

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
            # envs.envs[0].task.state2image(envs.envs[0].get_observation(), root_dir=project_root+'/environment/').show()

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

    parser.add_argument('--binaries_path', default="binary_test_stair", type=str, help="Path to the directory containing the models")
    parser.add_argument('--task_name', default="stair_climber", type=str, help="[stair_climber, maze, ...]")
    parser.add_argument('--game_width_eval', default=100, type=int)
    parser.add_argument('--max_steps', default=100, type=int)
    parser.add_argument('--sparse_reward', action='store_true')
    parser.add_argument('--karel_seeds', nargs='+', type=int, default=list(range(0,100)), help="For testing on multiple seeds")
    parser.add_argument('--feature_extractor', action='store_true')

    args = parser.parse_args()

    base_root = os.path.abspath(os.path.join(project_root, ".."))
    args.binaries_path = os.path.join(base_root, "binary", args.binaries_path)

    model_pattern = re.compile(
    r'gw(?P<game_width>\d+)-'
    r'gh(?P<game_height>\d+)-'
    r'h(?P<hidden_size>\d+)-'
    r'lr(?P<learning_rate>[0-9eE\.\-]+)-'
    r'sd(?P<model_seed>\d+)-'
    r'entcoef(?P<ent_coef>[0-9.]+)-'
    r'clipcoef(?P<clip_coef>[0-9.]+)-'
    r'l1(?P<l1_lambda>[0-9.]+)-'
    r'(?P<ppo_type>\w+)-MODEL'
)

    # Structure: {group_key: {params: dict, seeds: {model_seed: {karel_seed: (reward, steps)}}}}
    groups = defaultdict(lambda: {
        'params': None,
        'seeds': defaultdict(dict),
        'avg_reward': 0,
        'first_model': None # to save a model name for easier access to it for further testing
    })

    # Process all models
    model_files = [f for f in os.listdir(args.binaries_path) if f.endswith(".pt")]
    for model_file in tqdm(model_files, desc="Evaluating models"):
        if not model_file.endswith(".pt"):
            continue

        tqdm.write(f"\nProcessing: {model_file}")

        match = model_pattern.search(model_file)
        if not match:
            tqdm.write(f"**** Skipping unmatched file: {model_file}")
            continue

        params = {
            'game_width': int(match.group('game_width')),
            'game_height': int(match.group('game_height')),
            'hidden_size': int(match.group('hidden_size')),
            'learning_rate': float(match.group('learning_rate')),
            'ent_coef': float(match.group('ent_coef')),
            'clip_coef': float(match.group('clip_coef')),
            'l1_lambda': float(match.group('l1_lambda')),
            'ppo_type': match.group('ppo_type'),
            'model_seed': int(match.group('model_seed'))
        }
        args.ppo_type = params['ppo_type']
        args.hidden_size = params['hidden_size']
        
        # Create group key (excluding model seed)
        group_key = (params['game_width'], params['game_height'], params['hidden_size'],
                    params['learning_rate'], params['ent_coef'], 
                    params['clip_coef'], params['l1_lambda'], params['ppo_type'])

        # Store parameters if new group
        if not groups[group_key]['params']:
            groups[group_key]['params'] = {
                'lr': params['learning_rate'],
                'l1': params['l1_lambda'],
                'ent_coef': params['ent_coef'],
                'clip_coef': params['clip_coef'],
                'hidden_size': params['hidden_size'],
                'game_size': f"{params['game_width']}x{params['game_height']}",
                'ppo_type': params['ppo_type']
            }
            groups[group_key]['first_model'] = model_file

        # Evaluate model on all environment seeds
        model_path = os.path.join(args.binaries_path, model_file)
        seed_results = {}
        for karel_seed in args.karel_seeds:
            args.karel_seed = karel_seed
            reward, steps = evaluate_model_on_large_grid(model_path, args)
            seed_results[karel_seed] = (float(reward), int(steps))
        
        groups[group_key]['seeds'][params['model_seed']] = seed_results

    # Calculate averages and sort groups
    sorted_groups = []
    for group_key, group_data in groups.items():
        all_rewards = []
        for model_seed, seeds in group_data['seeds'].items():
            for karel_seed, (reward, _) in seeds.items():
                all_rewards.append(reward)
        group_data['avg_reward'] = sum(all_rewards)/len(all_rewards) if all_rewards else 0
        group_data['std_reward'] = np.std(all_rewards) if all_rewards else 0
        sorted_groups.append((group_data['avg_reward'], group_data))

    # Sort groups by average reward descending
    sorted_groups.sort(reverse=True, key=lambda x: x[0])

    # Print formatted results
    eval_name = args.binaries_path.split('/')[-2]   # [-1] is "binary"
    output_filename = f"{project_root}/Scripts/evaluation/karel/paper/eval{args.game_width_eval}_{eval_name}.txt"
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with open(output_filename, 'w') as f:
        f.write("\n=== Evaluation Results Grouped by Hyperparameters ===\n")
        f.write(f"Task: {args.task_name}, Game size: {args.game_width_eval}x{args.game_width_eval}, Max steps: {args.max_steps}, Karel_seeds: {args.karel_seeds}\n")
        for avg_reward, group in sorted_groups:
            params = group['params']
            f.write(f"\nHyperParams: "
                    f"H: {params['hidden_size']}, Lr: {params['lr']:.0e}, L1: {params['l1']}, "
                    f"Ent_coef: {params['ent_coef']}, Clip_coef: {params['clip_coef']}\n")
            f.write(f"Avg reward (over {len(group['seeds'])*len(args.karel_seeds)} runs): {avg_reward:.2f}, std: {group['std_reward']:.2f}\n")
            f.write("Results per training seed:\n")
            for model_seed, seeds in sorted(group['seeds'].items()):
                rewards = [f"{seeds[ks][0]:.2f}" for ks in args.karel_seeds]
                f.write(f"  sd{model_seed}: {' '.join(rewards)}\n")
            
            f.write(f"First model: {group['first_model']}\n")
            f.write("------------------------------------------------\n")

    print(f"\n\nEvaluation complete. Results saved to {output_filename}")

    # Save results to pickle file for further analysis
    import pickle
    output_filename = output_filename.replace(".txt", ".pkl")
    with open(output_filename, 'wb') as f:
        pickle.dump(sorted_groups, f, protocol=pickle.HIGHEST_PROTOCOL)
