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
from environment.cartpole_gym import LastActionObservationWrapper
from agents import PPOAgent, GruAgent
from tqdm import tqdm


def evaluate_model(model_path, ppo_type, hidden_size, eval_seed, max_steps, args):

    def make_env():
        return LastActionObservationWrapper(
            gym.make("CartPole-v1", 
                max_episode_steps=max_steps,
                render_mode="rgb_array"), 
            train_mode=(not args.test_mode), 
            last_action_in_obs=False
        )

    envs = gym.vector.SyncVectorEnv([make_env])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ppo_type == "original":
        agent = PPOAgent(envs, 
                        hidden_size=hidden_size,
                        feature_extractor=args.feature_extractor,
                        greedy=True).to(device)
    elif ppo_type == "gru":
        agent = GruAgent(envs,
                        h_size=hidden_size,
                        feature_extractor=args.feature_extractor,
                        greedy=True).to(device)
    else:
        raise ValueError(f"Invalid ppo_type: {ppo_type}")

    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset(seed=eval_seed)
    done = False
    episode_reward = 0
    step = 0

    if ppo_type == "original":
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()

            obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.any(terminated) or np.any(truncated)
            episode_reward += reward[0]
            step += 1
            if step > max_steps:
                break

    elif ppo_type == "gru":
        num_envs = envs.num_envs
        rnn_state = torch.zeros(agent.gru.num_layers, num_envs, agent.gru.hidden_size).to(device)
        done_tensor = torch.zeros(num_envs, dtype=torch.bool).to(device)

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action, _, _, _, rnn_state = agent.get_action_and_value(obs_tensor, rnn_state, done_tensor.float())
                action = action.cpu().numpy()

            obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.any(terminated) or np.any(truncated)
            episode_reward += reward[0]
            step += 1
            if step > max_steps:
                break

            done_tensor = torch.tensor(terminated | truncated, dtype=torch.bool).to(device)

    envs.close()
    return episode_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--binaries_path', type=str, required=True,
                       help="Directory containing trained models")
    parser.add_argument('--eval_seeds', nargs='+', type=int, required=True,
                       help="Seeds for evaluation runs")
    parser.add_argument('--max_steps', type=int, default=0,
                       help="Max steps per evaluation episode")
    parser.add_argument('--test_mode', action='store_true',
                        help="Determing the test/train distribution for evaluation")
    parser.add_argument('--feature_extractor', action='store_true')

    args = parser.parse_args()

    if args.max_steps == 0:
        args.max_steps = 15000 if args.test_mode else 250 # 250 for training (5 seconds), 15000 for testing (5 minutes)

    base_root = os.path.abspath(os.path.join(project_root, ".."))
    args.binaries_path = os.path.join(base_root, "binary", args.binaries_path)

    model_pattern = re.compile(
        r'PPO-Cartpole-gw\d+-gh\d+-'
        r'h(?P<hidden_size>\d+)-'
        r'lr(?P<lr>[0-9eE\.\-]+)-'  # Allow scientific notation (e.g., 1e-05)
        r'sd(?P<model_seed>\d+)-'
        r'entcoef(?P<ent_coef>[0-9.]+)-'
        r'clipcoef(?P<clip_coef>[0-9.]+)-'
        r'l1(?P<l1_lambda>[0-9.]+)-'
        r'(?P<ppo_type>\w+)-MODEL.*\.pt$'  # Match full filename including timestamp
    )

    groups = defaultdict(lambda: {
        'params': None,
        'seeds': defaultdict(dict),
        'avg_reward': 0,
        'first_model': None # to save a model name for easier access to it for further testing
    })

    model_files = [f for f in os.listdir(args.binaries_path) if f.endswith(".pt")]
    for model_file in tqdm(model_files, desc="Evaluating models"):
        if not model_file.endswith(".pt"):
            continue

        tqdm.write(f"\nProcessing: {model_file}")

        match = model_pattern.match(model_file)
        if not match:
            tqdm.write(f"**** Skipping unmatched file: {model_file}")
            continue

        params = {
            'hidden_size': int(match.group('hidden_size')),
            'lr': float(match.group('lr')),
            'ent_coef': float(match.group('ent_coef')),
            'clip_coef': float(match.group('clip_coef')),
            'l1_lambda': float(match.group('l1_lambda')),
            'ppo_type': match.group('ppo_type'),
            'model_seed': int(match.group('model_seed'))
        }

        group_key = (
            params['hidden_size'],
            params['lr'],
            params['ent_coef'],
            params['clip_coef'],
            params['l1_lambda'],
            params['ppo_type']
        )

        if not groups[group_key]['params']:
            groups[group_key]['params'] = {
                'hidden_size': params['hidden_size'],
                'lr': params['lr'],
                'ent_coef': params['ent_coef'],
                'clip_coef': params['clip_coef'],
                'l1_lambda': params['l1_lambda'],
                'ppo_type': params['ppo_type']
            }
            groups[group_key]['first_model'] = model_file

        model_path = os.path.join(args.binaries_path, model_file)
        seed_results = {}
        for eval_seed in args.eval_seeds:
            reward = evaluate_model(
                model_path=model_path,
                ppo_type=params['ppo_type'],
                hidden_size=params['hidden_size'],
                eval_seed=eval_seed,
                max_steps=args.max_steps,
                args=args
            )
            seed_results[eval_seed] = reward

        groups[group_key]['seeds'][params['model_seed']] = seed_results

    # Calculate averages and sort groups
    sorted_groups = []
    for group_key, group_data in groups.items():
        all_rewards = []
        for model_seed, seeds in group_data['seeds'].items():
            all_rewards.extend(seeds.values())
        group_data['avg_reward'] = np.mean(all_rewards) if all_rewards else 0
        sorted_groups.append((group_data['avg_reward'], group_data))

    sorted_groups.sort(reverse=True, key=lambda x: x[0])


    eval_name = args.binaries_path.split('/')[-2]   # [-1] is "binary"
    if args.test_mode:
        output_filename = f"{project_root}/Scripts/evaluation/cartpole/eval_{eval_name}_testMode.txt"
    else:
        output_filename = f"{project_root}/Scripts/evaluation/cartpole/eval_{eval_name}_trainMode.txt"
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # Write results to file
    with open(output_filename, 'w') as f:
        f.write("=== CartPole Evaluation Results ===\n")
        f.write(f"Evaluation seeds: {args.eval_seeds}\n")

        if args.test_mode:
            f.write(f"Evaluating on Test distribution: pole len: 1.0, max_steps: {args.max_steps}\n\n")
        else:
            f.write(f"Evaluating on Training distribution: pole len: 0.5, max_steps: {args.max_steps}\n\n")

        
        for avg_reward, group in sorted_groups:
            params = group['params']
            f.write(f"Hyperparameters:\n")
            f.write(f"PPO Type: {params['ppo_type']}")
            f.write(f"  H: {params['hidden_size']}")
            f.write(f"  LR: {params['lr']:.0e}")
            f.write(f"  Entropy_coef: {params['ent_coef']:.2f}")
            f.write(f"  Clip_coef: {params['clip_coef']:.2f}")
            f.write(f"  L1: {params['l1_lambda']}\n")
            f.write(f"Average Reward (over {len(group['seeds'])*len(args.eval_seeds)} runs): {avg_reward:.2f}\n")

            f.write("Results per training seed:\n")
            for model_seed, seeds in sorted(group['seeds'].items()):
                rewards = [f"{seeds[es]:.1f}" for es in args.eval_seeds]
                f.write(f"  sd {model_seed}: {' '.join(rewards)}\n")

            f.write(f"First model: {group['first_model']}\n")
            f.write("------------------------------------------------\n")

    print(f"\n\nEvaluation complete. Results saved to {output_filename}")