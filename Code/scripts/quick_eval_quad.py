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
from environment.quad.quad_gym import QuadEnv
from agents import PPOAgent, GruAgent
from tqdm import tqdm


def evaluate_model(model_path, ppo_type, hidden_size, eval_seed, max_steps, args, env_type, test_mode):
    
    torch.manual_seed(eval_seed)
    
    use_po = (env_type == 'QuadPO')
    def make_env():
        return QuadEnv(n_steps=max_steps, 
                       use_po=use_po, 
                       test_mode=test_mode, 
                       render_mode=None)
    
    envs = gym.vector.SyncVectorEnv([make_env])

    # For Quad 2d (original)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    
    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    goal_reached = False
    is_safe = True
    
    if ppo_type == "original":
        while not done:

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()

            obs, reward, terminated, truncated, _ = envs.step(action)
            is_safe = envs.envs[0].last_state_safety < 0.05
            goal_reached = envs.envs[0].is_last_state_goal
            done = (not is_safe) or np.any(truncated) or goal_reached

            episode_reward += reward[0]
            step += 1
            if step >= max_steps:
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
            
            obs, reward, terminated, truncated, _ = envs.step(action)
            done = np.any(terminated) or np.any(truncated)
            episode_reward += reward[0]
            step += 1
            if step >= max_steps:
                break
            
            done_tensor = torch.tensor(terminated | truncated, dtype=torch.bool).to(device)
    
    envs.close()
    return episode_reward, goal_reached


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--binaries_path', type=str, required=False, default="test_quad/binary",
                       help="Directory containing trained quad models")
    parser.add_argument('--eval_seeds', nargs='+', type=int, default=range(0, 100),
                       help="Seeds for evaluation runs")
    parser.add_argument('--max_steps', type=int, default=5000,
                       help="Max steps per evaluation episode")
    parser.add_argument('--test_mode', action='store_true',
                       help="Use test mode for the environment")
    parser.add_argument('--feature_extractor', action='store_true',
                       help="The agent has feature extractor or not")
    args = parser.parse_args()

    base_root = os.path.abspath(os.path.join(project_root, ".."))
    args.binaries_path = os.path.join(base_root, "binary", args.binaries_path)

    model_pattern = re.compile(
        r'PPO-(Quad|QuadPO)-'
        r'h(?P<hidden_size>\d+)-'
        r'lr(?P<lr>[0-9eE\.\-]+)-'
        r'sd(?P<model_seed>\d+)-'
        r'entcoef(?P<ent_coef>[0-9.]+)-'
        r'clipcoef(?P<clip_coef>[0-9.]+)-'
        r'l1(?P<l1_lambda>[0-9.]+)-'
        r'(?P<ppo_type>\w+)-MODEL.*\.pt$'
    )

    groups = defaultdict(lambda: {
        'params': None,
        'seeds': defaultdict(dict),
        'avg_reward': 0,
        'goal_reached_percent': 0,
        'first_model': None
    })

    model_files = [f for f in os.listdir(args.binaries_path) if f.endswith(".pt")]
    for model_file in tqdm(model_files, desc="Evaluating models"):
        
        tqdm.write(f"\nProcessing: {model_file}")
        
        match = model_pattern.match(model_file)
        if not match:
            tqdm.write(f"**** Skipping unmatched file: {model_file}")
            continue

        env_type = match.group(1)
        params = {
            'hidden_size': int(match.group('hidden_size')),
            'lr': float(match.group('lr')),
            'ent_coef': float(match.group('ent_coef')),
            'clip_coef': float(match.group('clip_coef')),
            'l1_lambda': float(match.group('l1_lambda')),
            'ppo_type': match.group('ppo_type'),
            'env_type': env_type,
            'model_seed': int(match.group('model_seed'))
        }

        group_key = (
            params['hidden_size'],
            params['lr'],
            params['ent_coef'],
            params['clip_coef'],
            params['l1_lambda'],
            params['ppo_type'],
            params['env_type']
        )

        if not groups[group_key]['params']:
            groups[group_key]['params'] = params
            groups[group_key]['first_model'] = model_file

        model_path = os.path.join(args.binaries_path, model_file)
        seed_results = {}
        for eval_seed in args.eval_seeds:
            reward, goal_reached = evaluate_model(
                model_path=model_path,
                ppo_type=params['ppo_type'],
                hidden_size=params['hidden_size'],
                eval_seed=eval_seed,
                max_steps=args.max_steps,
                args=args,
                env_type=env_type,
                test_mode=args.test_mode
            )
            seed_results[eval_seed] = (reward, goal_reached)

        groups[group_key]['seeds'][params['model_seed']] = seed_results
        
        # calculate goal reached percentage
        goal_reached_count = sum(1 for _, goal_reached in seed_results.values() if goal_reached)
        groups[group_key]['goal_reached_percent'] = (goal_reached_count / len(args.eval_seeds)) * 100

    sorted_groups = []
    for group_key, group_data in groups.items():
        all_rewards = []
        for model_file, seeds in group_data['seeds'].items():
            # all_rewards.extend(seeds.values())
            all_rewards.extend([reward for reward, _ in seeds.values()])
        group_data['avg_reward'] = np.mean(all_rewards) if all_rewards else 0
        sorted_groups.append((group_data['avg_reward'], group_data))

    sorted_groups.sort(reverse=True, key=lambda x: x[0])

    eval_name = args.binaries_path.split('/')[-2]   # [-1] is "binary"
    output_dir = os.path.join(project_root, "Scripts", "evaluation", env_type)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"eval_{eval_name}_{'test' if args.test_mode else 'train'}.txt")

    with open(output_filename, 'w') as f:
        f.write(f"=== {env_type} Evaluation Results ===\n")
        f.write(f"Evaluation seeds: {args.eval_seeds}\n")
        f.write(f"Test mode: {args.test_mode}\n\n")

        for avg_reward, group in sorted_groups:
            params = group['params']
            f.write(f"Hyperparameters:\n")
            f.write(f"Env: {params['env_type']}  ")
            f.write(f"H: {params['hidden_size']}  ")
            f.write(f"LR: {params['lr']:.0e}  ")
            f.write(f"Ent: {params['ent_coef']:.2f}  ")
            f.write(f"Clip: {params['clip_coef']:.2f}  ")
            f.write(f"L1: {params['l1_lambda']}\n")
            f.write(f"Avg Reward (over {len(group['seeds'])*len(args.eval_seeds)} runs): {avg_reward:.2f}\n")
            f.write(f"Goal Reached Percentage: {group['goal_reached_percent']:.2f}%\n")

            f.write("Results per training seed:\n")
            # for model_file, seeds in group['seeds'].items():
            for model_seed, seeds in sorted(group['seeds'].items()):
                rewards = []
                for es in args.eval_seeds:
                    reward_val, goal_reached = seeds[es]
                    if goal_reached:
                        rewards.append("Goal")
                    else:
                        rewards.append(f"{reward_val:.1f}")

                f.write(f"  sd {model_seed}: {' '.join(rewards)}\n")    
            f.write(f"First Model: {group['first_model']}\n")
            f.write("----------------------------------------\n")

    print(f"\nEvaluation saved to: {output_filename}")