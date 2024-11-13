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


def evaluate_model_on_large_grid(model_path, args):
    # Configuration for the large grid environment
    env_config = {
        'task_name': args.task_name,
        'env_height': args.game_height_eval,
        'env_width': args.game_width_eval,
        'max_steps': args.max_steps,
        'sparse_reward': args.sparse_reward,
        'crash_penalty': args.crash_penalty,
        'seed': 42,  # Not equal to "karel_seed"
        'initial_state': None
    }

    def make_env():
        return KarelGymEnv(env_config=env_config)

    envs = gym.vector.SyncVectorEnv([make_env])

    obs_shape = envs.single_observation_space.shape
    print(f"Observation Shape: {obs_shape}")
    action_space = envs.single_action_space
    print(f"Action Space: {action_space}")

    envs.reset()
    envs.envs[0].render()
    envs.envs[0].task.state2image(envs.envs[0].get_observation(), root_dir=project_root+'/environment/').show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.ppo_type == "original":
        agent = PPOAgent(envs, hidden_size=args.hidden_size).to(device)
    elif args.ppo_type == "gru":
        agent = GruAgent(envs, h_size=args.hidden_size).to(device)

    agent.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    agent.eval()

    MAX_STEPS = 1000

    total_rewards = []
    for episode in range(args.num_episode):
        obs, _ = envs.reset(seed=episode)
        done = False
        episode_reward = 0
        step = 0

        if args.ppo_type == "original":
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

                with torch.no_grad():
                    action, _, _, _, _ = agent.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()

                obs, reward, terminated, truncated, infos = envs.step(action)
                done = np.any(terminated) or np.any(truncated)
                episode_reward += reward[0]  # Since we have only one environment
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
                episode_reward += reward[0]  # Since we have only one environment
                step += 1
                if step > MAX_STEPS:
                    done = True

                # Update the done tensor for the next step
                done_tensor = torch.tensor(terminated | truncated, dtype=torch.bool).to(device)

                # envs.envs[0].render()
                envs.envs[0].task.state2image(envs.envs[0].get_observation(), root_dir=project_root+'/environment/').show()

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
        total_rewards.append(episode_reward)

    average_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {args.num_episode} episodes: {average_reward}")

    envs.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', default="stair_climber", type=str, help="[stair_climber, maze]")
    parser.add_argument('--game_width', default=10, type=int)
    parser.add_argument('--game_height', default=10, type=int)
    parser.add_argument('--game_width_eval', default=100, type=int)
    parser.add_argument('--game_height_eval', default=100, type=int)
    parser.add_argument('--max_steps', default=100, type=int)
    parser.add_argument('--sparse_reward', action='store_true')
    parser.add_argument('--crash_penalty', default=-1.0, type=float)
    parser.add_argument('--karel_seed', default=100, type=int, help="For recreating the same environment")

    parser.add_argument('--ppo_type', default="original", type=str, help="[original, lstm, gru]")
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=0.005, type=float)
    parser.add_argument('--clip_coef', default=0.01, type=float)
    parser.add_argument('--ent_coef', default=0.01, type=float)

    parser.add_argument('--model_seed', default=0, type=int)
    parser.add_argument('--log_path', default="logs/", type=str)
    parser.add_argument('--num_episode', default=1, type=int)

    args = parser.parse_args()
    print(vars(args))

    model_file_name = f'binary/PPO-Karel_{args.task_name}-gw{args.game_width}-gh{args.game_height}-h{args.hidden_size}-lr{args.learning_rate}-sd{args.model_seed}-entcoef{args.ent_coef}-clipcoef{args.clip_coef}_{args.ppo_type}_MODEL.pt'
    evaluate_model_on_large_grid(model_file_name, args)
