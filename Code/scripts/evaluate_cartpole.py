import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from environment.cartpole_gym import LastActionObservationWrapper
from agents import PPOAgent, GruAgent


def evaluate(args):

    def make_env():
        return LastActionObservationWrapper(gym.make("CartPole-v1", render_mode="rgb_array"))

    # Create base environment
    base_env = make_env()
    
    # Wrap with RecordVideo
    env = RecordVideo(
        base_env,
        video_folder=args.video_folder,
        name_prefix=args.video_prefix,
        episode_trigger=lambda episode: True
    )

    # envs = gym.vector.SyncVectorEnv([make_env])
    # Create vector env wrapper for agent compatibility
    envs = gym.vector.SyncVectorEnv([lambda: env])

    obs_shape = envs.single_observation_space.shape
    print(f"Observation Shape: {obs_shape}")
    action_space = envs.single_action_space
    print(f"Action Space: {action_space}")

    envs.reset()
    envs.envs[0].render()

    # Create the environment
    # env = gym.make("CartPole-v1")
    # env = LastActionObservationWrapper(env)
    # Record a video for every episode
    # env = RecordVideo(
    #     env,
    #     video_folder=args.video_folder,
    #     name_prefix=args.video_prefix,
    #     episode_trigger=lambda episode: True
    # )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the appropriate agent based on ppo_type
    if args.ppo_type == "original":
        agent = PPOAgent(envs,
                         hidden_size=args.hidden_size,
                         feature_extractor=args.feature_extractor,
                         greedy=True).to(device)
    elif args.ppo_type == "gru":
        agent = GruAgent(envs,
                         h_size=args.hidden_size,
                         feature_extractor=args.feature_extractor,
                         greedy=True).to(device)
    else:
        raise ValueError(f"Unsupported ppo_type: {args.ppo_type}")

    # Load the model state dictionary
    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")
    agent.eval()

    MAX_STEPS = args.num_timesteps

    obs, _ = envs.reset()
    done = False
    episode_reward = 0
    step = 0
        

    if args.ppo_type == "original":
        while not done:
            # envs.envs[0].render()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()

            obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.any(terminated) or np.any(truncated)
            episode_reward += reward[0]  # Since we have only one environment
            step += 1
            if step > MAX_STEPS:
                print("Max steps reached.")
                done = True

    envs.close()
    print(f"Total Reward = {episode_reward}")
    print(f"Evaluation complete after {step} timesteps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO model on CartPole with video recording."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model file (e.g., binary/PPO-Cartpole-...pt)")
    parser.add_argument("--ppo_type", type=str, default="original", choices=["original", "gru"],
                        help="Type of PPO agent to use")
    parser.add_argument("--hidden_size", type=int, required=True,
                        help="Hidden size for the agent network")
    parser.add_argument('--feature_extractor', action='store_true',
                        help="Feature extractor to be used by the agent")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of timesteps to run evaluation")
    parser.add_argument("--video_folder", type=str, default="videos",
                        help="Folder to save the recorded video")
    parser.add_argument("--video_prefix", type=str, default="eval",
                        help="Prefix for the video file name")
    args = parser.parse_args()

    # Ensure the video folder exists
    os.makedirs(args.video_folder, exist_ok=True)
    evaluate(args)