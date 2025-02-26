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
from models.student import StudentPolicy
from agents import PPOAgent, GruAgent
import pathlib


def evaluate(args):

    if args.num_timesteps == 0:
        args.num_timesteps = 15000 if args.test_mode else 250 # 250 for training (5 seconds), 15000 for testing (5 minutes)

    print("\n=== Test mode ===") if args.test_mode else print("\n=== Train mode ===")

    def make_env():
        return LastActionObservationWrapper(gym.make("CartPole-v1", 
                                                     max_episode_steps=args.num_timesteps, # 300s / 0.02 = 15000 steps
                                                     render_mode="rgb_array"), 
                                            train_mode=(not args.test_mode), 
                                            last_action_in_obs=False)  

    base_env = make_env()

    video_dir = str(pathlib.Path(__file__).parent.resolve() / "videos/cartpole")
    os.makedirs(video_dir, exist_ok=True)
    
    # For recording a video
    env = RecordVideo(
        base_env,
        video_folder=video_dir,
        name_prefix=args.video_prefix,
        episode_trigger=lambda episode: episode == 0
    )
    envs = gym.vector.SyncVectorEnv([lambda: env])

    obs_shape = envs.single_observation_space.shape
    print(f"Observation Shape: {obs_shape}")
    action_space = envs.single_action_space
    print(f"Action Space: {action_space}")

    envs.reset()
    envs.envs[0].render()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_student = "student" in args.model_path.lower()

    if is_student:
        agent = StudentPolicy(input_dim=obs_shape[0]).to(device)
        agent.load_state_dict(torch.load(args.model_path, map_location=device))
    
    else:
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

        agent.load_state_dict(torch.load(args.model_path, map_location=device))
    
    print(f"\nModel loaded from {args.model_path}\n")
    agent.eval()

    
    MAX_STEPS = args.num_timesteps
    obs, _ = envs.reset(seed=1)
    done = False
    total_reward = 0
    step = 0

    if args.ppo_type == "original":
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                if is_student:
                    logits = agent(obs_tensor)
                    action = torch.argmax(logits, dim=-1).cpu().numpy()
                else:
                    action, _, _, _, _ = agent.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()

            obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.any(terminated) or np.any(truncated)
            total_reward += reward[0]  # Since we have only one environment
            step += 1
            if step >= MAX_STEPS:
                print("\nMax steps reached ==> Successful")
                done = True

    envs.close()
    print(f"\nTotal Reward = {total_reward}")


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
    parser.add_argument('--test_mode', action='store_true',
                        help="Determing the test/train distribution for evaluation")
    parser.add_argument('--feature_extractor', action='store_true',
                        help="Feature extractor to be used by the agent")
    parser.add_argument("--num_timesteps", type=int, default=0,
                        help="Number of timesteps to run evaluation")
    parser.add_argument("--video_folder", type=str, default="videos",
                        help="Folder to save the recorded video")
    parser.add_argument("--video_prefix", type=str, default="eval",
                        help="Prefix for the video file name")
    args = parser.parse_args()

    os.makedirs(args.video_folder, exist_ok=True)
    evaluate(args)