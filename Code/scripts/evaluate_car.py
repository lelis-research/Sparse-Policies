import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from environment.car.car_gym import CarEnv
from environment.car.car_simulation import CarReversePP
from agents import PPOAgent, GruAgent
import time
import matplotlib.pyplot as plt
import pathlib


def evaluate(args):

    def make_env():
        return CarEnv(n_steps=args.num_timesteps, render_mode="rgb_array", test_mode=args.test_mode)  

    base_env = make_env()

    video_dir = str(pathlib.Path(__file__).parent.resolve() / "videos/car")
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
    action_space = envs.single_action_space.shape
    print(f"Action Space: {action_space}")

    envs.reset()
    # envs.envs[0].render()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # # After loading the agent
    # print("Agent's action_scale:", agent.action_scale)
    # print("Agent's action_bias:", agent.action_bias)

    MAX_STEPS = args.num_timesteps

    obs, _ = envs.reset(seed=1)
    done = False
    total_reward = 0
    step = 0
    state_action_list = []
    collision_states = []

    if args.ppo_type == "original":
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()
            
            print(f"== Step: {step}, Action: {action}")

            obs, reward, terminated, truncated, infos = envs.step(action)

            # Store collision for plotting
            state = obs.flatten()[:4]
            collision = envs.envs[0].sim.check_safe(state)
            if collision > 0:
                collision_states.append(state)

            # Store states for plotting
            state_action_list.append((state, action.flatten()))

            time.sleep(envs.envs[0].sim.dt)
            done = np.any(terminated) or np.any(truncated)
            total_reward += reward[0]  # Since we have only one environment
            step += 1
            if step > MAX_STEPS:
                print("Max steps reached ==> Successful")
                done = True

    envs.close()
    print(f"\nTotal Reward = {total_reward}")

    ###### Plotting the trajectory ######
    sim_plot = CarReversePP()
    plt.figure(figsize=(4, 8))

    start_state = [state_action_list[0][0]]       
    goal_state  = [state_action_list[-1][0]]       
    sim_plot.plot_init_paper(start_state, goal_state)
    sim_plot.plot_states(state_action_list, line=True)
    sim_plot.plot_collision_states(collision_states)

    plt.title("Sample Trajectory")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO model on Car (parallel park) with video recording."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model file (e.g., binary/PPO-car-...pt)")
    parser.add_argument("--ppo_type", type=str, default="original", choices=["original", "gru"],
                        help="Type of PPO agent to use")
    parser.add_argument("--hidden_size", type=int, required=True,
                        help="Hidden size for the agent network")
    parser.add_argument('--feature_extractor', action='store_true',
                        help="Feature extractor to be used by the agent")
    parser.add_argument('--test_mode', action='store_true',
                        help="Set test mode for the environment")
    parser.add_argument("--num_timesteps", type=int, default=100,
                        help="Number of timesteps to run evaluation")
    parser.add_argument("--video_prefix", type=str, default="eval",
                        help="Prefix for the video file name")
    args = parser.parse_args()

    evaluate(args)