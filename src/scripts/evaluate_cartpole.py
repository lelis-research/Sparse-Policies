import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from environment.cartpole.cartpole_gym import LastActionObservationWrapper, EasyCartPoleEnv
from models.student import StudentPolicy, StudentPolicySigmoid
from agents import PPOAgent, GruAgent
import pathlib
import re
from extract_policy import modify_model_weight


def evaluate(args):

    SEED = args.seed
    torch.manual_seed(SEED)

    easy_mode = False
    if "easy" in args.model_path.lower():
        easy_mode = True


    if args.num_timesteps == 0 and not easy_mode:
        args.num_timesteps = 15000 if args.test_mode else 250 # 250 for training (5 seconds), 15000 for testing (5 minutes)
    elif args.num_timesteps == 0 and easy_mode:
        args.num_timesteps = 30000 if args.test_mode else 500 # 500 for training (5 seconds), 30000 for testing (5 minutes)

    print("\n=== Test mode ===") if args.test_mode else print("\n=== Train mode ===")

    def make_env():
        return LastActionObservationWrapper(gym.make("CartPole-v1", 
                                                     max_episode_steps=args.num_timesteps, # 300s / 0.02 = 15000 steps
                                                     render_mode="rgb_array"), 
                                            train_mode=(not args.test_mode), 
                                            last_action_in_obs=False)  
    def make_env_easy():
        return EasyCartPoleEnv(train_mode=(not args.test_mode),
                               render_mode="rgb_array",
                               max_episode_steps=args.num_timesteps) # 300s / 0.01 = 30000 steps

    base_env = make_env() if not easy_mode else make_env_easy()

    # For recording a video
    video_dir = str(pathlib.Path(__file__).parent.resolve() / "videos/cartpoleEasy")
    os.makedirs(video_dir, exist_ok=True)
    env = RecordVideo(
        base_env,
        video_folder=video_dir,
        name_prefix=args.video_prefix,
        episode_trigger=lambda episode: episode == 0
    )
    
    envs = gym.vector.SyncVectorEnv([lambda: env])
    # envs = gym.vector.SyncVectorEnv([lambda: base_env])


    obs_shape = envs.single_observation_space.shape
    action_space = envs.single_action_space
    print(f"Observation Shape: {obs_shape}, Action Space: {action_space}")

    envs.reset(seed=SEED)
    envs.envs[0].render()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_student = "student" in args.model_path.lower()

    if is_student:
        match = re.search(r'sh(\d+)', args.model_path)
        try:
            student_hidden_size = int(match.group(1))
        except Exception as e:
            print(f"Error extracting student_hidden_size: {e}")
        
        # agent = StudentPolicy(input_dim=obs_shape[0], hidden_size=student_hidden_size).to(device)
        agent = StudentPolicySigmoid(input_dim=obs_shape[0], hidden_size=student_hidden_size).to(device)
        agent.load_state_dict(torch.load(args.model_path, map_location=device))
    
    else:
        if args.ppo_type == "original":
            arch_details = "cartpole_easy" if easy_mode else ""
            agent = PPOAgent(envs,
                            hidden_size=args.hidden_size,
                            feature_extractor=args.feature_extractor,
                            greedy=True,
                            arch_details=arch_details).to(device)
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

    # print("== Before ", agent.fc1.weight)
    # agent = modify_model_weight(agent, layer_name='fc1', neuron_idx=1, new_value=1.15)
    # print("== After ", agent.fc1.weight, "\n")

    
    MAX_STEPS = args.num_timesteps
    obs, _ = envs.reset(seed=SEED)
    done = False
    total_reward = 0
    step = 0

    if args.ppo_type == "original":
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                if is_student:
                    # # For Softmax student
                    # logits = agent(obs_tensor)
                    # action = torch.argmax(logits, dim=-1).cpu().numpy()

                    # For Sigmoid student
                    sigmoid_output = agent(obs_tensor)
                    action_scalar = (sigmoid_output >= 0.5).int().item()
                    # Format for vectorized environment
                    action = np.array([action_scalar])

                else:
                    action, _, _, _, _ = agent.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()

            obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.any(terminated) or np.any(truncated)
            total_reward += reward[0]  # Since we have only one environment

            # For Easy Cartpole
            if easy_mode:
                if -reward > 0.05:   # reward is the safe_error
                    print("Breaking because unsafe")
                    done = True 

            step += 1
            if step >= MAX_STEPS:
                print("\nMax steps reached ==> Successful")
                done = True

    print(f"Steps: {step}")
    envs.close()
    print(f"\nTotal Reward = {total_reward}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate a PPO model on CartPole with video recording.")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model file (e.g., binary/PPO-Cartpole-...pt)")
    parser.add_argument("--ppo_type", type=str, default="original", choices=["original", "gru"], help="Type of PPO agent to use")
    parser.add_argument("--hidden_size", type=int, required=True, help="Hidden size for the agent network")
    parser.add_argument('--test_mode', action='store_true', help="Determing the test/train distribution for evaluation")
    parser.add_argument('--feature_extractor', action='store_true', help="Feature extractor to be used by the agent")
    parser.add_argument("--num_timesteps", type=int, default=0, help="Number of timesteps to run evaluation")
    parser.add_argument("--video_folder", type=str, default="videos", help="Folder to save the recorded video")
    parser.add_argument("--video_prefix", type=str, default="eval", help="Prefix for the video file name")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for the environment")
    
    args = parser.parse_args()

    os.makedirs(args.video_folder, exist_ok=True)
    evaluate(args)