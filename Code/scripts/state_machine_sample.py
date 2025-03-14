import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import gymnasium as gym
import pathlib
from environment.cartpole.cartpole_gym import LastActionObservationWrapper, CustomForceWrapper
from gymnasium.wrappers import RecordVideo
import numpy as np



class StateMachinePolicy:
    def __init__(self):
        self.current_state = 'ms'  # Initial state

    def reset(self, initial_obs):
        # Transition from 'ms' to 'm1' or 'm2' based on initial observation's angular velocity (w)
        w = initial_obs[3]
        if w >= 0.02:
            self.current_state = 'm1'
        else:
            self.current_state = 'm2'

    def get_action(self):
        if self.current_state == 'm1':
            return 0  # Action 0 (left) corresponds to policy value -3.3
        elif self.current_state == 'm2':
            return 1  # Action 1 (right) corresponds to policy value 3.98
        else:
            raise ValueError(f"Invalid state: {self.current_state}")

    def update_state(self, new_obs):
        # Update state based on new observation
        w = new_obs[3]
        if self.current_state == 'm1':
            theta = new_obs[2]
            if w > 0.46 and theta > -0.06:
                self.current_state = 'm2'
        elif self.current_state == 'm2':
            if w < -0.46:
                self.current_state = 'm1'


def simulate_state_machine_policy():

    video_prefix = "state_machine_policy"
    train_mode = True

    if train_mode:
        max_episode_steps = 250  # 5 seconds
    else:
        max_episode_steps = 15000   # 5 minutes

    def make_env():
        return CustomForceWrapper(gym.make("CartPole-v1", 
                                                     max_episode_steps=max_episode_steps, # 300s / 0.02 = 15000 steps
                                                     render_mode="rgb_array"), 
                                            train_mode=train_mode, 
                                            last_action_in_obs=False)  

    base_env = make_env()

    video_dir = str(pathlib.Path(__file__).parent.resolve() / "videos/cartpole_state_machine")
    os.makedirs(video_dir, exist_ok=True)
    env = RecordVideo(
        base_env,
        video_folder=video_dir,
        name_prefix=video_prefix,
        episode_trigger=lambda episode: episode == 0
    )
    
    envs = gym.vector.SyncVectorEnv([lambda: env])

    policy = StateMachinePolicy()

    obs, _ = envs.reset(seed=42)
    envs.envs[0].render()
    policy.reset(obs[0])  # Initialize state based on initial observation

    total_reward = 0
    done = False
    step = 0
    while not done:
        action = policy.get_action()
        action_array = np.array([action])

        obs, reward, terminated, truncated, info = envs.step(action_array)
        current_obs = obs[0]
        current_reward = reward[0]
        print(f"Current state: {policy.current_state}, action: {action}, obs: {current_obs[2:]}")

        total_reward += current_reward
        done = np.any(terminated) or np.any(truncated)
        policy.update_state(current_obs)

    print(f"\nEpisode ended with total reward: {total_reward}")

if __name__ == "__main__":
    simulate_state_machine_policy()