import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
import numpy as np
from environment.car.car_simulation import CarReversePP
import pathlib
import argparse
import time
import matplotlib.pyplot as plt


class CarEnv(gym.Env):
    """
    A Gymnasium environment wrapper for the CarReversePP simulation.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, n_steps=100, render_mode=None, test_mode=False):
        super().__init__()
        self.sim = CarReversePP(n_steps=n_steps)
        self.render_mode = render_mode
        
        # Observation space: [x, y, angle, distance]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -5.0, -np.pi, 0.0]),
            high=np.array([5.0, 20.0, np.pi, 30.0]),
            dtype=np.float32
        )
        
        # Action space: [velocity, angular rate]
        self.action_space = spaces.Box(
            low=np.array([-5.0, -0.5]),
            high=np.array([5.0, 0.5]),
            dtype=np.float32
        )
        self.state = None
        self.test_mode = test_mode

        if self.test_mode:
            test_limit = (11, 12)
            self.sim.set_inp_limits(test_limit)
        else:
            train_limit = (12, 13.5)
            self.sim.set_inp_limits(train_limit)

    def step(self, action):

        self.sim.last_action = action

        dt = self.sim.dt
        next_state = self.sim.simulate(self.state, action, dt)
        self.state = next_state

        goal_err = self.sim.check_goal(self.state)
        reward = - (goal_err[0] + goal_err[1])
        
        collision = self.sim.check_safe(self.state)
        terminated = collision > 0
        truncated = self.sim.done(self.state)
        print(f"==== Collision: {collision}, Terminated: {terminated}, Truncated: {truncated}, Action: {action}")
        
        if terminated:
            reward -= 100
            
        info = {}
        
        if self.render_mode == 'human':
            self.render()
            
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset_render()
        self.state = self.sim.sample_init_state()
        self.sim.counter = 0

        if self.test_mode:
            test_limit = (11, 12)
            self.sim.set_inp_limits(test_limit)
        else:
            train_limit = (12, 13.5)
            self.sim.set_inp_limits(train_limit)

        return self.state, {}

    def render(self):
        if self.render_mode in ('human', 'rgb_array'):
            return self.sim.render(self.state, self.render_mode)
        return None

    def close(self):
        self.sim.reset_render()


def make_car_env(max_episode_steps=100):
    def thunk():
        env = CarEnv(n_steps=max_episode_steps, render_mode=None, test_mode=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run CarEnv in either simulation or test mode.")

    parser.add_argument('--mode', type=str, default='gym', choices=['gym', 'test'],
                        help="Choose 'gym' for simulation or 'test' for trajectory plotting.")
    parser.add_argument("--video_prefix", type=str, default="eval",
                        help="Prefix for the video file name")
    args = parser.parse_args()

    if args.mode == 'gym':

        video_dir = str(pathlib.Path(__file__).parent.resolve() / "videos")
        os.makedirs(video_dir, exist_ok=True)

        env = CarEnv(n_steps=200, render_mode='rgb_array', test_mode=False)
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode: episode == 0,
            name_prefix="parking_" + args.video_prefix,
        )
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        state_action_list = []
        collision_states = []
        while not done:

            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)

            # Store collision for plotting
            collision = env.sim.check_safe(state)
            if collision > 0:
                collision_states.append(state)

            # Store states for plotting
            state_action_list.append((state, action))

            total_reward += reward
            done = terminated or truncated
            
            time.sleep(env.sim.dt)
            
            if terminated or truncated:
                break

        print(f"Total Reward: {total_reward}")
        env.close()

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


    # Test in the original code
    elif args.mode == 'test':

        from math import pi
        def simulate_bicycle(state, action, dt):
            ns = np.copy(state)
            v, w = action 
            w = w / 10.0
            x, y, ang = ns  
            beta = np.arctan(0.5 * np.tan(w))
            dx = v * np.cos(ang + beta) * dt 
            dy = v * np.sin(ang + beta) * dt 
            da = v / 2.5 * np.sin(beta) * dt 
            ns[0] += dx 
            ns[1] += dy 
            ns[2] += da 
            return ns 

        def get_all_vertices(x, y, ang, w, h):
            res = []
            db = w / 2.0
            da = h / 2.0
            coa = np.cos(ang)
            sia = np.sin(ang)
            res.append((x + da * coa + db * sia, y + da * sia - db * coa))
            res.append((x + da * coa - db * sia, y + da * sia + db * coa))
            res.append((x - da * coa - db * sia, y - da * sia + db * coa))
            res.append((x - da * coa + db * sia, y - da * sia - db * coa))
            return res 

        def get_traj(s, a, T, w=1.8, h=5.0):
            X = []
            Y = []
            X1 = []
            Y1 = []  
            
            vertices = get_all_vertices(s[0], s[1], s[2], w, h)
            X.append(s[0])
            Y.append(s[1])
            # Using the average of the bottom vertices for one set of points
            X1.append((vertices[2][0] + vertices[3][0]) / 2.0)
            Y1.append((vertices[2][1] + vertices[3][1]) / 2.0)

            for i in range(T):
                s = simulate_bicycle(s, a, 0.01)
                vertices = get_all_vertices(s[0], s[1], s[2], w, h)
                X.append(s[0])
                Y.append(s[1])
                X1.append((vertices[2][0] + vertices[3][0]) / 2.0)
                Y1.append((vertices[2][1] + vertices[3][1]) / 2.0)
            
            color = 'g' if a[0] > 0 else 'r'
            plt.plot(X, Y, color, label="Car center" if i == 0 else "")
            plt.plot(X1, Y1, color+"--", label="Rear mid-point" if i == 0 else "")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title("Test Trajectory Plot")
            plt.legend()
            return s 

        # Set initial state: [x, y, angle]
        s = np.array([0.0, 0.0, pi/2.0])
        # Simulate a series of maneuvers (similar to the provided test code)
        s = get_traj(s, (5, 5), 20)
        s = get_traj(s, (-5, -5), 20)
        s = get_traj(s, (5, 5), 20)
        s = get_traj(s, (-5, -5), 20)
        s = get_traj(s, (5, 5), 20)
        
        print("Displaying test trajectory plot...")
        plt.show()
        plt.close()