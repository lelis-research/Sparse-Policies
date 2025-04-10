import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
import numpy as np
from environment.quad.quad import Quadcopter
from environment.quad.quad_po import QuadcopterPO
import pathlib
import argparse
import pygame


class QuadEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, n_steps=5000, use_po=False, test_mode=False, render_mode=None):
        super().__init__()
        self.use_po = use_po
        self.sim = QuadcopterPO(n_steps) if use_po else Quadcopter(n_steps)
        self.render_mode = render_mode
        self.test_mode = test_mode
        self.total_safe_error = 0.0
        self.screen = None
        self.clock = None
        self.total_rewrad = 0.0

        # Action space (vertical acceleration only)
        self.action_space = spaces.Box(
            low=np.array([-5.0]),
            high=np.array([5.0]),
            shape=(1,),
            dtype=np.float32
        )

        # Observation space
        if self.use_po:
            # QuadPO: [x, y, vx, vy]
            self.observation_space = spaces.Box(
                low=np.array([-np.inf]*4),
                high=np.array([np.inf]*4),
                dtype=np.float32
            )
        else:
            # Quad: [x, y, vx, vy, oyu, oyl, oxu, oxl]
            self.observation_space = spaces.Box(
                low=np.array([-np.inf]*8),
                high=np.array([np.inf]*8),
                dtype=np.float32
            )

        if self.use_po:
            if self.test_mode:
                self.sim.set_inp_limits((120, ))
            else:
                self.sim.set_inp_limits((60, ))
        else:   
            if self.test_mode:
                self.sim.set_inp_limits((80, ))
            else:
                self.sim.set_inp_limits((40, ))


    def step(self, action):
        self.state = self.sim.simulate(self.state, action, self.sim.dt)

        if self.use_po:
            obs = self.state[:4]  # [x, y, vx, vy]
        else:
            sensor_features = self.sim.get_features(self.state)   
            obs = np.concatenate([self.state[:4], sensor_features])

        self.total_safe_error += self.sim.check_safe(self.state)
        terminated = self.total_safe_error > 0.05 or np.sum(self.sim.check_goal(self.state)) < 0.01
        truncated = self.sim.done(self.state)

        if np.sum(self.sim.check_goal(self.state)) < 0.01:
            print("... Goal Reached! ...")
        
        reward = 1.0 
        self.total_rewrad += reward

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_rewrad = 0.0
        self.sim.counter = 0
        self.total_safe_error = 0.0
        if self.use_po:
            if self.test_mode:
                self.sim.set_inp_limits((120, ))
            else:
                self.sim.set_inp_limits((60, ))
        else:   
            if self.test_mode:
                self.sim.set_inp_limits((80, ))
            else:
                self.sim.set_inp_limits((40, ))
        self.state = self.sim.sample_init_state()

        return self.state[:4] if self.use_po else np.concatenate([self.state[:4], self.sim.get_features(self.state)]), {}

    def render(self):
        if self.render_mode is None:
            return

        if self.use_po:
            screen_width, screen_height = 7200, 800
        else:
            screen_width, screen_height = 4800, 800
        scale = 50  # Pixels per meter

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Quadcopter Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Draw tunnels
        tunnels = np.reshape(self.state[6:], (-1, 3))
        start_x = self.sim.x_start + self.sim.x_offset
        for t_yl, t_yu, t_l in tunnels:
            # Lower tunnel
            rect_lower = (start_x*scale, screen_height - t_yl*scale, t_l*scale, t_yl*scale)
            pygame.draw.rect(self.screen, (255,255,255), rect_lower)
            pygame.draw.rect(self.screen, (0,0,0), rect_lower, 2)

            # Upper tunnel
            rect_upper = (start_x*scale, 0, t_l*scale, (screen_height - t_yu*scale))
            pygame.draw.rect(self.screen, (255,255,255), rect_upper)
            pygame.draw.rect(self.screen, (0,0,0), rect_upper, 2)     
            
            start_x += t_l

        # Draw quadcopter
        x, y, _, _, angle, _ = self.state[:6]
        quad_size = self.sim.l * scale * 2
        quad_surface = pygame.Surface((quad_size, quad_size//10), pygame.SRCALPHA)
        quad_surface.fill((0, 150, 255))
        rotated = pygame.transform.rotate(quad_surface, np.degrees(angle))
        rect = rotated.get_rect(center=(x*scale, screen_height - y*scale))
        self.screen.blit(rotated, rect)

        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        elif self.render_mode == 'rgb_array':
            return np.transpose(pygame.surfarray.array3d(self.screen), (1,0,2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


def make_quad_env(max_episode_steps=5000, use_po=False):
    def thunk():
        env = QuadEnv(n_steps=max_episode_steps, 
                      use_po=use_po,
                      test_mode=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--po', action='store_true', help='Use partially observed version')
    parser.add_argument("--video_prefix", type=str, default="eval", help="Prefix for the video file name")

    args = parser.parse_args()

    video_dir = str(pathlib.Path(__file__).parent.resolve() / "videos")
    os.makedirs(video_dir, exist_ok=True)

    env = QuadEnv(render_mode='rgb_array', use_po=args.po, test_mode=False)
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode: episode == 0,
        name_prefix="quad_" + args.video_prefix,
    )
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"== obs: {obs}, action: {action}, reward: {reward}, done: {terminated or truncated}")
        total_reward += reward
        if terminated or truncated:
            done = True    
    print(f"\nTotal Reward: {total_reward}")
    env.close()