import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import gym
from gym import spaces
import numpy as np
from typing import Optional, Callable

from environment.karel_env.karel.environment import KarelEnvironment
from environment.karel_env.karel_tasks.top_off import TopOff, TopOffSparse
from environment.karel_env.karel_tasks.stair_climber import StairClimber, StairClimberSparse

class KarelGymEnv(gym.Env):
    """
    Gym environment wrapper for the KarelEnvironment.
    """
    metadata = {'render.modes': ['human', 'ansi']}
    SUPPORTED_TASKS = ['base', 'top_off', 'top_off_sparse', 
                      'stair_climber', 'stair_climber_sparse']

    def __init__(self, env_config: Optional[dict] = None, options: Optional[list] = None):
        super(KarelGymEnv, self).__init__()

        # Default configuration
        default_config = {
            'task_name': 'base',
            'env_height': 8,
            'env_width': 8,
            'max_steps': 100,
            'sparse_reward': False,
            # 'seed': None,
            'crash_penalty': -1.0,
            'initial_state': None
        }

        # Update with user-provided configuration
        if env_config is not None:
            default_config.update(env_config)
        self.config = default_config
        print("--- Config:", self.config)

        self._handle_initial_state()

        # Set random seed
        # self.seed(self.config['seed'])

        # Validate task name
        if self.config['task_name'] not in self.SUPPORTED_TASKS:
            raise ValueError(f"Task {self.config['task_name']} not supported. "
                           f"Choose from {self.SUPPORTED_TASKS}")

        # Initialize environment variables
        self.env_height = self.config['env_height']
        self.env_width = self.config['env_width']
        self.max_steps = self.config['max_steps']
        self.current_step = 0
        self.crash_penalty = self.config['crash_penalty']

        # Initialize the task
        self.task_name = self.config['task_name']
        self.task = self._initialize_task()

        # Define action and observation spaces
        self._set_action_observation_spaces()

        # Initialize the environment
        self.reset()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    def _initialize_task(self):
        env_args = {
            'env_height': self.env_height,
            'env_width': self.env_width,
            'crashable': True,
            'leaps_behaviour': False,
            'max_calls': self.max_steps,
            # 'seed': self.config.get('seed')
            'initial_state': self.config.get('initial_state') 
        }

        if self.task_name == 'top_off':
            task_class = TopOffSparse if self.config['sparse_reward'] else TopOff
            task = task_class(
                env_args=env_args,
                # crash_penalty=self.crash_penalty
            )
            task = task.generate_initial_environment(env_args)
        elif self.task_name == 'stair_climber':
            task_class = StairClimberSparse if self.config['sparse_reward'] else StairClimber
            task = task_class(
                env_args=env_args,
                # crash_penalty=self.crash_penalty
            )
            task = task.generate_initial_environment(env_args)
        elif self.task_name == 'base':
            # For base task, we initialize KarelEnvironment directly
            task = KarelEnvironment(**env_args)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        return task

    def _set_action_observation_spaces(self):
        # Base environment
        self.action_space = spaces.Discrete(len(self.task.actions_list))
        print("--- state shape:", self.task.state_shape)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=self.task.state_shape,
            dtype=np.float32
        )

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # Execute the action in the environment
        if hasattr(self.task, 'env'):
            action_name = self.task.env.actions_list[action]
            self.task.env.run_action(action_name)
            env = self.task.env
        else:
            action_name = self.task.actions_list[action]
            self.task.run_action(action_name)
            env = self.task

        # Increment the step counter
        self.current_step += 1

        # Get the reward and check if the episode is terminated
        if hasattr(self.task, 'get_reward'):
            terminated, reward = self.task.get_reward(env)
        else:
            # For 'base' task, define a default reward function
            terminated = self.current_step >= self.max_steps or env.is_crashed()
            reward = -1.0 if env.is_crashed() else 0.0  # Example reward

        # Handle max steps
        if self.current_step >= self.max_steps:
            terminated = True

        # Get the current observation
        observation = self._get_observation()

        # Info dictionary can include additional info if needed
        info = {}

        return observation, reward, terminated, info

    def reset(self):
        # Reset step counter
        self.current_step = 0

        # Reset the task
        if hasattr(self.task, 'reset_environment'):
            self.task.reset_environment()
        else:
            # For 'base' task, re-initialize the task
            self.task = self._initialize_task()

        # Get the initial observation
        observation = self._get_observation()
        return observation

    def render(self, mode='human'):
        if mode == 'human':
            if hasattr(self.task, 'env'):
                print("rendering ...", self.task.env.to_string())
            else:
                print("rendering ... \n", self.task.to_string(), "\n ...")
        elif mode == 'ansi':
            if hasattr(self.task, 'env'):
                return self.task.env.to_string()
            else:
                return self.task.to_string()
        else:
            super(KarelGymEnv, self).render(mode=mode)  # Just raise an exception

    def _get_observation(self):
        # Return the current state as the observation
        if hasattr(self.task, 'env'):
            return self.task.env.get_state().astype(np.float32)
        else:
            return self.task.get_state().astype(np.float32)
        
    def _handle_initial_state(self):
        initial_state = self.config.get('initial_state')
        if initial_state is not None:
            # Extract dimensions from initial_state
            if isinstance(initial_state, np.ndarray):
                num_features, env_height, env_width = initial_state.shape
            else:
                raise ValueError("initial_state must be a NumPy array")

            # Update env_height and env_width based on initial_state
            self.env_height = env_height
            self.env_width = env_width

            # Update the config dictionary to reflect the new dimensions
            self.config['env_height'] = env_height
            self.config['env_width'] = env_width
        else:
            # Use the dimensions from config
            self.env_height = self.config['env_height']
            self.env_width = self.config['env_width']




def make_karel_env(env_config: Optional[dict] = None) -> Callable:
    """
    Factory function to create a KarelGymEnv instance with the given configuration.
    """
    def thunk():
        env = KarelGymEnv(env_config=env_config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


# if __name__ == "__main__":
#     env_config = {
#         'task_name': 'top_off',
#         'env_height': 8,
#         'env_width': 8,
#         'max_steps': 5,
#         'sparse_reward': False,
#         # 'seed': 42,
#         'crash_penalty': -1.0
#     }
#     env = make_karel_env(env_config=env_config)()
#     obs = env.reset()
#     print("size of observation:", obs.shape)
#     print("Initial Observation:", obs)
#     env.render()
#     env.task.state2image(obs, root_dir=project_root + '/environment/').show()
#     done = False
#     total_reward = 0
#     while not done:
#         action = env.action_space.sample()  # Random action
#         print("-- Action:", action)
#         obs, reward, done, info = env.step(action)
#         print("-- Observation:", obs)
#         print("-- Reward:", reward)
#         total_reward += reward
#         env.render()
#         env.task.state2image(obs, root_dir=project_root + '/environment/').show()
#     print("Total Reward:", total_reward)


if __name__ == "__main__":
    import numpy as np

    # Define the initial state
    num_features = 16
    env_height = 2
    env_width = 5

    initial_state = np.zeros((num_features, env_height, env_width), dtype=bool)
    initial_state[1, 0, 0] = True  # Karel facing East at (0, 0)
    initial_state[4, 1, 2] = True  # Wall at (1, 2)

    env_config = {
        'task_name': 'base',
        'env_height': 4,
        'env_width': env_width,
        'max_steps': 5,
        'crash_penalty': -1.0,
        'initial_state': initial_state
    }

    env = make_karel_env(env_config=env_config)()
    obs = env.reset()
    print("size of observation:", obs.shape)
    print("Initial Observation:", obs)
    env.task.state2image(obs, root_dir=project_root + '/environment/').show()

    # Map action names to indices
    action_names = env.task.actions_list
    action_mapping = {name: idx for idx, name in enumerate(action_names)}

    # Define your action sequence
    action_sequence = ['move', 'turnRight', 'move', 'putMarker', 'pickMarker']
    actions = [action_mapping[name] for name in action_sequence]

    done = False
    total_reward = 0
    for action in actions:
        print("-- Action:", action_names[action])
        obs, reward, done, info = env.step(action)
        print("-- Observation:", obs)
        print("-- Reward:", reward)
        total_reward += reward
        env.render()
        img = env.task.state2image(obs, root_dir=project_root + '/environment/')
        img.show()
        if done:
            break
    print("Total Reward:", total_reward)





