import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import gymnasium as gym
import numpy as np
from typing import Optional, Callable, List, Any
import torch
import random

from environment.karel_env.karel.environment import KarelEnvironment, basic_actions
from environment.karel_env.karel_tasks.stair_climber import StairClimber, StairClimberSparse, StairClimberSparseAllInit, StairClimberAllInit
from environment.karel_env.karel_tasks.maze import Maze, MazeSparse, MazeSparseAllInit, MazeAllInit
from environment.karel_env.karel_tasks.four_corners import FourCorners, FourCornersSparse
from environment.karel_env.karel_tasks.top_off import TopOff, TopOffSparse
from environment.karel_env.karel_tasks.harvester import Harvester, HarvesterSparse


class KarelGymEnv(gym.Env):
    """
    Gym environment wrapper for the KarelEnvironment.
    """
    metadata = {'render.modes': ['human', 'ansi']}
    SUPPORTED_TASKS = ['base', 'top_off', 'top_off_sparse', 
                      'stair_climber', 'stair_climber_sparse',
                      'maze', 'maze_sparse',
                      'four_corner', 'four_corner_sparse',
                      'top_off', 'top_off_sparse',
                      'harvester', 'harvester_sparse']

    def __init__(self, env_config: Optional[dict] = None, options: Optional[list] = None):
        super(KarelGymEnv, self).__init__()

        default_config = {
            'task_name': 'base',
            'env_height': 8,
            'env_width': 8,
            'max_steps': 100,
            'sparse_reward': False,
            'seed': None,
            'crash_penalty': -1.0,
            'initial_state': None
        }

        if env_config is not None:
            default_config.update(env_config)
        self.config = default_config
        # print("--- Config:", self.config)

        self._handle_initial_state()

        # Set random seed
        self.seed(self.config['seed'])

        if self.config['task_name'] not in self.SUPPORTED_TASKS:
            raise ValueError(f"Task {self.config['task_name']} not supported. "
                           f"Choose from {self.SUPPORTED_TASKS}")

        # Initialize environment variables
        self.env_height = self.config['env_height']
        self.env_width = self.config['env_width']
        self.max_steps = self.config['max_steps']
        self.current_step = 0
        # self.crash_penalty = self.config['crash_penalty']
        self.reward_diff = self.config['reward_diff'] if 'reward_diff' in self.config else False
        self.rescale_reward = self.config['reward_scale'] if 'reward_scale' in self.config else True
        self.multi_initial_confs = self.config['multi_initial_confs'] if 'multi_initial_confs' in self.config else False
        self.all_initial_confs = self.config['all_initial_confs'] if 'all_initial_confs' in self.config else False
        self.all_initial_confs_envs = None
        self.last_action = -1.0

        # Initialize the task
        self.task_name = self.config['task_name']
        self.task, self.task_specific = self._initialize_task()

        self._set_action_observation_spaces(options)

        self.reset()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    def _initialize_task(self):
        env_args = {
            'env_height': self.env_height,
            'env_width': self.env_width,
            'crashable': False,
            'leaps_behaviour': False,
        }

        if self.task_name == 'top_off':
            task_class = TopOffSparse if self.config['sparse_reward'] else TopOff
            task_specific = task_class(
                env_args=env_args,
                seed=self.config.get('seed'),
                # crash_penalty=self.crash_penalty
            )
            task = task_specific.generate_initial_environment(env_args)

        elif self.task_name == 'stair_climber':
            if self.all_initial_confs:
                task_class = StairClimberSparseAllInit if self.config['sparse_reward'] else StairClimberAllInit
                task_specific = task_class(
                    env_args=env_args,
                    reward_diff=self.reward_diff,
                    rescale_reward=self.rescale_reward
                )
                self.all_initial_confs_envs = task_specific.all_initial_confs
            else:
                task_class = StairClimberSparse if self.config['sparse_reward'] else StairClimber
                task_specific = task_class(
                    env_args=env_args,
                    seed=self.config.get('seed'),
                    reward_diff=self.reward_diff,
                    rescale_reward=self.rescale_reward
                )
            task = task_specific.generate_initial_environment(env_args)

        elif self.task_name == 'maze':
            if self.all_initial_confs:
                task_class = MazeSparseAllInit if self.config['sparse_reward'] else MazeAllInit
                task_specific = task_class(env_args=env_args)
                self.all_initial_confs_envs = task_specific.all_initial_confs
            else:
                task_class = MazeSparse if self.config['sparse_reward'] else Maze
                task_specific = task_class(
                    env_args=env_args,
                    seed=self.config.get('seed'),
                )
            task = task_specific.generate_initial_environment(env_args)

        elif self.task_name == 'four_corner':
            task_class = FourCornersSparse if self.config['sparse_reward'] else FourCorners
            task_specific = task_class(
                env_args=env_args,
                seed=self.config.get('seed'),
            )
            task = task_specific.generate_initial_environment(env_args)

        elif self.task_name == 'harvester':
            task_class = HarvesterSparse if self.config['sparse_reward'] else Harvester
            task_specific = task_class(
                env_args=env_args,
                seed=self.config.get('seed'),
            )
            task = task_specific.generate_initial_environment(env_args)

        elif self.task_name == 'base':
            # we need to pass the initial state to the base task if we want a custom initial state
            env_args['initial_state'] = self.config.get('initial_state')
            task = KarelEnvironment(**env_args)
            task_specific = task
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        return task, task_specific

    def _set_action_observation_spaces(self, options: Optional[list] = None):
        # num_features = self.task.state_shape[0]
        observation_shape = self._get_observation_dsl().shape        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=observation_shape,
            dtype=np.float32
        )

        if options is not None:
            self.setup_options(options)
        else:
            self.action_space = gym.spaces.Discrete(len(self.task.actions_list))
            self.program_stack = None
            self.option_index = None

    def setup_options(self, options:List[Any]=None):
        """
        Enables the corresponding agents to choose from both actions and options
        """
        self.option_index = len(self.task.actions_list)
        self.program_stack = [basic_actions(i) for i in range(self.option_index)] + options # TODO: test this basic actions and replace it with something more general
        self.action_space = gym.spaces.Discrete(len(self.program_stack))
        self.option_sizes = [3 for _ in range(len(options))]    # TODO: change this to a more general way

    def step(self, action:int):
        # print("---- action index:", action)
        assert self.action_space.contains(action), "Invalid action"
        self.last_action = action
        truncated = False
        def process_action(action:int):
            nonlocal truncated
            action_name = self.task.actions_list[action]
            self.task.run_action(action_name)

            self.current_step += 1
            # print("-- Step:", self.current_step)

            # Get the reward and check if the episode is terminated
            if self.task_name != 'base':
                terminated, reward = self.task_specific.get_reward(self.task)

            if self.current_step >= self.max_steps:
                truncated = True

            # if terminated or truncated: print("-- Episode Done!!")
            # print("truncate:", truncated, "terminated:", terminated)

            return self._get_observation_dsl(), reward, terminated, truncated, {}
                
        # helper function for executing options
        def choose_action(env, model, greedy=False, verbose=False):
            _epsilon = 0.3
            _is_recurrent = False
            if random.random() < _epsilon:
                actions = env.get_actions()
                a = actions[random.randint(0, len(actions) - 1)]
            else:
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                if _is_recurrent:
                    prob_actions, _h = model(x_tensor, _h)
                else:
                    prob_actions = model(x_tensor)
                    # print("prob_actions: ", prob_actions)
                if greedy:
                    print("prob_actions: ", prob_actions)
                    a = torch.argmax(prob_actions).item()
                else:
                    print("prob_actions: ", prob_actions)
                    a = torch.multinomial(prob_actions, 1).item()
            return a
    
        def check_stopping(env, model_y2, verbose=False):
            """
            This method checks the stopping condition using model_y2.
            Returns True if the agent should stop, otherwise False.
            """
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            stopping_prob = model_y2(x_tensor).item()  # model_y2 outputs a probability
            if verbose:
                print(f"-- Stopping probability: {stopping_prob}")
            return stopping_prob <= 0.5

        # Execute the option
        if self.option_index and action >= self.option_index:
            print("--- Outside process action")
            reward_sum = 0
            option = self.program_stack[action]
            verbose = True
            if verbose: print("-- Option: ", option.sequence, "-- index: ", action)
            sequence_ended = False  # Flag to indicate the end of a sequence
            terminated, truncated = False, False

            if verbose: print('-- Beginning Option')

            current_length = 0
            max_length = 20
            while not sequence_ended and not terminated and not truncated:

                # Choose action using model_y1
                a = choose_action(self.task, option.model_y1, greedy=False, verbose=verbose)

                if verbose: 
                    # print(self.task, a, "\n")
                    print("-- opt action: ", a)

                # Check stopping condition using model_y2
                if check_stopping(self.task, option.model_y2, verbose=verbose):
                    if verbose: print("-- Stopping the current sequence based on model_y2.")
                    sequence_ended = True  # End the current sequence, but continue the outer loop
                    
                # Apply the chosen action
                obs, reward, terminated, truncated, _ = process_action(a)
                reward_sum += reward

                current_length += 1
                if current_length >= max_length:
                    sequence_ended = True

            return obs, reward_sum, terminated, truncated, {}
        else:
            return process_action(action)

    def reset(self, seed=0, options=None):
        self.current_step = 0
        self.last_action = -1.0
        
        if self.multi_initial_confs:   # choose between 10 random initial setups
            selected_seed = random.choice(list(range(10)))
            self.config['seed'] = selected_seed
            self.seed(selected_seed)
            self.task, self.task_specific = self._initialize_task()

        elif self.all_initial_confs:
            selected_conf = random.choice(self.all_initial_confs_envs)
            self.config['initial_state'] = selected_conf.copy()
            env_args = {
                'env_height': self.env_height,
                'env_width': self.env_width,
                'crashable': False,
                'leaps_behaviour': False,
                'initial_state': self.config.get('initial_state')
            }
            self.task = self.task_specific.generate_initial_environment(env_args)

        else:
            self.task, self.task_specific = self._initialize_task()

        # print(self.task.state2image(root_dir=project_root + '/environment/').show())
        return self._get_observation_dsl(), {}

    def render(self, mode='human'):
        if mode == 'human':
            print(self.task.to_string(), "\n")
        elif mode == 'ansi':
            return self.task.to_string()
        else:
            super(KarelGymEnv, self).render(mode=mode)

    def get_observation(self) -> np.ndarray:
        return self.task.get_state().astype(np.float32)
        
    def _get_reduced_observation(self) -> np.ndarray:
        """
        Returns a reduced observation containing only the current cell, the cell in front,
        the left cell, and the right cell.
        """
        if hasattr(self.task, 'env'):
            env = self.task.env
        else:
            env = self.task

        # Get the full state
        full_observation = env.get_state()

        # Get agent's position and orientation
        row, col, d = env.get_hero_pos()

        # Define direction deltas
        # Directions: 0 - North, 1 - East, 2 - South, 3 - West
        direction_deltas = {
            0: (-1, 0),  # North
            1: (0, 1),   # East
            2: (1, 0),   # South
            3: (0, -1)   # West
        }

        # Current cell
        current_cell_features = full_observation[:, row, col]

        # Front cell
        dr_front, dc_front = direction_deltas[d]
        front_row, front_col = row + dr_front, col + dc_front

        # Left cell
        d_left = (d - 1) % 4
        dr_left, dc_left = direction_deltas[d_left]
        left_row, left_col = row + dr_left, col + dc_left

        # Right cell
        d_right = (d + 1) % 4
        dr_right, dc_right = direction_deltas[d_right]
        right_row, right_col = row + dr_right, col + dc_right

        def get_cell_features(r, c):
            if 0 <= r < env.state_shape[1] and 0 <= c < env.state_shape[2]:
                return full_observation[:, r, c]
            else:
                # Return a zero vector if out of bounds
                return np.zeros(full_observation.shape[0], dtype=full_observation.dtype)

        front_cell_features = get_cell_features(front_row, front_col)
        left_cell_features = get_cell_features(left_row, left_col)
        right_cell_features = get_cell_features(right_row, right_col)

        # Combine the features into a single array
        reduced_observation = np.stack([
            current_cell_features,
            front_cell_features,
            left_cell_features,
            right_cell_features
        ], axis=0)  # Shape: (4, num_features)

        return reduced_observation.flatten()

    def _get_observation_dsl(self) -> np.ndarray:
        """
        Returns an observation that a DSL agent would see but for our RL agent
        """
        num_actions = 5 + 1 # number of actions + 1
        one_hot_action = np.zeros(num_actions, dtype=float)
        
        if self.last_action is not None and self.last_action != -1: 
            one_hot_action[int(self.last_action) + 1] = 1.0
        elif self.last_action == -1:
            one_hot_action[0] = 1.0

        # num_actions = 5   # number of actions
        # one_hot_action = np.zeros(num_actions, dtype=float)
        # if self.last_action is not None and self.last_action != -1: 
        #     one_hot_action[int(self.last_action)] = 1.0

        dsl_obs = np.array([
            self.task.get_bool_feature("frontIsClear"),
            self.task.get_bool_feature("leftIsClear"),
            self.task.get_bool_feature("rightIsClear"),
            self.task.get_bool_feature("markersPresent"),
        ], dtype=float)

        dsl_obs = np.concatenate((dsl_obs, one_hot_action))

        return dsl_obs

    def _handle_initial_state(self):
        initial_state = self.config.get('initial_state')
        if initial_state is not None:
            # Extract dimensions from initial_state
            if isinstance(initial_state, np.ndarray):
                num_features, env_height, env_width = initial_state.shape
            else:
                raise ValueError("initial_state must be a NumPy array")

            self.env_height = env_height
            self.env_width = env_width

            self.config['env_height'] = env_height
            self.config['env_width'] = env_width
        else:
            # print("---- Using env_height and env_width from input ----")
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


if __name__ == "__main__":

    num_features = 16
    env_height = 8
    env_width = 8

    # A custom initial state for the base task
    initial_state = np.zeros((num_features, env_height, env_width), dtype=bool)
    initial_state[1, 0, 0] = True  # Karel facing East at (0, 0)
    initial_state[4, 1, 2] = True  # Wall at (1, 2)

    env_config = {
        'task_name': 'maze',
        'env_height': env_height,
        'env_width': env_width,
        'max_steps': 1,
        'sparse_reward': True,
        'seed': 1,
        'initial_state': None,
        'multi_initial_confs': False, 
        'all_initial_confs': True
    }

    env = make_karel_env(env_config=env_config)()

    # # showing all the initial configurations
    # print("len of all initial confs:", len(env.all_initial_confs_envs))
    # for init_conf in env.all_initial_confs_envs:
    #     env.task.state2image(init_conf, root_dir=project_root + '/environment/').show()

    init_obs = env.reset()
    env.render()
    # env.task.state2image(env.get_observation(), root_dir=project_root + '/environment/').show()

    action_names = env.task.actions_list
    action_mapping = {name: idx for idx, name in enumerate(action_names)}
    action_sequence = ['move', 'turnLeft', 'move', 'move', 'turnRight', 'move', 'turnLeft', 'move', 'turnRight', 'move'] # for stairclimber 6*6
    # action_sequence = ['move', 'putMarker', 'turnLeft', 'move', 'move', 'move', 'move', 'move', 'putMarker', 'turnLeft', 'move', 'move', 'move', 'move', 'move', 'putMarker', 'turnLeft', 'move', 'move', 'move', 'move', 'move', 'putMarker', 'turnLeft', 'move', 'move', 'move', 'move', 'move'] # for stairclimber 6*6
    # action_sequence = ['pickMarker', 'move', 'pickMarker', 'turnLeft', 'move', 'pickMarker']
    # action_sequence = ['move', 'move', 'move', 'putMarker', 'move', 'move', 'putMarker']
    
    actions = [action_mapping[name] for name in action_sequence]

    done = False
    total_reward = 0
    for action in actions:
        print("--- Action:", action_names[action])
        obs, reward, done, truncated, info = env.step(action)
        print("--- Reward:", reward)
        print("--- Done:", done)
        total_reward += reward
        env.render()
        env.task.state2image(env.get_observation(), root_dir=project_root + '/environment/').show()
        if done or truncated:
            print("Episode done")
            break
    print("Total Reward:", total_reward)

    reset_obs = env.reset()
    env.render()
