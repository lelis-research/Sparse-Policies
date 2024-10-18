import gymnasium as gym
import numpy as np
import torch
from environment.combogrid import Game, basic_actions
from typing import List, Any
from gymnasium.envs.registration import register
import copy, random

class ComboGym(gym.Env):
    def __init__(self, rows=3, columns=3, problem="TL-BR", options=None):
        self._game = Game(rows, columns, problem)
        self._rows = rows
        self._columns = columns
        self._problem = problem
        self.render_mode = None
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self._game.get_observation()), ), dtype=np.float64)
        self.n_steps = 0
        
        if options is not None:
            self.setup_options(options)
        else:
            self.action_space = gym.spaces.Discrete(len(self._game.get_actions()))
            self.program_stack = None
            self.option_index = None

    def get_observation(self):
        return self._game.get_observation()
    
    def setup_options(self, options:List[Any]=None):
        """
        Enables the corresponding agents to choose from both actions and options
        """
        self.option_index = len(self._game.get_actions())
        self.program_stack = [basic_actions(i) for i in range(self.option_index)] + options
        self.action_space = gym.spaces.Discrete(len(self.program_stack))
        self.option_sizes = [3 for _ in range(len(options))]
    
    def reset(self, init_loc=None, seed=0, options=None):
        self._game.reset(init_loc)
        self.n_steps = 0
        return self.get_observation(), {}
    
    def step(self, action:int):
        truncated = False
        def process_action(action: int):
            nonlocal truncated
            self._game.apply_action(action)
            self.n_steps += 1
            terminated = self._game.is_over()
            reward = 0 if terminated else -1 
            if self.n_steps == 500:
                truncated = True
            return self.get_observation(), reward, terminated, truncated, {}
        

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
                if greedy:
                    a = torch.argmax(prob_actions).item()
                else:
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
                print(f"Stopping probability: {stopping_prob}")
            return stopping_prob <= 0.5

        # Execute the option
        if self.option_index and action >= self.option_index:
            reward_sum = 0
            option = self.program_stack[action]
            verbose = True
            # trajectory = Trajectory()
            sequence_ended = False  # Flag to indicate the end of a sequence
            terminated, truncated = False, False

            if verbose: print('Beginning Option')

            current_length = 0
            max_length = 20
            while not sequence_ended and not terminated and not truncated:

                # Choose action using model_y1
                a = choose_action(self._game, option.model_y1, greedy=False, verbose=verbose)

                if verbose: 
                    # print(self._game, a, "\n")
                    print("action: ", a)

                # Check stopping condition using model_y2
                if check_stopping(self._game, option.model_y2, verbose):
                    if verbose: print("Stopping the current sequence based on model_y2.")
                    sequence_ended = True  # End the current sequence, but continue the outer loop
                    
                # Apply the chosen action
                # self._game.apply_action(a)
                obs, reward, terminated, truncated, _ = process_action(a)
                reward_sum += reward

                current_length += 1
                if current_length >= max_length:
                    sequence_ended = True

            return obs, reward_sum, terminated, truncated, {}

            # # Execute the option with masks
            # for _ in range(self.option_sizes[action - self.option_index]):
            #     option_action = self.program_stack[action].get_action_with_mask(torch.tensor(self.get_observation(), dtype=torch.float32).view(1, -1))
            #     obs, reward, terminated, truncated, _ = process_action(option_action)
            #     reward_sum += reward
            #     if terminated or truncated:
            #         return obs, reward_sum, terminated, truncated, {}
            # return obs, reward_sum, terminated, truncated, {}
        else:
            return process_action(action)
    
    def is_over(self, loc=None):
        if loc:
            return loc == self._game.problem.goal
        return self._game.is_over()
        
    def get_observation_space(self):
        return self._rows * self._columns * 2 + 9
    
    def get_action_space(self):
        return self.action_space.n
    
    def represent_options(self, options):
        return self._game.represent_options(options)
    

def make_env(*args, **kwargs):
    def thunk():
        env = ComboGym(*args, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


register(
     id="ComboGridWorld-v0",
     entry_point=ComboGym
)