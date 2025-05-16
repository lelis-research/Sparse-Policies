import gym
from gym import spaces
import numpy as np
from karel import Karel_world
from generator import KarelStateGenerator

class KarelGymEnv(gym.Env):
    """
    Gym environment wrapper for the Karel_world class.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config=None):
        super(KarelGymEnv, self).__init__()
        self.config = config or {}
        self.width = self.config.get('width', 8)
        self.height = self.config.get('height', 8)
        self.max_steps = self.config.get('max_steps', 100)
        self.current_step = 0

        # Initialize Karel world and state generator
        self.state_generator = KarelStateGenerator()
        self.karel_world = Karel_world(
            make_error=False,
            env_task=self.config.get('task', 'program'),  # Default task is 'program'
            task_definition=self.config.get('task_definition', 'program'),
            reward_diff=self.config.get('reward_diff', False),
            final_reward_scale=self.config.get('final_reward_scale', True)
        )

        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)  # 5 possible actions
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.height, self.width, 16),
            dtype=np.bool_
        )

        # Initialize the environment
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        # Convert action to one-hot encoding
        action_one_hot = np.zeros(5, dtype=bool)
        action_one_hot[action] = True

        # Perform the action in the Karel world
        try:
            self.karel_world.state_transition(action_one_hot)
            reward = self._compute_reward()
            done = self._check_done()
        except RuntimeError as e:
            # Handle invalid actions or errors
            print("Invalid action: ", e)
            done = True

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        observation = self._get_observation()
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.current_step = 0
        # Generate a new initial state for a specific task
        task = self.config.get('task', 'program')
        if task == 'harvester':
            initial_state, _, _, _, metadata = self.state_generator.generate_single_state_harvester(
                h=self.height,
                w=self.width
            )
            self.karel_world.set_new_state(initial_state, metadata=metadata)
        elif task == 'stairClimber':
            initial_state, _, _, _, metadata = self.state_generator.generate_single_state_stair_climber(
                h=self.height,
                w=self.width
            )
            self.karel_world.set_new_state(initial_state, metadata=metadata)
        elif task == 'cleanHouse':
            initial_state, _, _, _, metadata = self.state_generator.generate_single_state_clean_house(
                h=self.height,
                w=self.width
            )
            self.karel_world.set_new_state(initial_state, metadata=metadata)
        else:
            # Default task or random state
            initial_state, _, _, _, _ = self.state_generator.generate_single_state(
                h=self.height,
                w=self.width,
                wall_prob=0.1
            )
            self.karel_world.set_new_state(initial_state)
        return self._get_observation()

    def render(self, mode='human'):
        if mode == 'human':
            self.karel_world.print_state()
        elif mode == 'rgb_array':
            return self.karel_world.state2image(self.karel_world.s)
        else:
            super(KarelGymEnv, self).render(mode=mode)  # Just raise an exception

    def _get_observation(self):
        # Return the current state as the observation
        return self.karel_world.s.astype(np.float32)
    
    def _compute_reward(self):
        # Define your reward function based on the task
        # This is a placeholder implementation; adjust it as needed
        if self.config.get('task') == 'harvester':
            # Implement reward logic for 'harvester' task
            reward, _ = self.karel_world._get_harvester_task_reward(self.karel_world.get_location())
        elif self.config.get('task') == 'stairClimber':
            reward, _ = self.karel_world._get_stairClimber_task_reward(self.karel_world.get_location())
        elif self.config.get('task') == 'cleanHouse':
            reward, _ = self.karel_world._get_cleanHouse_task_reward(self.karel_world.get_location())
        else:
            # Default reward (e.g., -0.1 per step)
            reward = -0.1
        return reward
    
    def _check_done(self):
        # Check if the task is completed
        if self.config.get('task') == 'harvester':
            _, done = self.karel_world._get_harvester_task_reward(self.karel_world.get_location())
        elif self.config.get('task') == 'stairClimber':
            _, done = self.karel_world._get_stairClimber_task_reward(self.karel_world.get_location())
        elif self.config.get('task') == 'cleanHouse':
            _, done = self.karel_world._get_cleanHouse_task_reward(self.karel_world.get_location())
        else:
            # Default done condition
            done = False
        return done
    

def make_karel_env(config=None):
    """
    Factory function to create a KarelGymEnv instance with the given configuration.
    """
    def _init():
        env = KarelGymEnv(config)
        return env
    return _init


if __name__ == "__main__":
    env_config = {
        'width': 8,
        'height': 8,
        'max_steps': 100,
        'task': 'harvester'
    }
    env = make_karel_env(config=env_config)()
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()  # Random action
        print("Action:", action)
        obs, reward, done, info = env.step(action)
        print("Reward:", reward)
        total_reward += reward
        env.render()
    print("Total Reward:", total_reward)


