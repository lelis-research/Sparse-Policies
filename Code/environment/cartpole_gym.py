import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class LastActionObservationWrapper(gym.Wrapper):
    """
    Gym wrapper that augments the CartPole observation by appending a one-hot encoding of the last action.
    
    The one-hot vector has a length equal to (number of actions + 1). 
    At the beginning (i.e. before any action is taken), the vector is set so that the first element is 1.
    After an action is taken, the one-hot vector is updated such that the index (action + 1) is set to 1.
    """
    def __init__(self, env):
        super(LastActionObservationWrapper, self).__init__(env)
        self.last_action = -1
        
        # if not isinstance(env.action_space, gym.spaces.Discrete):
        #     raise ValueError("LastActionObservationWrapper only supports environments with a discrete action space.")
        
        self.num_actions = env.action_space.n + 1   # The one-hot vector will have length: (n_actions + 1)
        
        # Modify the observation space: new obs = [original observation, one-hot vector]
        original_obs_space = env.observation_space
        if isinstance(original_obs_space, Box):
            new_low = np.concatenate([original_obs_space.low, np.zeros(self.num_actions, dtype=np.float32)])
            new_high = np.concatenate([original_obs_space.high, np.ones(self.num_actions, dtype=np.float32)])
            self.observation_space = Box(low=new_low, high=new_high, dtype=np.float32)
        else:
            raise ValueError("Unsupported observation space type for LastActionObservationWrapper.")

    def reset(self, **kwargs):
        """
        Reset the environment and the last action indicator.
        Returns the augmented initial observation.
        """
        self.last_action = -1
        obs, info = self.env.reset(**kwargs)
        return self._augment_observation(obs), info
    
    def step(self, action):
        """
        Execute the given action, store it as the last action, and return the augmented observation.
        """
        self.last_action = action 
        obs, reward, terminated, truncated, info = self.env.step(action)
        # done = terminated or truncated
        return self._augment_observation(obs), reward, terminated, truncated, info
    
    def _augment_observation(self, obs):
        """
        Create a one-hot encoding for the last action and concatenate it with the original observation.
        """
        one_hot = np.zeros(self.num_actions, dtype=np.float32)
        if self.last_action is None or self.last_action == -1:
            one_hot[0] = 1.0    # No action taken yet: set the first element to 1
        else:
            one_hot[self.last_action + 1] = 1.0     # Set the (last_action + 1) index to 1

        augmented_obs = np.concatenate([obs, one_hot])
        return augmented_obs

if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    wrapped_env = LastActionObservationWrapper(env)
    
    obs, info = wrapped_env.reset()
    print("Initial augmented observation:", obs, info)
    
    for i in range(5):
        action = wrapped_env.action_space.sample()  # sample a random action
        print("Step:", i, "action:", action)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"terminated: {terminated}, truncated: {truncated}, reward: {reward}, info: {info}")
