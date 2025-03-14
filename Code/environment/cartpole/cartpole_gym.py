import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


class LastActionObservationWrapper(gym.Wrapper):
    """
    Gym wrapper that augments the CartPole observation by appending a one-hot encoding of the last action.
    
    The one-hot vector has a length equal to (number of actions + 1). 
    At the beginning (i.e. before any action is taken), the vector is set so that the first element is 1.
    After an action is taken, the one-hot vector is updated such that the index (action + 1) is set to 1.
    """
    def __init__(self, env, train_mode=True, last_action_in_obs=False):
        super(LastActionObservationWrapper, self).__init__(env)
        self.last_action = -1
        self.train_mode = train_mode
        self.last_action_in_obs = last_action_in_obs
        self.num_actions = env.action_space.n + 1   # The one-hot vector will have length: (n_actions + 1)
        
        if self.last_action_in_obs:
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
        # print("max steps before: ", self.env.spec.max_episode_steps)
        if self.train_mode:
            self.env.unwrapped.length = 0.5
        else:
            self.env.unwrapped.length = 1.0

        self.last_action = -1
        obs, info = self.env.reset(**kwargs)

        if self.last_action_in_obs:
            return self._augment_observation(obs), info
        else:
            return obs, info
    
    def step(self, action):
        """
        Execute the given action, store it as the last action, and return the augmented observation.
        """
        self.last_action = action 
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.last_action_in_obs:
            return self._augment_observation(obs), reward, terminated, truncated, info
        else:
            return obs, reward, terminated, truncated, info
    
    def _augment_observation(self, obs):
        """
        Create a one-hot encoding for the last action and concatenate it with the original observation.
        """
        one_hot = np.zeros(self.num_actions, dtype=np.float32)
        if self.last_action_in_obs:
            if self.last_action is None or self.last_action == -1:
                one_hot[0] = 1.0    # No action taken yet: set the first element to 1
            else:
                one_hot[self.last_action + 1] = 1.0     # Set the (last_action + 1) index to 1

        augmented_obs = np.concatenate([obs, one_hot])
        return augmented_obs



class CustomForceWrapper(LastActionObservationWrapper):
    """
    Wrapper that modifies the force magnitude based on the action (0 or 1).
    - Action 0 applies a force of -3.3 (left)
    - Action 1 applies a force of 3.98 (right)
    This overrides the default force_mag in CartPoleEnv.
    """
    def __init__(self, env, train_mode=True, last_action_in_obs=False):
        super().__init__(env, train_mode=train_mode, last_action_in_obs=last_action_in_obs)
        
    def step(self, action):
        if action == 0:
            self.env.unwrapped.force_mag = 3.3  # Action 0: force = -3.3
        elif action == 1:
            self.env.unwrapped.force_mag = 3.98  # Action 1: force = 3.98
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0 or 1.")
        
        return super().step(action)



class EasyCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}
    
    def __init__(self, render_mode=None, max_episode_steps=500):
        super().__init__()
        print("== EasyCartPoleEnv init")
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.tau = 0.01  # 0.01s timestep (500 steps = 5 seconds)
        
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # Half pole length
        self.force_mag = 10.0
        self.x_threshold = 2.0  # Cart position limits
        self.theta_threshold = 0.21  # Angle limits
        
        self.action_space = Box(-np.inf, np.inf, (1,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        # State variables
        self.state = None
        self.steps = 0
        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            self._rand(-0.05, 0.05),
            self._rand(-0.05, 0.05),
            self._rand(-0.05, 0.05),
            self._rand(-0.05, 0.05)
        ])
        self.steps = 0
        
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        return self.state

    def step(self, action):
        # Convert action to continuous force
        force = np.clip(action[0] * 2.0, -self.force_mag, self.force_mag)
        # print("== action: ", force)
        
        x, v, theta, omega = self.state
        # print("== state: ", self.state)
        
        # Physics calculations from original simulate()
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.mass_pole*self.length*omega**2*sintheta)/self.total_mass
        theta_acc = (self.gravity*sintheta - costheta*temp)/(self.length*(4.0/3.0 - (self.mass_pole*costheta**2)/self.total_mass))
        x_acc = temp - (self.mass_pole*self.length*theta_acc*costheta)/self.total_mass
        
        # Euler integration
        x = x + v * self.tau
        v = v + x_acc * self.tau
        theta = theta + omega * self.tau
        omega = omega + theta_acc * self.tau
        
        # Wrap cart position when out of bounds
        if x < -self.x_threshold:
            x += 2 * self.x_threshold
        elif x > self.x_threshold:
            x -= 2 * self.x_threshold
            
        self.state = np.array([x, v, theta, omega], dtype=np.float32)
        self.steps += 1
        
        # Calculate safety violations (original check_safe logic)
        safe_error = 0.0
        if theta > self.theta_threshold:
            safe_error += theta - self.theta_threshold
        elif theta < -self.theta_threshold:
            safe_error += -self.theta_threshold - theta
        
        # Reward and termination
        reward = 1.0  # not used in their version
        terminated = False  # Never terminate early based on state
        truncated = self.steps >= self.max_episode_steps
        
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -safe_error, terminated, truncated, {}

    def render(self):
        screen_width = 600
        screen_height = 400
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            # Cart
            cart_width, cart_height = 50.0, 30.0
            cart = rendering.FilledPolygon([
                (-cart_width/2, -cart_height/2),
                (-cart_width/2, cart_height/2),
                (cart_width/2, cart_height/2),
                (cart_width/2, -cart_height/2)
            ])
            self.cart_transform = rendering.Transform()
            cart.add_attr(self.cart_transform)
            self.viewer.add_geom(cart)
            
            # Pole
            pole_width = 10.0
            pole_length = screen_height * 0.4
            pole = rendering.FilledPolygon([
                (-pole_width/2, 0),
                (-pole_width/2, pole_length),
                (pole_width/2, pole_length),
                (pole_width/2, 0)
            ])
            pole.set_color(0.8, 0.6, 0.4)
            self.pole_transform = rendering.Transform(translation=(0, cart_height/4))
            pole.add_attr(self.pole_transform)
            pole.add_attr(self.cart_transform)
            self.viewer.add_geom(pole)
            
            # Track
            self.track = rendering.Line((0, 100), (screen_width, 100))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        # Update positions
        x, _, theta, _ = self.state
        scale = screen_width / (self.x_threshold * 2)  # 4.0 total width
        cart_x = x * scale + screen_width/2.0
        self.cart_transform.set_translation(cart_x, 100)
        self.pole_transform.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=self.render_mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def _rand(self, a, b):
        if b < a:
            b = a
        return (b - a) * self.np_random.random() + a



def make_cartpole_env(train_mode=True, last_action_in_obs=True, 
                      max_episode_steps=250, easy_cartpole=False):
    def thunk():
        if easy_cartpole:
            env = EasyCartPoleEnv(max_episode_steps=max_episode_steps)  # should be 500 for 5s of training
        else:
            env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
            env = LastActionObservationWrapper(env, 
                                            train_mode=train_mode, 
                                            last_action_in_obs=last_action_in_obs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


if __name__ == "__main__":

    # Original CartPole
    # env = gym.make("CartPole-v1")
    # wrapped_env = LastActionObservationWrapper(env)

    # Easy CartPole
    wrapped_env = EasyCartPoleEnv(max_episode_steps=500)
    
    obs, info = wrapped_env.reset(seed=0)
    print("Initial augmented observation:", obs, info)
    
    for i in range(5):
        action = wrapped_env.action_space.sample()  # sample a random action
        print("Step:", i, "action:", action)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"terminated: {terminated}, truncated: {truncated}, reward: {reward}, info: {info}\n")
