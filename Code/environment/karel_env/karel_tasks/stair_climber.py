import numpy as np
from scipy import spatial

from environment.karel_env.base import BaseTask
from environment.karel_env.karel import KarelEnvironment

class StairClimber(BaseTask):
    def __init__(self, env_args, seed=None, reward_diff=False, rescale_reward=True):
        super().__init__(env_args, seed=seed)
        self.reward_diff = reward_diff
        self.rescale_reward = rescale_reward
        self.done = False
        self.prev_pos_reward = None
        self.sparse_reward = False
    
    def generate_initial_environment(self, env_args):
        self.done = False
        self.prev_pos_reward = None  # Reset previous reward
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]
        self.env_height = env_height  # Needed for rescaling
        self.env_width = env_width    # Needed for rescaling
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        # Set up the walls
        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True
        
        # Build the stairs
        for i in range(1, env_width - 2):
            state[4, env_height - i - 1, i + 1] = True
            state[4, env_height - i - 1, i + 2] = True
        
        on_stair_positions = [
            [env_height - i - 1, i] for i in range(1, env_width - 1)
        ]
        
        one_block_above_stair_positions = [
            [env_height - i - 2, i] for i in range(1, env_width - 2)
        ]
        
        # One cell above the stairs
        self.valid_positions = on_stair_positions + one_block_above_stair_positions
        
        # Initial position has to be on stair but cannot be on last step
        initial_position_index = self.rng.randint(0, len(on_stair_positions) - 1)
        
        # Marker has to be after initial position
        marker_position_index = self.rng.randint(initial_position_index + 1, len(on_stair_positions))
        
        self.initial_position = on_stair_positions[initial_position_index]
        state[1, self.initial_position[0], self.initial_position[1]] = True
        
        self.marker_position = on_stair_positions[marker_position_index]
        state[5, :, :] = True
        state[6, self.marker_position[0], self.marker_position[1]] = True   # Place marker
        state[5, self.marker_position[0], self.marker_position[1]] = False  # Ensure only one marker
        
        self.initial_distance = abs(self.initial_position[0] - self.marker_position[0]) \
            + abs(self.initial_position[1] - self.marker_position[1])
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        # self.previous_distance = self.initial_distance
        self.prev_pos_reward = None  # Reset previous reward
        self.done = False
        
    def get_reward(self, env: KarelEnvironment):
        # terminated = False
        done = False
        reward = 0.

        karel_pos = env.get_hero_pos()  # (row, col, direction)
        agent_pos = (karel_pos[0], karel_pos[1])
        
        # current_distance = abs(karel_pos[0] - self.marker_position[0]) \
        #     + abs(karel_pos[1] - self.marker_position[1])
        
        # Calculate current distance (negative Manhattan distance)
        current_distance = -1 * spatial.distance.cityblock(agent_pos, self.marker_position)

        # For the first step, initialize previous_reward
        if self.prev_pos_reward is None:
            self.prev_pos_reward = current_distance
        
        # # Reward is how much closer Karel is to the marker, normalized by the initial distance
        # reward = (self.previous_distance - current_distance) / self.initial_distance
        
        # if [karel_pos[0], karel_pos[1]] not in self.valid_positions:
        #     reward = self.crash_penalty
        #     print("** not valid position")
        #     terminated = True
            
        # if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
        #     print("** marker position")
        #     terminated = True
        
        # self.previous_distance = current_distance


        if not self.reward_diff:
            if self.rescale_reward:
                # Rescale reward to range between -1 and 0
                from_min = - (self.env_height + self.env_width)  # Negative sum of dimensions
                from_max, to_min, to_max = 0, -1, 0
                reward = ((current_distance - from_min) * (to_max - to_min) / (from_max - from_min)) + to_min
            else:
                reward = current_distance

            if [agent_pos[0], agent_pos[1]] not in self.valid_positions:
                reward = -1.0
                # print("** not valid position")
                # done = True

            if current_distance == 0:
                # Agent has reached the marker position
                print("** Agent reached the goal!!!!")
                done = True
        else:
            if [agent_pos[0], agent_pos[1]] not in self.valid_positions:
                reward = self.prev_pos_reward - 1.0
                # print("** not valid position")
                # done = True
            else:
                reward = current_distance - self.prev_pos_reward

            self.prev_pos_reward = current_distance

            if current_distance == 0:
                # Agent has reached the marker position
                print("** Agent reached the goal!!!!")
                done = True

        # print("reward dense: ", reward)

        reward = float(done) if self.sparse_reward else reward

        # Adjust reward for sparse or non-sparse version
        if self.sparse_reward:
            # reward = reward if done and not self.done else 0.0
            reward = 0.0 if done and not self.done else -1.0
            # print("reward sparse: ", reward)

        self.done = self.done or done
        return done, reward


class StairClimberSparse(StairClimber):
    def __init__(self, env_args, seed=None, reward_diff=False, rescale_reward=True):
        super().__init__(env_args, seed=seed, reward_diff=reward_diff, rescale_reward=rescale_reward)
        self.sparse_reward = True  # Enable sparse reward

    def get_reward(self, env: KarelEnvironment):
        # Use the same logic but with sparse reward
        return super().get_reward(env)
    
    # def get_reward(self, env: KarelEnvironment):
    #     terminated = False
    #     reward = 0.

    #     karel_pos = env.get_hero_pos()
        
    #     if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
    #         reward = 1.
    #         terminated = True
    #     elif [karel_pos[0], karel_pos[1]] not in self.valid_positions:
    #         reward = self.crash_penalty
    #         terminated = True
    #         print("** not valid position")
        
    #     return terminated, reward
    

class StairClimberAllInit(BaseTask):
    def __init__(self, env_args, seed=None, reward_diff=False, rescale_reward=True):
        super().__init__(env_args, seed=seed)
        self.reward_diff = reward_diff
        self.rescale_reward = rescale_reward
        self.done = False
        self.prev_pos_reward = None
        self.sparse_reward = False
        self.all_initial_confs = self._generate_all_initial_confs(env_args)

    def _generate_all_initial_confs(self, env_args):
        reference_env = KarelEnvironment(**env_args)
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]
        self.env_height = env_height
        self.env_width = env_width
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        # Set up the walls
        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True
        
        # Build the stairs
        for i in range(1, env_width - 2):
            state[4, env_height - i - 1, i + 1] = True
            state[4, env_height - i - 1, i + 2] = True
        
        on_stair_positions = [
            [env_height - i - 1, i] for i in range(1, env_width - 1)
        ]
        
        one_block_above_stair_positions = [
            [env_height - i - 2, i] for i in range(1, env_width - 2)
        ]
        
        # One cell above the stairs
        self.valid_positions = on_stair_positions + one_block_above_stair_positions
        
        all_confs = []
        for initial_position_index in range(len(on_stair_positions) - 1):
            for marker_position_index in range(initial_position_index + 1, len(on_stair_positions)):
                initial_position = on_stair_positions[initial_position_index]
                marker_position = on_stair_positions[marker_position_index]
                
                conf_state = state.copy()
                conf_state[1, initial_position[0], initial_position[1]] = True
                conf_state[5, :, :] = True
                conf_state[6, marker_position[0], marker_position[1]] = True
                conf_state[5, marker_position[0], marker_position[1]] = False
                
                all_confs.append(conf_state)

        # print(f"Generated {len(all_confs)} initial configurations")
        return all_confs

    def generate_initial_environment(self, env_args):
        if 'initial_state' in env_args and env_args['initial_state'] is not None:
            initial_state = env_args['initial_state']
            marker_pos = np.argwhere(initial_state[6, :, :])[0] # Extract marker position from the provided configuration
            self.marker_position = (marker_pos[0], marker_pos[1])
            return KarelEnvironment(**env_args)
        else:
            return KarelEnvironment(**env_args)
        
    def reset_environment(self):
        super().reset_environment()
        # self.previous_distance = self.initial_distance
        self.prev_pos_reward = None  # Reset previous reward
        self.done = False
        
    def get_reward(self, env: KarelEnvironment):
        # terminated = False
        done = False
        reward = 0.

        karel_pos = env.get_hero_pos()  # (row, col, direction)
        agent_pos = (karel_pos[0], karel_pos[1])
        
        # current_distance = abs(karel_pos[0] - self.marker_position[0]) \
        #     + abs(karel_pos[1] - self.marker_position[1])
        
        # Calculate current distance (negative Manhattan distance)
        current_distance = -1 * spatial.distance.cityblock(agent_pos, self.marker_position)     #TODO: check this

        # For the first step, initialize previous_reward
        if self.prev_pos_reward is None:
            self.prev_pos_reward = current_distance
        
        # # Reward is how much closer Karel is to the marker, normalized by the initial distance
        # reward = (self.previous_distance - current_distance) / self.initial_distance
        
        # if [karel_pos[0], karel_pos[1]] not in self.valid_positions:
        #     reward = self.crash_penalty
        #     print("** not valid position")
        #     terminated = True
            
        # if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
        #     print("** marker position")
        #     terminated = True
        
        # self.previous_distance = current_distance


        if not self.reward_diff:
            if self.rescale_reward:
                # Rescale reward to range between -1 and 0
                from_min = - (self.env_height + self.env_width)  # Negative sum of dimensions
                from_max, to_min, to_max = 0, -1, 0
                reward = ((current_distance - from_min) * (to_max - to_min) / (from_max - from_min)) + to_min
            else:
                reward = current_distance

            if [agent_pos[0], agent_pos[1]] not in self.valid_positions:
                reward = -1.0
                # print("** not valid position")
                # done = True

            if current_distance == 0:
                # Agent has reached the marker position
                print("** Agent reached the goal!!!!")
                done = True
        else:
            if [agent_pos[0], agent_pos[1]] not in self.valid_positions:
                reward = self.prev_pos_reward - 1.0
                # print("** not valid position")
                # done = True
            else:
                reward = current_distance - self.prev_pos_reward

            self.prev_pos_reward = current_distance

            if current_distance == 0:
                # Agent has reached the marker position
                print("** Agent reached the goal!!!!")
                done = True

        # print("reward dense: ", reward)

        reward = float(done) if self.sparse_reward else reward

        # Adjust reward for sparse or non-sparse version
        if self.sparse_reward:
            # reward = reward if done and not self.done else 0.0
            reward = 0.0 if done and not self.done else -1.0
            # print("reward sparse: ", reward)

        self.done = self.done or done
        return done, reward
    

class StairClimberSparseAllInit(StairClimberAllInit):
    def __init__(self, env_args, seed=None, reward_diff=False, rescale_reward=True):
        super().__init__(env_args, seed=seed, reward_diff=reward_diff, rescale_reward=rescale_reward)
        self.sparse_reward = True  # Enable sparse reward

    def get_reward(self, env: KarelEnvironment):
        # Use the same logic but with sparse reward
        return super().get_reward(env)