import numpy as np
from itertools import combinations, chain

from environment.karel_env.base import BaseTask
from environment.karel_env.karel import KarelEnvironment


class TopOff(BaseTask):
        
    def generate_initial_environment(self, env_args):
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]    
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True
        
        state[1, env_height - 2, 1] = True
        
        self.possible_marker_locations = [
            [env_height - 2, i] for i in range(2, env_width - 1)
        ]
        
        self.rng.shuffle(self.possible_marker_locations)
        
        self.num_markers = self.rng.randint(1, len(self.possible_marker_locations))
        self.markers = self.possible_marker_locations[:self.num_markers]
        
        for marker in self.markers:
            state[6, marker[0], marker[1]] = True
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.num_previous_correct_markers = 0

    def get_reward_Tales(self, env: KarelEnvironment):
        terminated = False
        
        num_markers = env.markers_grid.sum()
        num_correct_markers = 0

        for marker in self.markers:
            if env.markers_grid[marker[0], marker[1]] == 2:
                num_correct_markers += 1
            elif env.markers_grid[marker[0], marker[1]] == 0:
                return True, self.crash_penalty
        
        reward = (num_correct_markers - self.num_previous_correct_markers) / len(self.markers)
        
        if num_markers > num_correct_markers + len(self.markers):
            terminated = True
            reward = self.crash_penalty
        
        elif num_correct_markers == len(self.markers):
            terminated = True
            
        self.num_previous_correct_markers = num_correct_markers
        
        return terminated, reward
    
    def get_reward(self, env: KarelEnvironment):
        if env.crashed:
            return True, self.crash_penalty
        
        rows = env.state_shape[1]
        cols = env.state_shape[2]
        last_row = rows - 2
        playable_cols_start = 1
        playable_cols_end = cols - 2
        max_consecutive = playable_cols_end - playable_cols_start + 1
        
        markers_grid = env.markers_grid
        agent_pos = env.get_hero_pos()
        agent_row, agent_col, _ = agent_pos
        
        consecutive = 0
        for c in range(playable_cols_start, playable_cols_end + 1):
            current_pos = (last_row, c)
            if current_pos in self.markers:
                if markers_grid[last_row, c] == 2:
                    consecutive += 1
                else:
                    break   # Stop at first missing marker
            else:
                if markers_grid[last_row, c] == 0:
                    consecutive += 1
                else:
                    break   # Stop at first extra marker
        
        bonus = 0
        if (agent_row == last_row and agent_col == playable_cols_end and
            consecutive == max_consecutive):
            bonus = 1
        
        total = consecutive + bonus
        reward = total / (max_consecutive + 1)
        done = (consecutive == max_consecutive) and (bonus == 1)
        # print(f"reward: {reward:.2f}, done: {done}")
        
        return done, reward


class TopOffSparse(TopOff):
    
    def get_reward(self, env: KarelEnvironment):
        terminated = False
        num_correct_markers = 0
        reward = 0.

        for marker in self.markers:
            if env.markers_grid[marker[0], marker[1]] == 2:
                num_correct_markers += 1
            elif env.markers_grid[marker[0], marker[1]] == 0:
                return True, self.crash_penalty
        
        num_markers = env.markers_grid.sum()
        if num_markers > num_correct_markers + len(self.markers):
            terminated = True
            reward = self.crash_penalty
        
        if num_correct_markers == len(self.markers) and env.get_hero_pos()[0] == env.state_shape[1] - 2 and env.get_hero_pos()[1] == env.state_shape[2] - 2:
            terminated = True
            reward = 1.
        
        return terminated, reward
    

class TopOffAllInit(BaseTask):
    def __init__(self, env_args):
        super().__init__(env_args)
        self.all_initial_confs = self._generate_all_initial_confs(env_args)
        self.env_height = None
        self.env_width = None

    def _generate_all_initial_confs(self, env_args):
        reference_env = KarelEnvironment(**env_args)
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]

        state = np.zeros(reference_env.state_shape, dtype=bool)
        state[4, :, 0] = True  # Left wall
        state[4, :, env_width - 1] = True  # Right wall
        state[4, 0, :] = True  # Top wall
        state[4, env_height - 1, :] = True  # Bottom wall
        state[1, env_height - 2, 1] = True  # Karel's initial position

        possible_marker_locations = [
            [env_height - 2, c] for c in range(2, env_width - 1)
        ]

        def all_non_empty_subsets(s):
            return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

        all_confs = []
        for subset in all_non_empty_subsets(possible_marker_locations):
            conf_state = state.copy()
            for (row, col) in subset:
                conf_state[6, row, col] = True
            all_confs.append(conf_state)

        return all_confs

    def generate_initial_environment(self, env_args):
        if 'initial_state' in env_args and env_args['initial_state'] is not None:
            initial_state = env_args['initial_state']
            env_height = initial_state.shape[1]
            last_row = env_height - 2
            marker_cols = np.where(initial_state[6, last_row, :])[0].tolist()
            self.markers = [[last_row, c] for c in marker_cols]
            self.num_previous_correct_markers = 0
            return KarelEnvironment(**env_args)
        else:
            return KarelEnvironment(**env_args)

    def get_reward(self, env: KarelEnvironment):
        if env.crashed:
            return True, self.crash_penalty
        
        rows = env.state_shape[1]
        cols = env.state_shape[2]
        last_row = rows - 2
        playable_cols_start = 1
        playable_cols_end = cols - 2
        max_consecutive = playable_cols_end - playable_cols_start + 1
        
        markers_grid = env.markers_grid
        agent_pos = env.get_hero_pos()
        agent_row, agent_col, _ = agent_pos
        
        consecutive = 0
        for c in range(playable_cols_start, playable_cols_end + 1):
            current_pos = (last_row, c)
            if current_pos in self.markers:
                if markers_grid[last_row, c] >= 1:
                    consecutive += 1
                else:
                    break
            else:
                if markers_grid[last_row, c] == 0:
                    consecutive += 1
                else:
                    break
        
        bonus = 0
        if (agent_row == last_row and agent_col == playable_cols_end and
            consecutive == max_consecutive):
            bonus = 1
        
        total = consecutive + bonus
        reward = total / (max_consecutive + 1)
        done = (consecutive == max_consecutive) and (bonus == 1)
        
        return done, reward


class TopOffSparseAllInit(TopOffAllInit):

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        num_correct_markers = 0
        reward = 0.

        for marker in self.markers:
            if env.markers_grid[marker[0], marker[1]] == 2:
                num_correct_markers += 1
            elif env.markers_grid[marker[0], marker[1]] == 0:
                return True, self.crash_penalty
        
        num_markers = env.markers_grid.sum()
        if num_markers > num_correct_markers + len(self.markers):
            terminated = True
            reward = self.crash_penalty
        
        if num_correct_markers == len(self.markers) and env.get_hero_pos()[0] == env.state_shape[1] - 2 and env.get_hero_pos()[1] == env.state_shape[2] - 2:
            terminated = True
            reward = 1.
        
        return terminated, reward