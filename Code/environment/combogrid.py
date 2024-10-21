import random
import numpy as np
import math
import gc

class Problem:
    def __init__(self, rows, columns, problem_str):
        self.rows = rows
        self.columns = columns
        self.initial, self.goal = self._parse_problem(problem_str)

    def _parse_problem(self, problem_str):
        initial = self._parse_position(problem_str[:2])
        print(f"Initial position: {initial}")
        goal = self._parse_position(problem_str[3:])
        print(f"Goal position: {goal}")
        return initial, goal

    def _parse_position(self, pos_str):
        if pos_str[0] == 'T':
            row = 0
        elif pos_str[0] == 'B':
            row = self.rows - 1
        elif pos_str[0] == 'M':
            row = math.floor(self.rows / 2)
        else:
            raise ValueError("Invalid row specifier. Use 'T' for top, 'B' for bottom, or 'M' for middle.")
        
        if pos_str[1] == 'L':
            col = 0
        elif pos_str[1] == 'R':
            col = self.columns - 1
        elif pos_str[1] == 'M':
            col = math.floor(self.columns / 2)
        else:
            raise ValueError("Invalid column specifier. Use 'L' for left, 'R' for right, or 'M' for middle.")
        
        return (row, col)


class Game:
    """
    The (0, 0) in the matrices show top and left and it goes to the bottom and right as 
    the indices increases.
    """
    def __init__(self, rows, columns, problem_str, init_x=None, init_y=None):
        self._rows = rows
        self._columns = columns

        self.problem = Problem(rows, columns, problem_str)
        
        if init_x and init_y:
            self.reset((init_x, init_y))
        else:
            self.reset()

        self._matrix_structure = np.zeros((rows, columns))
        self._matrix_goal = np.zeros((rows, columns))        

        goal = self.problem.goal
        
        self._matrix_goal[goal[0]][goal[1]] = 1

        # state of current action sequence
        """
        Mapping used: 
        0, 0, 1 -> up (0)
        0, 1, 2 -> down (1)
        2, 1, 0 -> left (2)
        1, 0, 2 -> right (3)
        """
        self._pattern_length = 3

        self._action_pattern = {}
        self._action_pattern[(0, 0, 1)] = 0
        self._action_pattern[(0, 1, 2)] = 1
        self._action_pattern[(2, 1, 0)] = 2
        self._action_pattern[(1, 0, 2)] = 3

    def reset(self, init_loc=None):
        self._matrix_unit = np.zeros((self._rows, self._columns))
        initial = self.problem.initial
        self._x, self._y = init_loc if init_loc else initial
        self._matrix_unit[self._x][self._y] = 1
        self._state = []
        gc.collect()

    def __repr__(self) -> str:
        str_map = ""
        for i in range(self._rows):
            for j in range(self._columns):
                if self._matrix_unit[i][j] == 1:
                    str_map += " A "
                elif self._matrix_structure[i][j] == 1:
                     str_map += " B "
                elif self._matrix_goal[i][j] == 1:
                     str_map += " G "
                else: 
                     str_map += " 0 "
            str_map += "\n"
        return str_map
    
    def represent_options(self, options: dict) -> str:
        str_map = ""
        option_letters = "UDLR"
        for i in range(self._rows):
            for j in range(self._columns):
                if self._matrix_structure[i][j] == 1:
                     str_map += " B "
                elif self._matrix_goal[i][j] == 1:
                     str_map += " G "
                elif (i,j) in options and tuple(options[(i,j)]) in self._action_pattern:
                    str_map += f" {option_letters[self._action_pattern[tuple(options[(i,j)])]]} "
                elif self._matrix_unit[i][j] == 1:
                    str_map += " A "
                else: 
                     str_map += " 0 "
            str_map += "\n"
        return str_map
    
    def get_observation(self):
        one_hot_matrix_state = np.zeros((self._pattern_length, self._pattern_length), dtype=int)
        for i, v in enumerate(self._state):
            one_hot_matrix_state[v][i] = 1
        return np.concatenate((self._matrix_unit.ravel(), one_hot_matrix_state.ravel(), self._matrix_goal.ravel()))
       
    def is_over(self):
        return self._matrix_goal[self._x][self._y] == 1
    
    def get_actions(self):
        return [0, 1, 2]
   
    def apply_action(self, action):
        """
        Mapping used: 
        0, 0, 1 -> up (0)
        0, 1, 2 -> down (1)
        2, 1, 0 -> left (2)
        1, 0, 2 -> right (3)
        """
        # each column in _state_matrix represents an action
        self._state.append(action)

        if len(self._state) == self._pattern_length:
            action_tuple = tuple(self._state)
            if action_tuple in self._action_pattern:
                # moving up
                if self._action_pattern[action_tuple] == 0:
                    if self._x - 1 >= 0 and self._matrix_structure[self._x - 1][self._y] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._x -= 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving down
                if self._action_pattern[action_tuple] == 1:
                    if self._x + 1 < self._matrix_unit.shape[0] and self._matrix_structure[self._x + 1][self._y] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._x += 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving left
                if self._action_pattern[action_tuple] == 2:
                    if self._y - 1 >= 0 and self._matrix_structure[self._x][self._y - 1] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._y -= 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving right
                if self._action_pattern[action_tuple] == 3:
                    if self._y + 1 < self._matrix_unit.shape[1] and self._matrix_structure[self._x][self._y + 1] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._y += 1
                        self._matrix_unit[self._x][self._y] = 1
            self._state = []

class basic_actions:
    def __init__(self, action):
        self.action = action

    def predict(self, x):
        return self.action

    def predict_hierarchical(self, x, epsilon):
        return self.predict(x)
    

class TestGame:
    """
    The (0, 0) in the matrices show top and left and it goes to the bottom and right as 
    the indices increases.

    In this game, we have 4 goals, and the agent in the middle of the grid has to reach all of them.
    """
    def __init__(self, rows, columns, init_x=None, init_y=None):
        self._rows = rows
        self._columns = columns
        self._problems = []

        for prob in ["MM-TM", "MM-BM", "MM-MR", "MM-ML"]:
            print(f"Adding problem: {prob}")
            self._problems.append(Problem(rows, columns, prob))
        
        if init_x and init_y:
            self.reset((init_x, init_y))
        else:
            self.reset()

        self._matrix_structure = np.zeros((rows, columns))
        self._matrix_goal = np.zeros((rows, columns))        

        for problem in self._problems:
            goal = problem.goal
            self._matrix_goal[goal[0]][goal[1]] = 1

        # for handling number of goals the agent has reached
        self.goals_reached = 0
        self.total_goals = len(self._problems)

        # state of current action sequence
        """
        Mapping used: 
        0, 0, 1 -> up (0)
        0, 1, 2 -> down (1)
        2, 1, 0 -> left (2)
        1, 0, 2 -> right (3)
        """
        self._pattern_length = 3

        self._action_pattern = {}
        self._action_pattern[(0, 0, 1)] = 0
        self._action_pattern[(0, 1, 2)] = 1
        self._action_pattern[(2, 1, 0)] = 2
        self._action_pattern[(1, 0, 2)] = 3

    def reset(self, init_loc=None):
        self._matrix_unit = np.zeros((self._rows, self._columns))
        initial = self._problems[0].initial
        self._x, self._y = init_loc if init_loc else initial
        self._matrix_unit[self._x][self._y] = 1
        self._state = []
        self.goals_reached = 0
        
        # Reset the goal matrix
        # self._matrix_goal.fill(0)
        # for problem in self._problems:
        #     goal = problem.goal
        #     self._matrix_goal[goal[0]][goal[1]] = 1

        gc.collect()

    def __repr__(self) -> str:
        str_map = ""
        for i in range(self._rows):
            for j in range(self._columns):
                if self._matrix_unit[i][j] == 1:
                    str_map += " A "
                elif self._matrix_structure[i][j] == 1:
                     str_map += " B "
                elif self._matrix_goal[i][j] == 1:
                     str_map += " G "
                else: 
                     str_map += " 0 "
            str_map += "\n"
        return str_map
    
    def represent_options(self, options: dict) -> str:
        str_map = ""
        option_letters = "UDLR"
        for i in range(self._rows):
            for j in range(self._columns):
                if self._matrix_structure[i][j] == 1:
                     str_map += " B "
                elif self._matrix_goal[i][j] == 1:
                     str_map += " G "
                elif (i,j) in options and tuple(options[(i,j)]) in self._action_pattern:
                    str_map += f" {option_letters[self._action_pattern[tuple(options[(i,j)])]]} "
                elif self._matrix_unit[i][j] == 1:
                    str_map += " A "
                else: 
                     str_map += " 0 "
            str_map += "\n"
        return str_map
    
    def get_observation(self):
        one_hot_matrix_state = np.zeros((self._pattern_length, self._pattern_length), dtype=int)
        for i, v in enumerate(self._state):
            one_hot_matrix_state[v][i] = 1
        return np.concatenate((self._matrix_unit.ravel(), one_hot_matrix_state.ravel(), self._matrix_goal.ravel()))
       
    def is_over(self):
        return self.goals_reached == self.total_goals
    
    def get_actions(self):
        return [0, 1, 2]
   
    def apply_action(self, action):
        """
        Mapping used: 
        0, 0, 1 -> up (0)
        0, 1, 2 -> down (1)
        2, 1, 0 -> left (2)
        1, 0, 2 -> right (3)
        """
        # each column in _state_matrix represents an action
        self._state.append(action)

        if len(self._state) == self._pattern_length:
            action_tuple = tuple(self._state)
            if action_tuple in self._action_pattern:
                # moving up
                if self._action_pattern[action_tuple] == 0:
                    if self._x - 1 >= 0 and self._matrix_structure[self._x - 1][self._y] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._x -= 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving down
                if self._action_pattern[action_tuple] == 1:
                    if self._x + 1 < self._matrix_unit.shape[0] and self._matrix_structure[self._x + 1][self._y] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._x += 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving left
                if self._action_pattern[action_tuple] == 2:
                    if self._y - 1 >= 0 and self._matrix_structure[self._x][self._y - 1] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._y -= 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving right
                if self._action_pattern[action_tuple] == 3:
                    if self._y + 1 < self._matrix_unit.shape[1] and self._matrix_structure[self._x][self._y + 1] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._y += 1
                        self._matrix_unit[self._x][self._y] = 1

                # After moving, check if the agent is at a goal position
                if self._matrix_goal[self._x][self._y] == 1:
                    self._matrix_goal[self._x][self._y] = 0  # Remove the goal
                    self.goals_reached += 1
                    print(f"Goal reached at ({self._x}, {self._y}). Total goals reached: {self.goals_reached}/{self.total_goals}")

            self._state = []