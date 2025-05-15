import copy
import math
from algorithms import State
import numpy as np
import random

class Map:
    """
    Class to store the map. The maps in folder dao-map are from movingai.org.
    """
    def __init__(self, file_name):
        self.file_name = file_name        
        self.map_file = open(self.file_name)
        self.type_map = self.map_file.readline()
        self.height = int(self.map_file.readline().split(' ')[1])
        self.width = int(self.map_file.readline().split(' ')[1])
        
        State.map_width = self.width
        State.map_height = self.height
        
        self.read_map()
        self.convert_data()
        
        self.map_file.close()
        
    def read_map(self):
        """
        Reads map from the file and stores it in memory.
        """
        line = self.map_file.readline()
        while 'map' not in line:
            line = self.map_file.readline()
        lines = self.map_file.readlines()

        self.data_str = []
        for line in lines:
            line_list = []
            line = line.replace('\n', '')
            for c in line:
                line_list.append(c)
            self.data_str.append(line_list)
        
    def convert_data(self):
        """
        Converts the map, initially in the movingai.org format, to a matrix of integers, where
        traversable cells have the value of 1 and non-traversable cells have the value of 0.
        
        The movingai.com maps are encoded as follows. 
        
        . - passable terrain
        G - passable terrain
        @ - out of bounds
        O - out of bounds
        T - trees (unpassable)
        S - swamp (passable from regular terrain)
        W - water (traversable, but not passable from terrain)
        """
        self.data_int = np.zeros((len(self.data_str), len(self.data_str[0])))

        for i in range(0, self.height):
            for j in range(0, self.width):
                if self.data_str[i][j] == '.' or self.data_str[i][j] == 'G':
                    self.data_int[i][j] = 0
                else:
                    self.data_int[i][j] = 1        
    
    def plot_map(self, closed_data, start, goal, filename):
        import matplotlib.pyplot as plt

        data_plot = copy.deepcopy(self.data_int)
        data_plot *= 100

        for i in range(0, self.height):
            for j in range(0, self.width):
                if data_plot[i][j] == 0:
                    data_plot[i][j] = -100

        for _, state in closed_data.items():
            data_plot[state.get_y()][state.get_x()] = 1

        data_plot[start.get_y()][start.get_x()] = -50
        data_plot[goal.get_y()][goal.get_x()] = -50

        plt.axis('off')
        plt.imshow(data_plot, cmap='Greys', interpolation='nearest')
        plt.savefig(filename)
        # plt.show()

    def random_state(self):
        """
        Generates a valid random state for a given map. 
        """
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        while self.data_int[y][x] == 1:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
        state = State(x, y)
        return state
    
    def is_valid_pair(self, x, y):
        """
        Verifies if an x-y pair is valid for a map.
        """
        if x < 0 or y < 0:
            return False
        if x >= self.width or y >= self.height:
            return False
        if self.data_int[y][x] == 1:
            return False
        return True
    
    def cost(self, x, y):
        """
        Returns the cost of an action.
        
        Diagonal moves cost 1.5; each action in the 4 cardinal directions costs 1.0
        """
        if x == 0 or y == 0:
            return 1
        else:
            return 1.5
    
    def successors(self, state):
        """
        Transition function: receives a state and returns a list with the neighbors of that state in the space
        """
        children = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if self.is_valid_pair(state.get_x() + i, state.get_y() + j):
                    s = State(state.get_x() + i, state.get_y() + j)
                    s.set_g(state.get_g() + self.cost(i, j))
                    children.append(s)
        return children
    

class MapKarel:
    """
    Adapter Map class for a KarelGymEnv instance.
    Builds a static grid of walls, extracts start and goal states,
    and provides successors() for four-directional search.
    """
    def __init__(self, env):
        # dimensions
        self.width  = env.env_width
        self.height = env.env_height
        State.map_width  = self.width
        State.map_height = self.height

        # static walls from feature index 4 of the Karel state
        state_arr = env.task.get_state()             # shape: (features, H, W)
        self.walls = state_arr[4].astype(bool)       # True where wall

        # start position (x, y, direction)
        r, c, d = env.task.get_hero_pos()
        self.start = State(c, r, d)

        # goal = marker position stored in task_specific
        goal_r, goal_c = env.task_specific.marker_position
        self.goal = State(goal_c, goal_r, 0)   # assume reaching any orientation at goal is OK

    def is_valid_pair(self, x, y):
        # inside bounds and not a wall
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return not self.walls[y, x]

    def cost(self, dx, dy):
        # uniform cost per move
        return 1.0

    def successors(self, state):
        children = []
        x, y, d = state.get_x(), state.get_y(), state.get_d()
        
        # 1) move forward
        # directions: 0=N,1=E,2=S,3=W
        delta = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)}
        dx, dy = delta[d]
        nx, ny = x + dx, y + dy
        if self.is_valid_pair(nx, ny):
            child = State(nx, ny, d)
            child.set_g(state.get_g() + 1)
            children.append(child)
        
        # 2) turn left
        newd = (d - 1) % 4
        left = State(x, y, newd)
        left.set_g(state.get_g() + 1)
        children.append(left)
        
        # 3) turn right
        newd = (d + 1) % 4
        right = State(x, y, newd)
        right.set_g(state.get_g() + 1)
        children.append(right)
        
        return children

    def plot_map(self, path, start, goal, filename, cost=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        img = np.ones((self.height, self.width), dtype=np.uint8)
        img[self.walls] = 0
        ax.imshow(img, cmap='gray', interpolation='nearest')

        # overlay path as red line with arrows
        xs = [s.get_x() for s in path]
        ys = [s.get_y() for s in path]
        ax.plot(xs, ys, '-o', color='red', markersize=4)
        for i in range(len(path)-1):
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[i+1], ys[i+1]
            ax.arrow(x0, y0, x1-x0, y1-y0,
                     head_width=0.2, head_length=0.2, length_includes_head=True, color='red')

        # mark start and goal
        ax.scatter([start.get_x()], [start.get_y()], marker='s', color='green', s=100)
        ax.scatter([goal.get_x()], [goal.get_y()], marker='*', color='blue', s=100)

        if cost is None and path:
            cost = path[-1].get_g()
    
        # Add cost as title
        if cost is not None:
            plt.title(f"Path Cost: {cost}", fontsize=12)

        ax.axis('off')
        plt.savefig(filename + '.png')
        print(f"Map saved to {filename}.png")