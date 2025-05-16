import heapq
import math

class State:
    """
    Class to represent a state on grid-based pathfinding problems. The class contains two static variables:
    map_width and map_height containing the width and height of the map. Although these variables are properties
    of the map and not of the state, they are used to compute the hash value of the state, which is used
    in the CLOSED list. 

    Each state has the values of x, y, g, h, and cost. The cost is used as the criterion for sorting the nodes
    in the OPEN list for both Dijkstra's algorithm and A*. For Dijkstra the cost should be the g-value, while
    for A* the cost should be the f-value of the node. 
    """
    map_width = 0
    map_height = 0
    
    def __init__(self, x, y, d=0):
        """
        Constructor - requires the values of x and y of the state. All the other variables are
        initialized with the value of 0.
        """
        self._x = x
        self._y = y
        self._d = d
        self._g = 0
        self._cost = 0
        self._parent = None
        
    def __repr__(self):
        """
        This method is invoked when we call a print instruction with a state. It will print [x, y],
        where x and y are the coordinates of the state on the map. 
        """
        state_str = "[" + str(self._x) + ", " + str(self._y) + ", " + str(self._d) + "]"
        return state_str
    
    def __lt__(self, other):
        """
        Less-than operator; used to sort the nodes in the OPEN list
        """
        return self._cost < other._cost
    
    def state_hash(self):
        """
        Given a state (x, y), this method returns the value of x * map_width + y. This is a perfect 
        hash function for the problem (i.e., no two states will have the same hash value). This function
        is used to implement the CLOSED list of the algorithms. 
        """
        # return self._y * State.map_width + self._x
        base = self._y * State.map_width + self._x
        return self._d * (State.map_width * State.map_height) + base
    
    def __eq__(self, other):
        """
        Method that is invoked if we use the operator == for states. It returns True if self and other
        represent the same state; it returns False otherwise. 
        """
        return self._x == other._x and self._y == other._y and self._d == other._d

    def get_x(self):
        """
        Returns the x coordinate of the state
        """
        return self._x
    
    def set_parent(self, parent):
        """
        Sets the parent of a node in the search tree
        """
        self._parent = parent

    def get_parent(self):
        """
        Returns the parent of a node in the search tree
        """
        return self._parent
    
    def get_y(self):
        """
        Returns the y coordinate of the state
        """
        return self._y
    
    def get_g(self):
        """
        Returns the g-value of the state
        """
        return self._g
    
    def get_d(self):      
        """
        Returns the direction of the agent in the state
        """
        return self._d
        
    def set_g(self, g):
        """
        Sets the g-value of the state
        """
        self._g = g

    def get_cost(self):
        """
        Returns the cost of a state; the cost is determined by the search algorithm
        """
        return self._cost
    
    def set_cost(self, cost):
        """
        Sets the cost of the state; the cost is determined by the search algorithm 
        """
        self._cost = cost

    def get_heuristic(self, target_state):
        """
        Returns the Octile distance heuristic between the state and the target state.

        Octile distance function. For states (x1, y1) and (x2, y2), it returns 
        max(|x1 - x2|, |y1 - y2|) + 0.5 * min(|x1 - x2|, |y1 - y2|)
        """
        dist_x = abs(self.get_x() - target_state.get_x())
        dist_y = abs(self.get_y() - target_state.get_y())
        return abs(dist_x - dist_y) + 1.5 * min(dist_x, dist_y)
    
class Search:
    """
    Interface for a search algorithm. It contains an OPEN list and a CLOSED list.

    The OPEN list is implemented with a heap, which can be done with the library heapq
    (https://docs.python.org/3/library/heapq.html).    
    
    The CLOSED list is implemented as a dictionary where the state hash value is used as key.
    """
    def __init__(self, gridded_map):
        self.map = gridded_map
        self.OPEN = []
        self.CLOSED = {}
    
    def search(self, start, goal):
        """
        Search method that needs to be implemented (either Dijkstra or BiHS).
        """
        raise NotImplementedError()
    
    def get_path(self, node):
        path = [node]
        parent = node.get_parent()
        while parent is not None:
            path.append(parent)
            parent = parent.get_parent()
        return path[::-1]

    def get_closed_data(self):
        """
        Returns the set of states in the CLOSED list (or lists) of the search algorithm
        """
        raise NotImplementedError()
    
    def get_closed_data(self):
        """
        Returns the CLOSED list of the search algorithm; used for plotting map with the nodes generated
        """
        return self.CLOSED

class Dijkstra(Search):

    def search(self, start, goal):
        """
        Disjkstra's Algorithm: receives a start state and a goal state as input. It returns the
        cost of a path between start and goal and the number of nodes expanded.

        If a solution isn't found, it returns -1 for the cost.
        """
        start.set_cost(start.get_g())
        self.start = start
        self.goal = goal
        
        self.OPEN.clear()
        self.CLOSED.clear()
        nodes_expanded = 1
        
        heapq.heappush(self.OPEN, start)
        self.CLOSED[start.state_hash()] = start
        
        while len(self.OPEN) > 0:
            node = heapq.heappop(self.OPEN)
            nodes_expanded += 1
            
            # goal test ignoring direction
            if node.get_x() == goal.get_x() and node.get_y() == goal.get_y():
                return self.get_path(node), node.get_g(), nodes_expanded
            
            children = self.map.successors(node)
            for child in children:
                hash_value = child.state_hash()
                child.set_cost(child.get_g())
                child.set_parent(node)
                
                if hash_value in self.CLOSED and self.CLOSED[hash_value].get_g() > child.get_g():
                    heapq.heappush(self.OPEN, child)
                    self.CLOSED[hash_value].set_g(child.get_g())
                    self.CLOSED[hash_value].set_parent(child.get_parent())
                
                if hash_value not in self.CLOSED:
                    heapq.heappush(self.OPEN, child)
                    self.CLOSED[hash_value] = child
                    
        return None, -1, nodes_expanded
    
class AStar(Search):
    
    def compute_cost(self, state):
        state.set_cost(state.get_g() + state.get_heuristic(self.goal))

    def search(self, start, goal):
        """
        A* Algorithm: receives a start state and a goal state as input. It returns the
        cost of a path between start and goal and the number of nodes expanded.

        If a solution isn't found, it returns -1 for the cost.
        """
        self.start = start
        self.goal = goal

        self.compute_cost(self.start)
        self.compute_cost(self.goal)
        
        self.OPEN.clear()
        self.CLOSED.clear()
        nodes_expanded = 0
        
        heapq.heappush(self.OPEN, self.start)
        self.CLOSED[start.state_hash()] = self.start
        while len(self.OPEN) > 0:
            node = heapq.heappop(self.OPEN)
            
            if node == self.goal:
                path = self.get_path(node)
                return path, node.get_g(), nodes_expanded
            
            if node.state_hash() in self.CLOSED and self.CLOSED[node.state_hash()].get_g() < node.get_g():
                continue
            
            nodes_expanded += 1

            children = self.map.successors(node)
            for child in children:
                hash_value = child.state_hash()
                child.set_parent(node)
                self.compute_cost(child)
                
                if hash_value in self.CLOSED and self.CLOSED[hash_value].get_g() > child.get_g():
                    # self.CLOSED[hash_value].set_g(child.get_g())
                    # self.compute_cost(self.CLOSED[hash_value])
                    # heapq.heapify(self.OPEN)
                    heapq.heappush(self.OPEN, child)
                    self.CLOSED[hash_value] = child

                
                if hash_value not in self.CLOSED:
                    heapq.heappush(self.OPEN, child)
                    self.CLOSED[hash_value] = child
        return None, -1, nodes_expanded
    