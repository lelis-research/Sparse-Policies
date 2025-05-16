import time
import getopt
import sys
import os
import argparse
import gymnasium as gym

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_FOLDER = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, CODE_FOLDER)

from algorithms import Dijkstra, AStar, State
from map import Map, MapKarel
from environment.karel_env.gym_envs.karel_gym import KarelGymEnv



def verify_path(start, goal, path, map):
    if path is None:
        return True

    if not (start == path[0]) or not (goal == path[-1]):
        return False

    for i in range(len(path) - 1):
        current = path[i]
        children = map.successors(current)
        
        contains_next = False
        for child in children:
            if child == path[i + 1]:
                contains_next = True
                break

        if not contains_next:
            return False
    return True


def main(args):

    env_config = {
        'task_name': args.task_name,
        'env_height': args.game_width,
        'env_width': args.game_width,
        'max_steps': args.max_steps,
        'seed': args.seed,
        'initial_state': None,
        'reward_scale': False,
        'wide_maze': args.wide
    }
    env = KarelGymEnv(env_config=env_config)
    env.reset()
    print(f"env to string: \n{env.task.to_string()}")


    gridded_map = MapKarel(env)
    dijkstra = Dijkstra(gridded_map)

    start = gridded_map.start
    goal  = gridded_map.goal
    
    nodes_expanded_dijkstra = []  
    time_dijkstra = []  
    start_states = []
    goal_states = []

    start_states.append(start)
    goal_states.append(goal)
       
        
    for i in range(0, len(start_states)):    
        start = start_states[i]
        goal = goal_states[i]
    
        time_start = time.time()
        print(f"Dijkstra start={start} -> goal={goal}")
        path, cost, expanded_diskstra = dijkstra.search(start, goal)
        print("Dijkstra cost", cost, "steps", len(path))
        print(f"path: {path}")
        time_end = time.time()
        gridded_map.plot_map(path, start, goal, f'{BASE_DIR}/plots/dijkstra_path_{args.task_name}_grid{args.game_width}_seed{args.seed}_wide{env_config["wide_maze"]}')
        nodes_expanded_dijkstra.append(expanded_diskstra)
        time_dijkstra.append(time_end - time_start)
        verified_path = verify_path(start, goal, path, gridded_map)

        # if cost != solution_costs[i] or not verified_path:
        # if not verified_path:
        #     print("There is a mismatch in the solution cost found by Dijkstra and what was expected for the problem:")
        #     print("Start state: ", start)
        #     print("Goal state: ", goal)
        #     print("Solution cost encountered: ", cost)
        #     # print("Solution cost expected: ", solution_costs[i])
        #     print("Is the path correct?", verified_path)
        #     print()    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', default="maze", type=str, help="[stair_climber, maze, ...]")
    parser.add_argument('--game_width', default=12, type=int)
    parser.add_argument('--max_steps', default=100, type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wide', action='store_true', help="Use wide maze")

    args = parser.parse_args()
    main(args)