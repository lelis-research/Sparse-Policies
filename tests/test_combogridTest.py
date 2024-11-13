import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Put it on /Code/ then run


from environment.combogrid import TestGame

def test_game_goals():
    game = TestGame(rows=3, columns=3)
    
    print("Initial Agent Position:", (game._x, game._y))
    print("Initial Goals Matrix:")
    print(game._matrix_goal)
    print("Total Goals:", game.total_goals)
    print("Goals Reached:", game.goals_reached)
    print("---")

    # Move Up to reach the top-middle goal
    action_sequence = [0, 0, 1]  # Upper goal
    for action in action_sequence:
        game.apply_action(action)
    print(game)
    
    # Move Right to reach the right-middle goal
    action_sequence = [1, 0, 2, 0, 1, 2]  # Right goal
    for action in action_sequence:
        game.apply_action(action)
    print(game)
    
    # Move Down to reach the bottom-middle goal
    action_sequence = [0, 1, 2, 2, 1, 0]  # lower goal
    for action in action_sequence:
        game.apply_action(action)
    print(game)
    
    # Move Left to reach the left-middle goal
    action_sequence = [2, 1, 0, 0, 0, 1]  # Left goal
    for action in action_sequence:
        game.apply_action(action)
    
    # Check if the game is over
    if game.is_over():
        print("All goals reached!")
    else:
        print("Goals remaining.")
    
    # Print final state
    print("Final Agent Position:", (game._x, game._y))
    print("Final Goals Matrix:")
    print(game._matrix_goal)
    print("Total Goals:", game.total_goals)
    print("Goals Reached:", game.goals_reached)

test_game_goals()