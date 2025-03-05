import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
from oblique_tree import ObliqueTree, LabelNode
from models.student import StudentPolicy
import argparse
from graphviz import Source
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch



def load_model(model_path, input_dim, hidden_size):
    model = StudentPolicy(input_dim, hidden_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def extract_weights(model):
    weights = {
        'W1': model.fc1.weight.detach().numpy(),  # Shape: (hidden_size, input_dim)
        'B1': model.fc1.bias.detach().numpy().reshape(-1, 1),
        'W2': model.fc2.weight.detach().numpy(),   # Shape: (output_size, hidden_size)
        'B2': model.fc2.bias.detach().numpy().reshape(-1, 1)
    }
    return weights

def print_tree(node, indent="", feature_names=None):
    """Recursively print the oblique decision tree structure."""
    if isinstance(node, LabelNode):
        print(f"{indent}Label: {node.label()}")
        return
    
    # Extract weights and bias
    weights = node._weights[f'OW{node._layer}'][node._neuron - 1]
    bias = node._weights[f'OB{node._layer}'][node._neuron - 1][0]
    
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(len(weights))]
    
    # Build decision condition string
    condition = " + ".join([f"{w:.2f}*{name}" for w, name in zip(weights, feature_names)])
    condition += f" + {bias:.2f} < 0"
    
    print(f"{indent}Layer {node._layer}, Neuron {node._neuron}: {condition}")
    print(f"{indent}├── Left (True):")
    print_tree(node._left, indent + "│   ", feature_names)
    print(f"{indent}└── Right (False):")
    print_tree(node._right, indent + "    ", feature_names)
    print()

def export_tree_to_dot(node, node_id=0, dot_lines=None):
    if dot_lines is None:
        dot_lines = ['digraph ObliqueTree {', '  node [fontname="Helvetica"];']
    
    current_id = node_id
    
    if isinstance(node, LabelNode):
        dot_lines.append(f'  {current_id} [label="Label: {node.label()}", shape="box"];')
        return current_id + 1, dot_lines
    
    weights = node._weights[f'OW{node._layer}'][node._neuron - 1]
    bias = node._weights[f'OB{node._layer}'][node._neuron - 1][0]
    
    # Simplify the condition to avoid special characters
    condition = " + ".join([f"{w:.2f}x{i+1}" for i, w in enumerate(weights)])
    condition += f" + {bias:.2f} < 0"
    
    # Escape quotes and newlines
    condition = condition.replace('"', '\\"').replace('\n', '\\n')
    dot_lines.append(f'  {current_id} [label="L{node._layer}N{node._neuron}\\n{condition}"];')
    
    left_id, dot_lines = export_tree_to_dot(node._left, current_id + 1, dot_lines)
    dot_lines.append(f'  {current_id} -> {left_id} [label="True"];')
    
    right_id, dot_lines = export_tree_to_dot(node._right, left_id + 1, dot_lines)
    dot_lines.append(f'  {current_id} -> {right_id} [label="False"];')
    
    if current_id == 0:
        dot_lines.append('}')
    return right_id + 1, dot_lines

def plot_tree_with_matplotlib(root, figsize=(30, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    
    # Track node positions and connections
    node_positions = {}
    connections = []    # Format: (parent_pos, child_pos, branch)
    
    # Recursive layout helper
    def layout(node, depth=0, pos=0.5, parent_pos=None, branch=None):
        if not node:
            return pos
        
        # Store node position
        node_id = id(node)
        node_positions[node_id] = (pos, -depth)
        
        # Draw node
        if isinstance(node, LabelNode):
            ax.text(pos, -depth, f"Label: {node.label()}", 
                    ha='center', va='center',
                    bbox=dict(facecolor='lightgreen', alpha=0.9))
        else:
            weights = node._weights[f'OW{node._layer}'][node._neuron - 1]
            bias = node._weights[f'OB{node._layer}'][node._neuron - 1][0]
            condition = "\n".join([
                f"Σwᵢxᵢ + b < 0",
                f"x: [CartPos, CartVel, PoleAng, PoleVel]",
                f"w: {weights.round(2)}",
                f"b: {bias:.2f}"
            ])
            ax.text(pos, -depth, condition, 
                    ha='center', va='center',
                    bbox=dict(facecolor='lightblue', alpha=0.9))
            
        
        # Layout children
        if parent_pos is not None:
            connections.append((parent_pos, (pos, -depth), branch))
        
        if not isinstance(node, LabelNode):
            left_pos = layout(node._left, depth+1, pos - 0.5/(depth+2), 
                             (pos, -depth), branch='left')
            right_pos = layout(node._right, depth+1, pos + 0.5/(depth+2), 
                              (pos, -depth), branch='right')
        return pos
    
    # Start layout from root
    layout(root)
    
    # Draw connections
    for (x1, y1), (x2, y2), branch in connections:
        ax.plot([x1, x2], [y1, y2], 'k-', lw=1)
        ax.text((x1+x2)/2, (y1+y2)/2, 
                   "T" if branch == 'left' else "F", 
                   ha='center', va='center', 
                   backgroundcolor='white', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('tree_matplotlib.png')
    plt.close()


def main():

    parser = argparse.ArgumentParser(description="Train a student model using DAgger.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model.")
    parser.add_argument('--hidden_size', type=int, default=1, help="Size of the hidden layer.")
    args = parser.parse_args()

    model_path = args.model_path
    input_dim = 4 
    hidden_size = args.hidden_size
    
    model = load_model(model_path, input_dim, hidden_size)
    print(model)
    weights = extract_weights(model)
    dims = [input_dim, hidden_size, 2]
    
    ot = ObliqueTree()
    ot.induce_oblique_tree(weights, dims)

    print("\n=== Oblique Decision Tree Structure ===")
    print_tree(ot._root, feature_names=["CartPos", "CartVel", "PoleAng", "PoleVel"])
    
    plot_tree_with_matplotlib(ot._root)


    # # Example 1
    # sample_obs = np.array([0.1, -0.2, 0.05, 0.3])
    # print("\nOblique Tree prediction:", ot.classify(sample_obs))
    
    # # Compare with student model
    # with torch.no_grad():
    #     logits = model(torch.FloatTensor(sample_obs))
    # print("Student prediction:", torch.argmax(logits).item())

    
    # Example 2
    similar_answers = 0
    total = 1000
    for _ in range(total):
        # Sample random observations from the CartPole observation space
        random_obs = np.array([
            np.random.uniform(-4.8, 4.8),                # Cart position
            np.random.uniform(-10, 10),                  # Cart velocity (using reasonable bounds)
            np.random.uniform(-0.41887903, 0.41887903),  # Pole angle
            np.random.uniform(-10, 10)                   # Pole angular velocity (using reasonable bounds)
        ])
        
        # Get predictions from both models
        tree_prediction = ot.classify(random_obs)
        
        with torch.no_grad():
            model_logits = model(torch.FloatTensor(random_obs))
            model_prediction = torch.argmax(model_logits).item()
        
        # Check if predictions match
        if tree_prediction != model_prediction:
            print(f"-- Mismatch found! Observation: {random_obs}")
            # print(f"-- Tree predicted: {tree_prediction}, Model predicted: {model_prediction}")

        else:
            similar_answers += 1

    print(f"Similarity: {float(similar_answers/total):.3f}")




if __name__ == "__main__":
    main()