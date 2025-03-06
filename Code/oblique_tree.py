"""
From ReLU to Oblique Trees Project
"""


import copy
import numpy as np


class Node:
    def __init__(self, weights, layer, neuron):
        self._weights = weights
        self._layer = layer
        self._neuron = neuron

    def add_left_child(self, left):
        self._left = left

    def add_right_child(self, right):
        self._right = right


class LabelNode:
    def __init__(self, label):
        self._label = label

    def label(self):
        return self._label


class ObliqueTree:

    def build_tree(self, root):

        if root._layer == 1 and False:  # Stop at hidden layer

            # Get output layer weights and bias
            W2_0 = self.output_weights[0][0]  # Output neuron 0 weight
            W2_1 = self.output_weights[1][0]  # Output neuron 1 weight
            B2_0 = self.output_bias[0][0]     # Output neuron 0 bias
            B2_1 = self.output_bias[1][0]     # Output neuron 1 bias

            # # Method 1
            # # Calculate output values based on hidden activation
            # left_output = W2_0 * (-1) + B2_0   # If hidden < 0
            # right_output = W2_1 * (+1) + B2_1  # If hidden >= 0
            # print(f"Left output: {left_output:.2f}, Right output: {right_output:.2f}")

            # # Determine labels using argmax
            # left_label = 0 if left_output > right_output else 1
            # right_label = 1 - left_label


            # Method 2
            # Case 1: ReLU(a1) = 0 when a1 < 0
            left_output_0 = B2_0  # For output neuron 0
            left_output_1 = B2_1  # For output neuron 1
            left_label = 0 if left_output_0 > left_output_1 else 1

            # Case 2: For a1 >= 0
            if W2_0 == W2_1:  # Parallel lines, bias determines winner for all positive a1
                right_label = 0 if B2_0 > B2_1 else 1
            else:
                # Find the intersection point: where a1*W2_0 + B2_0 = a1*W2_1 + B2_1
                # a1*(W2_0 - W2_1) = B2_1 - B2_0
                # a1 = (B2_1 - B2_0)/(W2_0 - W2_1)
                intersection = (B2_1 - B2_0)/(W2_0 - W2_1)
                right_label = 0 if W2_0 > W2_1 else 1
                print(f"Intersection at a1={intersection:.4f}")
            
            if left_label == right_label:
                print("=== Warning: Decision boundary is not oblique")


            root.add_left_child(LabelNode(left_label))
            root.add_right_child(LabelNode(right_label))
            return

        if root._layer > 1:
            OW = root._weights[f'OW{root._layer - 1}'] * root._weights[f'W{root._layer}'][root._neuron - 1].T.reshape((root._weights[f'W{root._layer}'][root._neuron - 1].T.shape[0], 1))
            OB = root._weights[f'OB{root._layer - 1}'] * root._weights[f'W{root._layer}'][root._neuron - 1].T.reshape((root._weights[f'W{root._layer}'][root._neuron - 1].T.shape[0], 1))

            root._weights[f'OW{root._layer}'][root._neuron - 1] = np.sum(OW, axis = 0)
            root._weights[f'OB{root._layer}'][root._neuron - 1] = np.sum(OB, axis = 0) + root._weights[f'B{root._layer}'][root._neuron - 1]

        if root._layer == len(self._dims) - 1 and root._neuron == self._dims[-1]: # last neuron in the last layer
            print("== Last neuron in the last layer")
            label_left = LabelNode(0)
            label_right = LabelNode(1)

            root.add_left_child(label_left)
            root.add_right_child(label_right)
            return

        left_weights = copy.deepcopy(root._weights)
        right_weights = copy.deepcopy(root._weights)

        left_weights[f'OW{root._layer}'][root._neuron - 1] = 0
        left_weights[f'OB{root._layer}'][root._neuron - 1] = 0

        if root._neuron == self._dims[root._layer]:
            left_child = Node(left_weights, root._layer + 1, 1)
            right_child = Node(right_weights, root._layer + 1, 1)
        else:
            left_child = Node(left_weights, root._layer, root._neuron + 1)
            right_child = Node(right_weights, root._layer, root._neuron + 1)

        root.add_left_child(left_child)
        root.add_right_child(right_child)

        self.build_tree(left_child)
        self.build_tree(right_child)

    def induce_oblique_tree(self, weights, dims):
        self._dims = dims

        for i in range(1, len(dims)):
            weights[f'OW{i}'] = np.zeros((dims[i], dims[0]))
            weights[f'OB{i}'] = np.zeros((dims[i], 1))
        # Initialize only for the hidden layer (layer 1)

        weights['OW1'] = copy.deepcopy(weights['W1'])  # Hidden layer weights
        weights['OB1'] = copy.deepcopy(weights['B1'])  # Hidden layer bias

        # Output layer weights for argmax
        self.output_weights = weights['W2']  # Shape: (2, 1)
        self.output_bias = weights['B2']     # Shape: (2, 1)

        print(weights[f'OW{1}'])
        print(weights[f'OB{1}'])

        self._root = Node(weights, 1, 1)
        self.build_tree(self._root)

    def classify_instance(self, root, x):
        if isinstance(root, LabelNode):
            return root.label()

        value_node = np.dot(root._weights[f'OW{root._layer}'][root._neuron - 1], x) + root._weights[f'OB{root._layer}'][root._neuron - 1]

        if value_node < 0:
            label = self.classify_instance(root._left, x)
        else:
            label = self.classify_instance(root._right, x)

        return label

    def classify(self, x):
        return self.classify_instance(self._root, x)