import copy
import random
import torch
import numpy as np
import torch.nn as nn
from models.model import CustomRNN, CustomRelu 
from environment.combogrid_gym import ComboGym
from gymnasium.vector import SyncVectorEnv
from environment.minigrid import MiniGridWrap
from torch.distributions.categorical import Categorical
from typing import Union

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")


class Trajectory:
    def __init__(self):
        self._sequence = []
        self.logits = []

    def add_pair(self, state, action, logits=None):
        self._sequence.append((state, action))
        # self.logits.append(copy.deepcopy(logits))

    def get_logits_sequence(self):
        return self.logits
    
    def get_trajectory(self):
        return self._sequence
    
    def get_action_sequence(self):
        return [a for _, a in self._sequence]
    
    def get_state_sequence(self):
        return [s for s, _ in self._sequence]
    
    def __repr__(self):
        return f"Trajectory(sequence={self._sequence})"

class RandomAgent:
    def run(self, env):
        trajectory = Trajectory()

        while not env.is_over():
            actions = env.get_actions()
            a = actions[random.randint(0, len(actions) - 1)]

            trajectory.add_pair(copy.deepcopy(env.get_observation()), a)
            env.apply_action(a)
        
        return trajectory
    
class PolicyGuidedAgent:
    def __init__(self):
        self._h = None
        self._epsilon = 0.3
        self._is_recurrent = False

    def choose_action(self, env, model, greedy=False, verbose=False):
        if random.random() < self._epsilon:
            actions = env.get_actions()
            a = actions[random.randint(0, len(actions) - 1)]
        else:
            if self._is_recurrent and self._h == None:
                self._h = model.init_hidden()
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            if self._is_recurrent:
                prob_actions, self._h = model(x_tensor, self._h)
            else:
                prob_actions = model(x_tensor)
                # print("prob actions: ", prob_actions)
            if greedy:
                a = torch.argmax(prob_actions).item()
            else:
                a = torch.multinomial(prob_actions, 1).item()
        return a
        
    def run(self, env, model, greedy=False, length_cap=None, verbose=False):
        if greedy:
            self._epsilon = 0.0

        if isinstance(model, CustomRNN):
            self._is_recurrent = True

        trajectory = Trajectory()
        current_length = 0

        if verbose: print('Beginning Trajectory')
        while not env.is_over():
            a = self.choose_action(env, model, greedy, verbose)
            trajectory.add_pair(copy.deepcopy(env), a)

            if verbose:
                print("env: \n", env)
                print("action: ", a)
                print()

            env.apply_action(a)

            current_length += 1
            if length_cap is not None and current_length > length_cap:
                break        
        
        self._h = None
        if verbose: print("End Trajectory \n\n")
        return trajectory

    def run_with_relu_state(self, env, model):
        trajectory = Trajectory()
        current_length = 0

        while not env.is_over():
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            prob_actions, hidden_logits = model.forward_and_return_hidden_logits(x_tensor)
            a = torch.argmax(prob_actions).item()
            
            trajectory.add_pair(copy.deepcopy(env), a)
            print(env.get_observation(), a, (hidden_logits >= 0).float().numpy().tolist())
            env.apply_action(a)

            current_length += 1  

        return trajectory
    
    def run_with_mask(self, env, model, mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        while not env.is_over():
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            # mask_tensor = torch.tensor(mask, dtype=torch.int8).view(1, -1)
            prob_actions = model.masked_forward(x_tensor, mask)
            a = torch.argmax(prob_actions).item()
            
            trajectory.add_pair(copy.deepcopy(env), a)
            env.apply_action(a)

            length += 1

            if length >= max_size_sequence:
                return trajectory


        return trajectory

    def check_stopping(self, env, model_y2, verbose=False):
        """
        This method checks the stopping condition using model_y2.
        Returns True if the agent should stop, otherwise False.
        """
        x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
        stopping_prob = model_y2(x_tensor).item()  # model_y2 outputs a probability
        if verbose:
            print(f"Stopping probability: {stopping_prob}")
        return stopping_prob <= 0.5
    
    def run_with_y1_y2(self, env, model_y1, model_y2, greedy=False, length_cap=None, verbose=False):
        """
        This method runs the environment using model_y1 to choose actions
        and model_y2 to determine when to stop.
        """

        if greedy:  self._epsilon = 0.0

        if isinstance(model_y1, CustomRNN): self._is_recurrent = True

        trajectory = Trajectory()
        current_length = 0

        if verbose: print('Beginning Trajectory')

        sequence_ended = False  # Flag to indicate the end of a sequence
    
        while not env.is_over():

            # Choose action using model_y1
            a = self.choose_action(env, model_y1, greedy, verbose)
            trajectory.add_pair(copy.deepcopy(env), a)

            if verbose: print(env, a, "\n")

            # Check stopping condition using model_y2
            if self.check_stopping(env, model_y2, verbose):
                if verbose: print("Stopping the current sequence based on model_y2.")
                sequence_ended = True  # End the current sequence, but continue the outer loop
                
            # Apply the chosen action
            env.apply_action(a)

            if sequence_ended:
                break

            current_length += 1
            if length_cap is not None and current_length > length_cap:
                break

        self._h = None
        if verbose: print("End Trajectory \n\n")

        return trajectory
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, envs, hidden_size=6):
        super().__init__()
        if isinstance(envs, ComboGym):
            observation_space_size = envs.get_observation_space()
            action_space_size = envs.get_action_space()
        elif isinstance(envs, MiniGridWrap):
            observation_space_size = envs.get_observation_space()
            action_space_size = envs.get_action_space()
        elif isinstance(envs, SyncVectorEnv):
            observation_space_size = envs.observation_space.shape[1]
            action_space_size = envs.action_space[0].n.item()
        else:
            raise NotImplementedError

        print(observation_space_size, action_space_size)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_space_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_space_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_space_size), std=0.01),
        )
        self.mask = None
        
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), logits
    
    def to_option(self, mask, option_size):
        self.mask = mask
        self.option_size = option_size

    def _masked_neuron_operation(self, logits, mask):
        """
        Apply a mask to neuron outputs in a layer.

        Parameters:
            x (torch.Tensor): The pre-activation outputs (linear outputs) from neurons.
            mask (torch.Tensor): The mask controlling the operation, where:
                                1 = pass the linear input
                                0 = pass zero,
                                -1 = compute ReLU as usual (part of the program).

        Returns:
            torch.Tensor: The post-masked outputs of the neurons.
        """
        if mask is None:
            raise Exception("No mask is set for the agent.")
        relu_out = torch.relu(logits)
        output = torch.zeros_like(logits)
        output[mask == -1] = relu_out[mask == -1]
        output[mask == 1] = logits[mask == 1]

        return output

    def _masked_forward(self, x, mask=None):
        if mask is None:
            mask = self.mask
        hidden_logits = self.actor[0](x)
        hidden = self._masked_neuron_operation(hidden_logits, mask)
        hidden_tanh = self.actor[1](hidden)
        output_logits = self.actor[2](hidden_tanh)

        probs = Categorical(logits=output_logits).probs
        
        return probs

    def run(self, env: Union[ComboGym, MiniGridWrap], length_cap=None, verbose=False):

        trajectory = Trajectory()
        current_length = 0
        self.actor.requires_grad = False

        o, _ = env.reset()
        
        done = False

        if verbose: print('Beginning Trajectory')
        while not done:
            o = torch.tensor(o, dtype=torch.float32)
            a, _, _, _, logits = self.get_action_and_value(o)
            trajectory.add_pair(copy.deepcopy(env), a.item())

            if verbose:
                print(env._game)
                print(a.item(), env.is_over())
                print()

            next_o, _, terminal, truncated, _ = env.step(a.item())
            
            current_length += 1
            if (length_cap is not None and current_length > length_cap) or \
                terminal or truncated:
                done = True     

            o = next_o   
        
        self._h = None
        if verbose: print("End Trajectory \n\n")
        return trajectory

    def run_with_relu_state(self, env, model):
        trajectory = Trajectory()
        current_length = 0

        while not env.is_over():
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
            prob_actions, hidden_logits = model.forward_and_return_hidden_logits(x_tensor)
            a = torch.argmax(prob_actions).item()
            
            trajectory.add_pair(copy.deepcopy(env), a)
            print(env.get_observation(), a, (hidden_logits >= 0).float().numpy().tolist())
            env.apply_action(a)

            current_length += 1  

        return trajectory
    
    def get_action_with_mask(self, x_tensor, mask=None):
        prob_actions = self._masked_forward(x_tensor, mask)
        a = torch.argmax(prob_actions).item()
        return a
    
    def run_with_mask(self, env, mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        while not env.is_over():
            x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)

            a = self.get_action_with_mask(x_tensor, mask)
            
            trajectory.add_pair(copy.deepcopy(env), a)
            env.step(a)

            length += 1

            if length >= max_size_sequence:
                return trajectory

        return trajectory


class IdentityLayer(nn.Module):
    def forward(self, x):
        return x

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)  # Quantize to -1 or 1
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output 

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

#TO DO: UPDATE LSTM STRUCTURE TO BE ABLE TO ENALBE/DISABLE FEATURE EXTRACTOR AND INPUT_TO_ACTOR
class LstmAgent(nn.Module):
    def __init__(self, envs, h_size=64):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 512)),
        )
        self.lstm = nn.LSTM(512, h_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        # self.actor = layer_init(nn.Linear(128 + envs.single_observation_space.shape[0], envs.single_action_space.n), std=0.01)
        # self.critic = layer_init(nn.Linear(128 + envs.single_observation_space.shape[0], 1), std=1)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(h_size + envs.single_observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n)),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(h_size + envs.single_observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1)),
        )

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            # print('d: ', d)
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        concatenated = torch.cat((hidden, x), dim=1)
        return self.critic(concatenated)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        concatenated = torch.cat((hidden, x), dim=1)
        logits = self.actor(concatenated)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(concatenated), lstm_state

class GruAgent(nn.Module):
    def __init__(self, envs, h_size=64, feature_extractor=False, greedy=False):
        super().__init__()
        self.input_to_actor = False
        self.greedy = greedy
        if feature_extractor:
            self.network = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 512)),
            )
            self.gru = nn.GRU(512, h_size, 1)
        else:
            self.network = IdentityLayer()
            self.gru = nn.GRU(envs.single_observation_space.shape[0], h_size, 1)

        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        # self.actor = layer_init(nn.Linear(128 + envs.single_observation_space.shape[0], envs.single_action_space.n), std=0.01)
        # self.critic = layer_init(nn.Linear(128 + envs.single_observation_space.shape[0], 1), std=1)
        if self.input_to_actor:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(h_size + envs.single_observation_space.shape[0], 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.single_action_space.n)),
            )

            self.critic = nn.Sequential(
                layer_init(nn.Linear(h_size + envs.single_observation_space.shape[0], 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1)),
            )
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(h_size, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.single_action_space.n)),
            )

            self.critic = nn.Sequential(
                layer_init(nn.Linear(h_size , 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1)),
            )

    def get_states(self, x, gru_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            # print('d: ', d)
            h, gru_state = self.gru(h.unsqueeze(0), (1.0 - d).view(1, -1, 1) * gru_state)
            # quantized_hidden = STEQuantize.apply(h) # no need to quantize hidden state
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, STEQuantize.apply(gru_state)

    def get_value(self, x, gru_state, done):
        if self.input_to_actor:
            hidden, _ = self.get_states(x, gru_state, done)
            concatenated = torch.cat((hidden, x), dim=1)
        else:
            hidden, _ = self.get_states(x, gru_state, done)
            concatenated = hidden
        return self.critic(concatenated)

    def get_action_and_value(self, x, gru_state, done, action=None):
        if self.input_to_actor:
            hidden, gru_state = self.get_states(x, gru_state, done)
            concatenated = torch.cat((hidden, x), dim=1)
        else: 
            hidden, gru_state = self.get_states(x, gru_state, done)
            concatenated = hidden
        logits = self.actor(concatenated)
        probs = Categorical(logits=logits)
        if action is None:
            if self.greedy:
                action = torch.tensor([torch.argmax(logits[i]).item() for i in range(len(logits))])
            else:
                action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(concatenated), gru_state
