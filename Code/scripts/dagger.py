import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from environment.cartpole_gym import LastActionObservationWrapper
from models.student import StudentPolicy, StudentPolicySigmoid
from data.custom_dataset import DemonstrationDataset
import re



def collect_teacher_demonstrations(teacher, env, num_episodes, device):
    dataset = DemonstrationDataset()
    for _ in range(num_episodes):

        seed = np.random.randint(0, 1e6)
        obs, _ = env.reset(seed=seed)
                
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _, _ = teacher.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
            dataset.add(obs, action)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = next_obs
    return dataset


def dagger_iteration(student, teacher, env, dataset, num_episodes, device):
    student.eval()
    for _ in range(num_episodes):

        seed = np.random.randint(0, 1e6)
        obs, _ = env.reset(seed=seed)

        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get student action (not used for stepping)
            with torch.no_grad():
                # # For Softmax student
                # student_logits = student(obs_tensor)
                # student_action = torch.argmax(student_logits, dim=-1).cpu().numpy()[0]

                # For Sigmoid student
                # print(f"== full obs: {obs_tensor} vs last2element obs: {obs_tensor[:, -2:]}")
                student_output = student(obs_tensor)
                student_action = (student_output >= 0.5).int().cpu().numpy()[0][0]
            
            # Get teacher action for correction
            with torch.no_grad():
                teacher_action, _, _, _, _ = teacher.get_action_and_value(obs_tensor)
            teacher_action = teacher_action.cpu().numpy()[0]

            # print(f"=== Student action: {student_action}, Teacher action: {teacher_action}")
            
            # Add corrected (obs, teacher_action) to dataset
            dataset.add(obs, teacher_action)
            
            # Step using student action to collect next states
            next_obs, _, terminated, truncated, _ = env.step(student_action)
            done = terminated or truncated
            obs = next_obs
    return dataset


def train_student(student, dataset, batch_size, epochs, lr, l1_lambda, device):
    print("Training student...")
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    
    # criterion = nn.CrossEntropyLoss()   # for Softmax student
    criterion = nn.BCELoss()            # for Sigmoid student

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0
        for obs_batch, action_batch in dataloader:
            obs_batch = obs_batch.to(device)

            # action_batch = action_batch.to(device)
            action_batch = action_batch.float().to(device)  # Convert to float for BCE loss

            # logits = student(obs_batch)
            # loss = criterion(logits, action_batch)

            # print(f"=== full obs: {obs_batch} vs last2element obs: {obs_batch[:, -2:]}")

            sigmoid_output = student(obs_batch)  # Get sigmoid output from student
            action_batch = action_batch.view(sigmoid_output.shape) # Reshape action_batch to match sigmoid_output
            loss = criterion(sigmoid_output, action_batch) # Calculate BCE loss

            l1_loss = sum(torch.norm(param, p=1) for param in student.parameters())
            loss += l1_lambda * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"==== Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def main():

    parser = argparse.ArgumentParser(description="Train a student model using DAgger.")

    parser.add_argument('--teacher_model_path', type=str, required=True, help="Path to the teacher model.")
    parser.add_argument('--teacher_feature_extractor', action='store_true', help="Teacher uses a feature extractor.")

    parser.add_argument('--student_hidden_size', type=int, default=1, help="Size of the student hidden layer.")
    parser.add_argument('--student_l1', type=float, default=0.0, help="L1 regularization for student model.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")

    parser.add_argument('--dagger_iterations', type=int, default=5, help="Number of DAgger iterations.")
    parser.add_argument('--initial_episodes', type=int, default=10, help="Initial teacher demonstrations.")
    parser.add_argument('--epochs_per_iteration', type=int, default=10, help="Training epochs per DAgger iteration.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def make_env():
        base_env = gym.make("CartPole-v1", max_episode_steps=250)   # 250 for training
        wrapped_env = LastActionObservationWrapper(
            base_env, 
            train_mode=True, 
            last_action_in_obs=False
        )
        return wrapped_env
    env = make_env()
    teacher_envs = gym.vector.SyncVectorEnv([lambda: make_env()])  # For initializing teacher

    model_pattern = re.compile(
        r'binary/PPO-Cartpole-gw\d+-gh\d+-'
        r'h(?P<hidden_size>\d+)-'
        r'lr(?P<lr>[0-9eE\.\-]+)-'  # Allow scientific notation (e.g., 1e-05)
        r'sd(?P<model_seed>\d+)-'
        r'entcoef(?P<ent_coef>[0-9.]+)-'
        r'clipcoef(?P<clip_coef>[0-9.]+)-'
        r'l1(?P<l1_lambda>[0-9.]+)-'
        r'(?P<ppo_type>\w+)-MODEL.*\.pt$'
    )
    match = model_pattern.match(args.teacher_model_path)
    if not match:
        print(f"**** Skipping unmatched file: {args.teacher_model_path}")
        
    teacher_hidden_size = int(match.group('hidden_size'))
    teacher_ppo_type = match.group('ppo_type')

    
    # Load teacher model
    if teacher_ppo_type == 'original':
        from agents import PPOAgent
        teacher = PPOAgent(
            teacher_envs,
            hidden_size=teacher_hidden_size,
            feature_extractor=args.teacher_feature_extractor,
            greedy=True
        ).to(device)
    elif teacher_ppo_type == 'gru':
        from agents import GruAgent
        teacher = GruAgent(
            teacher_envs,
            h_size=teacher_hidden_size,
            feature_extractor=args.teacher_feature_extractor,
            greedy=True
        ).to(device)
    teacher.load_state_dict(torch.load(args.teacher_model_path, map_location=device))
    teacher.eval()
    print(f"\nTeacher model loaded from {args.teacher_model_path}\n")
    
    # Determine input dimension
    input_dim = env.observation_space.shape[0]
    # student = StudentPolicy(input_dim, hidden_size=args.student_hidden_size).to(device)
    student = StudentPolicySigmoid(input_dim, hidden_size=args.student_hidden_size).to(device)
    dataset = DemonstrationDataset()
    
    # Collect initial teacher demonstrations
    print("Collecting initial demonstrations...")
    initial_data = collect_teacher_demonstrations(teacher, env, args.initial_episodes, device)
    dataset.observations = initial_data.observations
    dataset.actions = initial_data.actions
    print(f"Initial demonstrations collected: {len(dataset)}")
    prev_len = len(dataset)
    
    # DAgger iterations
    for iter in range(args.dagger_iterations):
        
        print(f"\nDAgger Iteration {iter + 1}/{args.dagger_iterations}")
        dataset = dagger_iteration(student, teacher, env, dataset, num_episodes=5, device=device)
        
        train_student(student, dataset, args.batch_size, args.epochs_per_iteration, args.lr, args.student_l1, device)
        print(f"Dataset len added: {len(dataset) - prev_len}")
        prev_len = len(dataset)
    
    
    teacher_name = args.teacher_model_path.split('/')[-1]
    student_name = teacher_name.replace(".pt", f"-student-sh{args.student_hidden_size}-sl1{args.student_l1}-slr{args.lr}.pt")
    print(f"\nstudent_name: {student_name}")

    student_model_path = f"{project_root}/Scripts/binary/cartpole/obs2/"
    os.makedirs(student_model_path, exist_ok=True)
    torch.save(student.state_dict(), student_model_path + student_name)
    print(f"\nStudent model saved to {student_model_path + student_name}")


if __name__ == "__main__":
    main()