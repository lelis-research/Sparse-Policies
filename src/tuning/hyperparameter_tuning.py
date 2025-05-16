import optuna
import time
import torch
import random
import numpy as np
import gymnasium as gym
import os

from scripts.args import Args
from code.training.train_ppo_agent import train_ppo
from train_ppo import get_logger 
from environment.combogrid_gym import make_env
from environment.minigrid import make_env_simple_crossing, make_env_four_rooms
from torch.utils.tensorboard import SummaryWriter

def objective(trial):
    args = Args()

    # Suggest hyperparameters using Optuna
    args.learning_rate = trial.suggest_categorical("learning_rate", [0.005, 0.001, 0.0005, 0.0001, 0.00005])
    args.clip_coef = trial.suggest_categorical("clip_coef", [0.1, 0.15, 0.2, 0.25, 0.3])
    args.ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.05, 0.1, 0.15, 0.2])
    # args.l1_lambda = trial.suggest_categorical("l1_lambda", [0.0, 0.005, 0.001, 0.0005, 0.0001])

    # avg_returns = []
    # for seed in args.seeds:
    args.seed = 0

    # Recalculate derived parameters
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.exp_name = f"{args.learning_rate}"

    run_time = int(time.time())
    run_name = f"{args.env_id}__{args.total_timesteps}__{args.learning_rate}__{args.seed}__{run_time}"

    logger = get_logger('ppo_trainer_logger_' + str(args.seed) + "_" + args.exp_name, args.log_level, args.log_path)

    writer = SummaryWriter(f"outputs/tensorboard/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger.info(f"Constructing tensorboard summary writer on outputs/tensorboard/runs/{run_name}")

    # Seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    game_width = args.game_width
    hidden_size = args.hidden_size
    problem = args.env_id
    l1_lambda = args.l1_lambda

    # Create environments based on args.env_id
    if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        model_file_name = f'binary/simple-crossing-s9n1-v0/PPO-gw{args.game_width}' + \
                            f'-h{args.hidden_size}-lr{args.learning_rate}-sd{seed}_MODEL.pt'
        envs = gym.vector.SyncVectorEnv(
            [make_env_simple_crossing(view_size=game_width, seed=seed) for _ in range(args.num_envs)])
    elif "ComboGrid" in args.env_id:
        problem = args.env_id[len("ComboGrid_"):]
        model_file_name = f'binary/PPO-{problem}-gw{game_width}-h{hidden_size}-l1l{l1_lambda}-lr{args.learning_rate}-totaltimestep{args.total_timesteps}-{run_time}_MODEL.pt'
        envs = gym.vector.SyncVectorEnv(
            [make_env(rows=game_width, columns=game_width, problem=problem) for _ in range(args.num_envs)],
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        logger.info("envs.action_space.n", envs.action_space[0].n)
    elif args.env_id == "MiniGrid-FourRooms-v0":
        model_file_name = f'binary/four-rooms/PPO-gw{args.game_width}' + \
                            f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
        envs = gym.vector.SyncVectorEnv(
            [make_env_four_rooms(view_size=game_width, seed=seed) for _ in range(args.num_envs)])
    else:
        raise NotImplementedError

    # Call train_ppo and get the average episodic return
    avg_return = train_ppo(envs, args, model_file_name, device, writer, logger=logger, seed=seed)
    # avg_returns.append(avg_return)

    envs.close()
    writer.close()

    # mean_return = sum(avg_returns) / len(avg_returns)
    trial.report(avg_return, 0)  # Report the metric to Optuna

    return -avg_return  # Optuna minimizes the objective function

if __name__ == "__main__":
    # Create the Optuna study
    study = optuna.create_study(direction='minimize', study_name='ppo_hyperparameter_tuning')
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters
    print("Best hyperparameters:")
    print(study.best_params)
    print("Best average episodic return:")
    print(-study.best_value)  # Negate because we minimized negative return

    # Save the study to a database
    study_storage = 'sqlite:///ppo_study.db'
    study_name = 'ppo_hyperparameter_tuning'
    study = optuna.create_study(study_name=study_name, storage=study_storage, load_if_exists=True, direction='minimize')
    study.optimize(objective, n_trials=50)