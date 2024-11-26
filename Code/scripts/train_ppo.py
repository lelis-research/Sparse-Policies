import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import random
import time
import torch
import tyro
import numpy as np
import pickle
import gymnasium as gym
from args import Args
from utils import *
from typing import Callable
from torch.utils.tensorboard import SummaryWriter
from environment.combogrid_gym import make_env, make_env_combo_four_goals
from environment.karel_env.gym_envs.karel_gym import make_karel_env
# from environment.minigrid import make_env_simple_crossing, make_env_four_rooms
from train_ppo_agent import train_ppo


@timing_decorator
def main(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_time = int(time.time())
    
    # run_name = f"{args.env_id}__{args.total_timesteps}__{args.learning_rate}__{args.seed}__{run_time}_{args.exp_name}"
    run_name = f"{args.env_id}__{args.total_timesteps}__{args.seed}__{run_time}_{args.exp_name}"


    logger = get_logger('ppo_trainer_logger_' + str(args.seed) + "_" + run_name, args.log_level, args.log_path)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        # Update args with values from wandb.config
        args.learning_rate = wandb.config.learning_rate
        args.clip_coef = wandb.config.clip_coef
        args.ent_coef = wandb.config.ent_coef
        args.value_learning_rate = wandb.config.value_learning_rate
        args.hidden_size = wandb.config.hidden_size
        args.karel_seed = wandb.config.karel_seed

    writer = SummaryWriter(f"outputs/tensorboard/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger.info(f"Constructing tensorboard summary writer on outputs/tensorboard/runs/{run_name}")

    # TRY NOT TO MODIFY: seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Loading options if toggled
    if args.options_enabled:
        if args.options_only_len_3: # Not handled with options selected from levin loss
            save_path = 'binary/' + args.options_base_model + '_options_list_relu_' + str(args.options_hidden_size) + '_game_width_' + str(args.options_game_width) + '_num_epochs_' + str(args.options_num_epochs) + '_l1_' + str(args.options_l1_lambda) + '_lr_' + str(args.options_learning_rate) + '_onlyws3.pkl'
        else:
            save_path = 'binary/selected_options_relu_' + str(args.options_hidden_size) + '_gw_' + str(args.options_game_width) + '_num_epochs_' + str(args.options_num_epochs) + '_l1_' + str(args.options_l1_lambda) + '_lr_' + str(args.options_learning_rate) + '.pkl'
            # save_path = 'binary/' + args.options_base_model + '_options_list_relu_' + str(args.options_hidden_size) + '_game_width_' + str(args.options_game_width) + '_num_epochs_' + str(args.options_num_epochs) + '_l1_' + str(args.options_l1_lambda) + '_lr_' + str(args.options_learning_rate) + '.pkl'

        with open(save_path, 'rb') as f:
            options_list = pickle.load(f)
        print(f'Options list loaded from {save_path}')

        if args.env_id != "ComboTest":
            # excluding options from current problem
            current_problem = args.env_id[len("ComboGrid_"):]
            options_list = [option for option in options_list if option.problem != current_problem]
    

    # env setup
    game_width = args.game_width
    hidden_size = args.hidden_size
    problem = args.env_id
    l1_lambda = args.l1_lambda
    
    params = {
        'seed': seed,
        'hidden_size': hidden_size,
        'hidden_size_options': args.options_hidden_size,
        'game_width': game_width,
        'game_width_options': args.options_game_width,
        'l1_lambda': l1_lambda,
        'problem': problem,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'minibatch_size': args.minibatch_size,
        'num_iterations': args.num_iterations,
        'total timesteps': args.total_timesteps,
        "num_envs": args.num_envs,
        "ent_coef": args.ent_coef,
        "clip_coef": args.clip_coef,
        'sparse_reward': args.sparse_reward,
        'reward_diff': args.reward_diff,
        'reward_scale': args.reward_scale,
    }

    buffer = "\nParameters:"
    for key, value in params.items():
        buffer += f"\n- {key}: {value}"
    logger.info(buffer)

    # if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
    #     model_file_name = f'binary/simple-crossing-s9n1-v0/PPO-gw{args.game_width}' + \
    #                     f'-h{args.hidden_size}-lr{args.learning_rate}-sd{seed}_MODEL.pt'
    #     envs = gym.vector.SyncVectorEnv( 
    #         [make_env_simple_crossing(view_size=game_width, seed=seed) for _ in range(args.num_envs)])
    
    # elif args.env_id == "MiniGrid-FourRooms-v0":
    #     model_file_name = f'binary/four-rooms/PPO-gw{args.game_width}' + \
    #                     f'-h{args.hidden_size}-sd{seed}_MODEL.pt'
    #     envs = gym.vector.SyncVectorEnv( 
    #         [make_env_four_rooms(view_size=game_width, seed=seed) for _ in range(args.num_envs)])
 
    if "ComboGrid" in args.env_id:
        problem = args.env_id[len("ComboGrid_"):]
        if args.options_enabled:
            model_file_name = f'binary/PPO-{problem}-gw{game_width}-h{hidden_size}-l1l{l1_lambda}-lr{args.learning_rate}-totaltimestep{args.total_timesteps}-entcoef{args.ent_coef}-clipcoef{args.clip_coef}_options_MODEL.pt'
            envs = gym.vector.SyncVectorEnv(
                [make_env(rows=game_width, columns=game_width, problem=problem, options=options_list) for _ in range(args.num_envs)],
            ) 
        else:
            model_file_name = f'binary/PPO-{problem}-gw{game_width}-h{hidden_size}-l1l{l1_lambda}-lr{args.learning_rate}-totaltimestep{args.total_timesteps}-entcoef{args.ent_coef}-clipcoef{args.clip_coef}_MODEL.pt'
            envs = gym.vector.SyncVectorEnv(
                [make_env(rows=game_width, columns=game_width, problem=problem) for _ in range(args.num_envs)],
            )    
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        logger.info("envs.action_space.n", envs.action_space[0].n)

    elif "ComboTest" in args.env_id:
        if args.options_enabled:
            model_file_name = f'binary/PPO-{args.env_id}-gw{game_width}-h{hidden_size}-l1l{l1_lambda}-lr{args.learning_rate}-totaltimestep{args.total_timesteps}-entcoef{args.ent_coef}-clipcoef{args.clip_coef}_options_MODEL.pt'
            envs = gym.vector.SyncVectorEnv(
                [make_env_combo_four_goals(rows=game_width, columns=game_width, options=options_list) for _ in range(args.num_envs)],
            ) 
        else:
            model_file_name = f'binary/PPO-{args.env_id}-gw{game_width}-h{hidden_size}-l1l{l1_lambda}-lr{args.learning_rate}-totaltimestep{args.total_timesteps}-entcoef{args.ent_coef}-clipcoef{args.clip_coef}_MODEL.pt'
            envs = gym.vector.SyncVectorEnv(
                [make_env_combo_four_goals(rows=game_width, columns=game_width) for _ in range(args.num_envs)],
            )    
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        logger.info("envs.action_space.n", envs.action_space[0].n)

   
    elif "Karel" in args.env_id:
        model_file_name = f'binary/PPO-{args.env_id}-gw{args.game_width}-gh{args.game_height}-h{args.hidden_size}-lr{args.learning_rate}-sd{seed}-entcoef{args.ent_coef}-clipcoef{args.clip_coef}_vlr{args.value_learning_rate}_{args.ppo_type}_MODEL_{run_time}.pt'
        problem = args.env_id[len("Karel_"):]

        ## Method 4
        # args.num_envs = 10
        # seeds = [i for i in range(args.num_envs)]
        # envs = gym.vector.AsyncVectorEnv(
        #     [lambda seed=seed: make_karel_env(env_config={
        #         'task_name': problem,
        #         'env_height': args.game_height,
        #         'env_width': args.game_width,
        #         'max_steps': args.max_steps,
        #         'sparse_reward': args.sparse_reward,
        #         'crash_penalty': args.crash_penalty,
        #         'seed': seed,
        #         'initial_state': None,
        #     })() for seed in seeds]
        # )

        ## Method 3
        # args.num_envs = 10
        # seeds = [i for i in range(args.num_envs)]
        # env_fns = []

        # for i in range(args.num_envs):
        #     seed = seeds[i]  # Use a unique seed for each environment
        #     env_config = {
        #         'task_name': problem,
        #         'env_height': args.game_height,
        #         'env_width': args.game_width,
        #         'max_steps': args.max_steps,
        #         'sparse_reward': args.sparse_reward,
        #         'crash_penalty': args.crash_penalty,
        #         'seed': seed,  
        #         'initial_state': None,
        #     }
        #     env_fns.append(make_karel_env(env_config=env_config))

        # envs = gym.vector.SyncVectorEnv(env_fns)

        ## Method 2
        # args.num_envs = 10
        # seeds = [i for i in range(args.num_envs)]

        # def make_env_with_seed(seed: int) -> Callable:
        #     def _init():
        #         env_config = {
        #             'task_name': problem,
        #             'env_height': args.game_height,
        #             'env_width': args.game_width,
        #             'max_steps': args.max_steps,
        #             'sparse_reward': args.sparse_reward,
        #             'crash_penalty': args.crash_penalty,
        #             'seed': seed,  # Each env gets its own seed
        #             'initial_state': None,
        #         }
        #         # env = make_karel_env(env_config=env_config)()
        #         env_fn = make_karel_env(env_config=env_config) 
        #         env = env_fn()
        #         return env
        #     return _init
        
        # env_fns = [make_env_with_seed(seed) for seed in seeds]
        # envs = gym.vector.SyncVectorEnv(env_fns)

        ## Method 1
        env_config = {
            'task_name': problem,
            'env_height': args.game_height,
            'env_width': args.game_width,
            'max_steps': args.max_steps,
            'sparse_reward': args.sparse_reward,
            'crash_penalty': args.crash_penalty,
            'seed': args.karel_seed,
            'initial_state': None,

            'reward_diff': args.reward_diff,
            'final_reward_scale': args.reward_scale
        }
        envs = gym.vector.SyncVectorEnv(
            [make_karel_env(env_config=env_config) for _ in range(args.num_envs)]
        )

    else:
        raise NotImplementedError
    
        
    average_return = train_ppo(envs, args, model_file_name, device, writer, logger=logger, seed=seed)



if __name__ == "__main__":
    args = tyro.cli(Args)

    if "All" in args.env_id:
        for prob in ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]:
            args.env_id = f"ComboGrid_{prob}"
            main(args)
    else:
        main(args)
    