# Training
You can train `PPO` and `PPO+GRU` models on these environments:

- **Karel Tasks:** `Karel_stair_climber`, `Karel_maze`, `Karel_top_off`, `Karel_four_corner`, `Karel_harvester` 
- **Cartpole Tasks:** `Cartpole`, `CartpoleEasy`
- **Parallel Park:** `car`
- **Quad:** `Quad`, `QuadPO`

<br>
To train models:

```
python src/scripts/train_ppo.py \
--env_id Karel_stair_climber \
--game_width 12 \
--game_height 12 \
--max_steps 50 \
--num_steps 500 \
--sparse_reward \  # for tasks with sparse reward only
--hidden_size 32 \  # hidden size of the actor
--total_timesteps 2_000_000 \
--num_envs 1 \  # num envs to run in parallel
--num_minibatches 1 \
--ppo_type original \  # set gru if you want to train PPO+GRU
--all_initial_confs \  # only for Karel_stair_climber, Karel_maze, Karel_top_off. for Karel_four_corner and Karel_harvester use --multi_initial_confs
--l1_lambda 0.0 \
--learning_rate 0.001 \
--ent_coef 0.1 \
--clip_coef 0.2 \
--exp_name nam _of_the_experiment
```

To train 30 seeds in parallel add `--multiprocessing`.

To train the `wide maze`:
```
--env_id Karel_maze --wide_maze
```

# Evaluation

Evaluation script for karel tasks:
```
python src/scripts/evaluate_on_bigger_grid.py \
--task_name stair_climber \
--game_width 12 \   # details of the model you want to load
--game_height 12 \   # details of the model you want to load
--max_steps 100 \   # for evaluation
--sparse_reward \   # details of the model you want to load
--model_seed 0 \   # details of the model you want to load
--karel_seed 9 \    # initial conf seed, for reproducability
--hidden_size 32 \   # details of the model you want to load
--ppo_type original \   # details of the model you want to load
--game_width_eval 12 \  # grid width that you want to test on
--game_height_eval 12 \  # grid height that you want to test on
--learning_rate 0.0001 \   # details of the model you want to load
--ent_coef 0.1 \   # details of the model you want to load
--clip_coef 0.2 \   # details of the model you want to load
--time 1738007119 \  # timestep of the model
--record_video  # to record a video of it
```

<br>

You can use [`evaluate_cartpole.py`](https://github.com/lelis-research/Sparse-Policies/blob/main/src/scripts/evaluate_cartpole.py) to evaluate Cartpole tasks, and [`evaluate_car_quad.py`](https://github.com/lelis-research/Sparse-Policies/blob/main/src/scripts/evaluate_car_quad.py) for Parallel park.


