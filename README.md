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

To train the wide maze:
```
--env_id Karel_maze --wide_maze
```

# Evaluation

## Evaluating Karel Tasks

## Evaluating Cartpole Tasks

## Evaluating Quad and Parallel Park Tasks

