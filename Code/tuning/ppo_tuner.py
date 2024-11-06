import optuna
from code.tuning.tuner import Tuner 

tuner = Tuner(
    script="/Users/amirhosseinrajabpour/Documents/UofA/Thesis/neural-policy-decomposition-main/code/train_ppo.py --env_id ComboGrid_TL-BR",
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "ComboGrid_TL-BR": [-500, 0],
        # "ComboGrid_TR-BL": [0, 500],
        # "ComboGrid_BR-TL": [0, 500],
        # "ComboGrid_BL-TR": [0, 500]
    },
    params_fn=lambda trial: {
        # "env_id": "ComboGrid_TL-BR",
        "total-timesteps": 100000,
        "num-envs": 4,
        "game_width": 3,
        "hidden_size": 32,
        "log_path": "ppo_logs/combogrid/logfile",
        "options_num_epochs": 5000,
        "options_l1_lambda": 0.0005,
        "options_learning_rate": 0.1,
        "exp_name": "ComboGrid_TL-BR_optuna_test",

        "learning_rate": trial.suggest_categorical("learning_rate", [0.005, 0.001, 0.0005, 0.0001, 0.00005]),
        "clip_coef": trial.suggest_categorical("clip_coef", [0.1, 0.15, 0.2, 0.25, 0.3]),
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 0.05, 0.1, 0.15, 0.2]),
        "l1_lambda": trial.suggest_categorical("l1_lambda", [0.0, 0.005, 0.001, 0.0005, 0.0001]),

    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
    # wandb_kwargs={"project": "cleanrl"},
)
tuner.tune(
    num_trials=100,
    num_seeds=2,
)