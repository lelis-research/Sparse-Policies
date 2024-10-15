import optuna
from tuner import Tuner 

tuner = Tuner(
    script="/Users/amirhosseinrajabpour/Documents/UofA/Thesis/neural-policy-decomposition-main/code/train_ppo.py --env_id ComboGrid_TL-BR --game_width 3 --hidden_size 32 --log_path ppo_logs/combogrid/logfile",
    metric="charts/episodic_length",
    metric_last_n_average_window=50,
    direction="minimize",
    aggregation_type="average",
    target_scores={
        "ComboGrid_TL-BR": [0, 500],
        # "ComboGrid_TR-BL": [0, 500],
        # "ComboGrid_BR-TL": [0, 500],
        # "ComboGrid_BL-TR": [0, 500]
    },
    params_fn=lambda trial: {
        # "learning-rate": trial.suggest_float("learning-rate", 0.0003, 0.003, log=True),
        # "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 2, 4]),
        # "update-epochs": trial.suggest_categorical("update-epochs", [1, 2, 4, 8]),
        # "num-steps": trial.suggest_categorical("num-steps", [5, 16, 32, 64, 128]),
        # "vf-coef": trial.suggest_float("vf-coef", 0, 5),
        # "max-grad-norm": trial.suggest_float("max-grad-norm", 0, 5),

        "total-timesteps": 50000,
        "num-envs": 16,

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