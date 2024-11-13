import wandb
from args import Args
from train_ppo import main as main_training_function
import tyro

sweep_config = {
    'method': 'bayes',  # or 'random'
    'metric': {
        'name': 'charts/episodic_return',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 1e-5,
            'max': 1e-3
        },
        'clip_coef': {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.2
        },
        'ent_coef': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.02
        },
    }
}

sweep_config2 = {
    'method': 'grid',  # Use 'grid' to try all combinations of specified values
    'metric': {
        'name': 'charts/episodic_return',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-5, 1e-4, 1e-3]  
        },
        'clip_coef': {
            'values': [0.05, 0.1, 0.2] 
        },
        'ent_coef': {
            'values': [0.0, 0.005, 0.01, 0.02]  
        },
        'value_learning_rate': {    # it was useless
            'values': [5e-3, 5e-2, 5e-1]  
        }
    }
}

sweep_config3 = {
    'method': 'grid', 
    'metric': {
        'name': 'charts/episodic_return',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-5, 1e-4, 1e-3]  
        },
        'clip_coef': {
            'values': [0.05, 0.1, 0.2] 
        },
        'ent_coef': {
            'values': [0.005, 0.01, 0.02]  
        },
        'value_learning_rate': {  
            'values': [5e-3, 5e-2, 5e-1]  
        }
    }
}


sweep_id = wandb.sweep(sweep_config3, project='sweep ppo gru')

def sweep_main():
    args = tyro.cli(Args)
    args.track = True

    main_training_function(args)

wandb.agent(sweep_id, function=sweep_main)
