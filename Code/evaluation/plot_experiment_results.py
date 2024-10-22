import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from tensorboard.backend.event_processing import event_accumulator

def extract_scalar_data(event_file, tag):
    """Extracts scalar values and steps from a TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    if tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        steps = np.array([event.step for event in events])
        values = np.array([event.value for event in events])
        return steps, values
    else:
        return np.array([]), np.array([])

def load_experiment_data(log_dir, tag, experiment_pattern, experiment_name):
    """Loads data from multiple runs (seeds) for a given experiment."""
    data_frames = []

    # Construct the glob pattern to match all runs for the experiment, including all seeds
    run_pattern = os.path.join(log_dir, experiment_pattern)
    run_dirs = glob.glob(run_pattern)

    if not run_dirs:
        print(f"No runs found for experiment pattern '{experiment_pattern}'.")
        return None

    for run_dir in run_dirs:
        # Ensure that the run_dir is a directory
        if not os.path.isdir(run_dir):
            continue

        event_files = glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))
        if not event_files:
            continue

        steps, values = extract_scalar_data(event_files[0], tag)
        if steps.size > 0 and values.size > 0:
            df = pd.DataFrame({'Step': steps, 'Value': values})

            # Extract seed number from the run directory name
            run_name = os.path.basename(run_dir)

            # Assuming the seed number is the fourth element when splitting by '__'
            # Adjust the index based on your directory naming convention
            try:
                parts = run_name.split('__')
                seed = parts[3]
            except IndexError:
                seed = 'UnknownSeed'

            df['Run'] = f'Seed {seed}'
            df['Experiment'] = experiment_name
            data_frames.append(df)

    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        print(f"No data found for experiment pattern '{experiment_pattern}'.")
        return None

def write_average_to_tensorboard(exp_data, log_dir, experiment_name, tag):
    """Writes the averaged data to a new TensorBoard event file."""
    from torch.utils.tensorboard import SummaryWriter

    # Compute mean and std
    grouped = exp_data.groupby('Step')['Value'].mean().reset_index()

    # Create a new run directory for the averaged data
    average_run_name = f"{experiment_name}_Average"
    average_run_dir = os.path.join(log_dir, average_run_name)
    os.makedirs(average_run_dir, exist_ok=True)

    # Initialize the SummaryWriter
    writer = SummaryWriter(log_dir=average_run_dir)

    # Write the averaged data to the TensorBoard event file
    for step, value in zip(grouped['Step'], grouped['Value']):
        writer.add_scalar(tag, value, step)

    writer.close()
    print(f"Averaged data written to TensorBoard at '{average_run_dir}'.")

def plot_all_experiments_together(data, metric_name, plot_individual_runs=False):
    """Plots individual runs and averages for all experiments on the same plot using Matplotlib."""
    plt.figure(figsize=(12, 8))
    sns.set(style='darkgrid')

    experiments = data['Experiment'].unique()
    palette = sns.color_palette("husl", len(experiments))

    # For each experiment
    for i, experiment in enumerate(experiments):
        exp_data = data[data['Experiment'] == experiment]
        runs = exp_data['Run'].unique()

        # Plot individual runs
        if plot_individual_runs:
            for run in runs:
                run_data = exp_data[exp_data['Run'] == run]
                plt.plot(run_data['Step'], run_data['Value'],
                         label=f'{experiment} - {run}',
                         color=palette[i], alpha=0.5, linewidth=1)

        # Compute mean and std
        grouped = exp_data.groupby('Step')['Value'].agg(['mean', 'std']).reset_index()
        grouped['std'] = grouped['std'].fillna(0)

        # Plot mean
        plt.plot(grouped['Step'], grouped['mean'],
                 label=f'{experiment} Average',
                 color=palette[i], linewidth=2)

        # Plot variance
        plt.fill_between(grouped['Step'],
                         grouped['mean'] - grouped['std'],
                         grouped['mean'] + grouped['std'],
                         color=palette[i], alpha=0.2)

    plt.xlabel('Global Step')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Across Experiments')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_experiments_together_plotly(data, metric_name, plot_individual_runs=False):
    """Plots individual runs and averages for all experiments on the same plot using Plotly."""
    fig = go.Figure()

    experiments = data['Experiment'].unique()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']  # Extend this list if needed

    for i, experiment in enumerate(experiments):
        exp_data = data[data['Experiment'] == experiment]
        runs = exp_data['Run'].unique()

        # Plot individual runs
        if plot_individual_runs:
            for run in runs:
                run_data = exp_data[exp_data['Run'] == run]
                fig.add_trace(go.Scatter(
                    x=run_data['Step'],
                    y=run_data['Value'],
                    mode='lines',
                    name=f'{experiment} - {run}',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                    opacity=0.5,
                    showlegend=True
                ))

        # Compute mean and std
        grouped = exp_data.groupby('Step')['Value'].agg(['mean', 'std']).reset_index()
        grouped['std'] = grouped['std'].fillna(0)

        # Plot mean line
        fig.add_trace(go.Scatter(
            x=grouped['Step'],
            y=grouped['mean'],
            mode='lines',
            name=f'{experiment} Average',
            line=dict(color=colors[i % len(colors)], width=3)
        ))

        # Add shaded area for std dev
        fig.add_trace(go.Scatter(
            x=np.concatenate([grouped['Step'], grouped['Step'][::-1]]),
            y=np.concatenate([grouped['mean'] - grouped['std'], (grouped['mean'] + grouped['std'])[::-1]]),
            fill='toself',
            fillcolor=f'rgba({(i * 50) % 256}, {(i * 80) % 256}, {(i * 110) % 256}, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title=f'{metric_name} Comparison Across Experiments',
        xaxis_title='Global Step',
        yaxis_title=metric_name,
        hovermode='x unified',
        legend_title='Experiments and Seeds',
    )

    fig.show()

if __name__ == '__main__':
    # Directory where your TensorBoard logs are stored
    log_dir = 'outputs/tensorboard/runs'  # Update this path as needed

    # The metric you want to plot
    tag = 'charts/episodic_return'
    metric_name = 'Episodic Return'

    # Set to True to plot individual runs
    plot_individual_runs = False  # Set to False if you don't want to plot individual runs

    # List of experiments to plot
    experiments = [
        {
            'pattern': 'ComboTest__150000__0.005__*__1729539342_base_ppo_comboTest_gw3*',
            'name': 'Base PPO'
        },
        {
            'pattern': 'ComboTest__150000__0.005__*__1729545086_ppo_comboTest_gw3_options*',
            'name': 'PPO with Options'
        },
        # Add more experiments as needed
    ]

    all_data = []

    # Load data for all experiments
    for exp in experiments:
        experiment_pattern = exp['pattern']
        experiment_name = exp['name']

        # Load experiment data
        data = load_experiment_data(log_dir, tag, experiment_pattern, experiment_name)

        if data is not None:
            print(f"Loaded data for experiment: {experiment_name}")
            print(f"Runs loaded: {data['Run'].unique()}")
            all_data.append(data)

            # Write the averaged data to TensorBoard
            write_average_to_tensorboard(data, log_dir, experiment_name, tag)
        else:
            print(f"No data to plot for experiment: {experiment_name}")

    if all_data:
        # Combine data from all experiments
        combined_data = pd.concat(all_data, ignore_index=True)

        # Plot all experiments together using Matplotlib
        plot_all_experiments_together(combined_data, metric_name, plot_individual_runs=plot_individual_runs)

        # Plot all experiments together using Plotly
        plot_all_experiments_together_plotly(combined_data, metric_name, plot_individual_runs=plot_individual_runs)
    else:
        print('No data to plot.')