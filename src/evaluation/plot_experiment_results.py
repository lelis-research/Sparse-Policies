import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util

def extract_scalar_data(event_file, tag):
    """Extracts scalar values and steps from a TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags()
    print("Available tags: ", tags, "\n")  

    # --- Case 1: standard scalar summaries ---
    if tag in tags.get('scalars', []):
        events = ea.Scalars(tag)
        steps = np.array([event.step for event in events])
        values = np.array([event.value for event in events])
        return steps, values
    
    # --- Case 2: scalar stored as tensor summary (your TORCS case) ---
    if tag in tags.get('tensors', []):
        tensor_events = ea.Tensors(tag)
        steps = []
        values = []
        for e in tensor_events:
            # Convert TensorProto -> numpy array -> scalar
            arr = tensor_util.make_ndarray(e.tensor_proto)
            # assume scalar or 0-d tensor
            values.append(float(arr.squeeze()))
            steps.append(e.step)
        return np.array(steps), np.array(values, dtype=float)
    
    else:
        return np.array([]), np.array([])

def extract_scalar_data_from_file(args):
    """Standalone helper for parallel execution."""
    event_file, tag = args

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags()

    # Case 1 — standard scalar
    if tag in tags.get('scalars', []):
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events], dtype=float)
        return steps, values

    # Case 2 — tensor summary
    if tag in tags.get('tensors', []):
        tensor_events = ea.Tensors(tag)
        steps = []
        values = []
        for e in tensor_events:
            arr = tensor_util.make_ndarray(e.tensor_proto)
            values.append(float(arr.squeeze()))
            steps.append(e.step)
        return np.array(steps), np.array(values, dtype=float)

    # No tag found
    return np.array([]), np.array([])

def load_experiment_data(log_dir, tag, experiment_pattern, experiment_name):
    """Loads data from multiple runs (seeds) for a given experiment."""
    data_frames = []

    # Construct the glob pattern to match all runs for the experiment, including all seeds
    run_pattern = os.path.join(log_dir, experiment_pattern)
    run_dirs = glob.glob(run_pattern)

    print("run dirs: ", run_dirs)

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
                seed = parts[2] if len(parts) > 2 else parts[-1]
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

from concurrent.futures import ProcessPoolExecutor
def load_experiment_data_parallel(log_dir, tag, experiment_pattern, experiment_name):
    """Loads data from multiple runs (seeds) in parallel."""
    data_frames = []

    run_pattern = os.path.join(log_dir, experiment_pattern)
    run_dirs = [d for d in glob.glob(run_pattern) if os.path.isdir(d)]
    print("run dirs:", run_dirs)

    if not run_dirs:
        print(f"No runs found for pattern '{experiment_pattern}'")
        return None

    # Collect all event files
    event_files = []
    seeds = []
    for run_dir in run_dirs:
        files = glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))
        if not files:
            continue

        event_files.append(files[0])

        # Extract seed
        run_name = os.path.basename(run_dir)
        parts = run_name.split('__')
        seed = parts[2] if len(parts) > 2 else parts[-1]
        seeds.append(seed)

    # Run parsing in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            extract_scalar_data_from_file,
            [(f, tag) for f in event_files]
        ))

    # Build DataFrames
    for (steps, values), seed in zip(results, seeds):
        if steps.size == 0:
            continue

        df = pd.DataFrame({
            'Step': steps,
            'Value': values,
            'Run': f"Seed {seed}",
            'Experiment': experiment_name,
        })
        data_frames.append(df)

    if not data_frames:
        print(f"No usable data for pattern '{experiment_pattern}'")
        return None

    return pd.concat(data_frames, ignore_index=True)

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

def aggregate_across_seeds_on_grid(exp_data, smoothing_window=10, num_points=500):
    """
    Interpolate each run onto a common Step grid, then compute:
      - mean
      - 95% CI
      - Watermark: average across seeds of per-seed running best-so-far

    For each seed:
      - interpolate its Value to the grid
      - compute its running max over time (monotone non-decreasing)
    Then:
      - average raw returns across seeds -> Mean, CI95
      - average running-max curves across seeds -> Watermark

    Returns:
      df with columns: Step, Mean, CI95, Watermark
      n_eff: effective number of seeds used
    """
    runs = exp_data["Run"].unique()

    # Common grid over Steps
    min_step = exp_data["Step"].min()
    max_step = exp_data["Step"].max()
    grid_steps = np.linspace(min_step, max_step, num_points)

    raw_series = []        # interpolated raw values per run
    watermark_series = []  # running-max per run

    for run in runs:
        run_data = exp_data[exp_data["Run"] == run].sort_values("Step")
        steps = run_data["Step"].to_numpy()
        vals  = run_data["Value"].to_numpy()

        # Need at least 2 points to interpolate
        if len(steps) < 2:
            continue

        # Interpolate this run onto the common grid
        interp_vals = np.interp(grid_steps, steps, vals)
        raw_series.append(interp_vals)

        # Per-seed watermark: running best-so-far over time
        wm_vals = np.maximum.accumulate(interp_vals)
        watermark_series.append(wm_vals)

    if not raw_series:
        raise ValueError("No runs had enough points to interpolate.")

    arr_raw = np.vstack(raw_series)        # shape: (n_runs_eff, num_points)
    arr_wm  = np.vstack(watermark_series)  # same shape

    n_eff = arr_raw.shape[0]

    # --- Stats for the raw returns ---
    mean_raw = arr_raw.mean(axis=0)
    std_raw  = arr_raw.std(axis=0, ddof=1)
    sem_raw  = std_raw / np.sqrt(n_eff)
    ci95_raw = 1.96 * sem_raw

    # --- Stats for the watermark curves (per-seed running best) ---
    watermark_raw    = arr_wm.mean(axis=0)
    std_wm_raw       = arr_wm.std(axis=0, ddof=1)
    sem_wm_raw       = std_wm_raw / np.sqrt(n_eff)
    ci95_wm_raw      = 1.96 * sem_wm_raw

    # Apply smoothing if requested (in grid index space)
    if smoothing_window > 1:
        mean_s = (
            pd.Series(mean_raw)
            .rolling(window=smoothing_window, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
        ci95_s = (
            pd.Series(ci95_raw)
            .rolling(window=smoothing_window, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
        watermark_s = (
            pd.Series(watermark_raw)
            .rolling(window=smoothing_window, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
        watermark_ci_s = (
            pd.Series(ci95_wm_raw)
            .rolling(window=smoothing_window, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
    else:
        mean_s = mean_raw
        ci95_s = ci95_raw
        watermark_s = watermark_raw
        watermark_ci_s = ci95_wm_raw

    df = pd.DataFrame({
        "Step":      grid_steps,
        "Mean":      mean_s,
        "CI95":      ci95_s,
        "Watermark": watermark_s,
        "WatermarkCI95": watermark_ci_s,
    })

    return df, n_eff

def plot_all_experiments_with_ci(
    data,
    smoothing_window=10,
    plot_individual_runs=False,
    metric_name="Return",
    save_path="output_plot_ci.png",
    watermark=False
):
    """
    Plot experiments with sliding-window smoothing and 95% CI across seeds.

    Assumes `data` has columns: ['Step', 'Value', 'Run', 'Experiment'].
    """

    # --- Figure + style tuned for paper plots ---
    plt.figure(figsize=(6, 4))  # smaller, more paper-like
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "font.size": 14,       # base font size
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    })

    experiments = data["Experiment"].unique()
    palette = sns.color_palette("husl", len(experiments))

    for i, experiment in enumerate(experiments):
        exp_data = data[data["Experiment"] == experiment].copy()
        runs = exp_data["Run"].unique()
        n_seeds = len(runs)

        # Report number of seeds
        print(f"Experiment '{experiment}': {n_seeds} seeds used.")

        color = palette[i]

        # --- Optional: plot individual runs (light + thin) ---
        if plot_individual_runs and not watermark:
            for run in runs:
                run_data = exp_data[exp_data["Run"] == run].sort_values("Step")
                vals = run_data["Value"]
                if smoothing_window > 1:
                    vals = (
                        vals.rolling(
                            window=smoothing_window, min_periods=1, center=True
                        ).mean()
                    )
                plt.plot(
                    run_data["Step"],
                    vals,
                    color=color,
                    alpha=0.25,
                    linewidth=0.8,
                    label="_nolegend_",
                )

        # --- Aggregate across seeds: mean + 95% CI ---

        grouped, n_eff = aggregate_across_seeds_on_grid(
            exp_data, smoothing_window=smoothing_window, num_points=500
        )
        print(f"Effective runs used after interpolation: {n_eff}")

        steps     = grouped["Step"].to_numpy()
        mean_vals = grouped["Mean"].to_numpy()
        ci_vals   = grouped["CI95"].to_numpy()
        watermark_vals = grouped["Watermark"].to_numpy()
        watermark_ci_vals= grouped["WatermarkCI95"].to_numpy()


        if not watermark:
            # Mean line
            plt.plot(
                steps,
                mean_vals,
                # label=f"{experiment} - 30 seeds",
                color=color,
                linewidth=2.0,
            )

            # 95% CI band
            plt.fill_between(
                steps,
                mean_vals - ci_vals,
                mean_vals + ci_vals,
                color=color,
                alpha=0.25,
            )
        else:
            # Watermark: running best-so-far across seeds
            if watermark:
                plt.plot(
                    steps,
                    watermark_vals,
                    color=color,
                    linewidth=2.0,
                )
                # CI band for the watermark
                plt.fill_between(
                    steps,
                    watermark_vals - watermark_ci_vals,
                    watermark_vals + watermark_ci_vals,
                    color=color,
                    alpha=0.25,        
                )


    # --- Standardized axes + layout ---
    plt.xlabel("Episodes")
    plt.ylabel("Return")         
    plt.title("Torcs - Gtrack1")
    plt.tight_layout()
    # plt.legend(frameon=True)
    plt.savefig(save_path, dpi=300)


if __name__ == '__main__':
    log_dir = 'outputs/tensorboard/final_runs/'

    # For Karel
    # tag = 'charts/episodic_return'
    # metric_name = 'Episodic Return'

    # For Parking
    # tag = 'Training/EpisodeReturnOrg'
    # metric_name = 'Episodic Return'

    # For TORCS
    tag = 'Episode/Score-Total Reward'
    metric_name = 'Episodic Return'

    # Set to True to plot individual runs
    plot_individual_runs = False  # Set to True if you want to plot individual runs

    # Smoothing window size
    smoothing_window = 10  # Adjust the window size as needed

    # List of experiments to plot
    experiments = [
        # {
        #     'pattern': 'stair/Karel_stair_climber__2000000__*__1747445166_Karel_stair_30seeds_7sets',
        #     'name': 'Karel stair_climber'
        # },
        # {
        #     'pattern': 'top/Karel_top_off__2000000__*__1747154824_Karel_top_30seeds_noFE',
        #     'name': 'Karel top_off'
        # },
        # {
        #     'pattern': 'harvester/Karel_harvester__*',
        #     'name': 'Karel harvester'
        # },
        # {
        #     'pattern': 'maze/Karel_maze__2000000__*__*_Karel_maze_30seeds_noFE',
        #     'name': 'Karel maze'
        # },
        # {
        #     'pattern': 'four/Karel_four_corner__2000000__*__*_Karel_fourcorner_30seeds_noFE',
        #     'name': 'Karel four_corner'
        # },

        # {
        #     'pattern': 'parking/Archive/Results_car_ndqn/car-train_*',
        #     'name': 'Parking'
        # },

        {
            'pattern': 'torcs/gtrack1_final/2025*',
            'name': 'Torcs - Gtrack1'
        }
        
    ] 

    all_data = []

    # Load data for all experiments
    for exp in experiments:
        experiment_pattern = exp['pattern']
        experiment_name = exp['name']

        # Load experiment data
        data = load_experiment_data_parallel(log_dir, tag, experiment_pattern, experiment_name)

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

        plot_all_experiments_with_ci(
            combined_data,
            smoothing_window=smoothing_window,
            plot_individual_runs=plot_individual_runs,
            metric_name=metric_name,
            save_path=f"output_plot_ci_final_smoothing{smoothing_window}_{exp['name']}.png",
            watermark=True
        )
    else:
        print('No data to plot.')