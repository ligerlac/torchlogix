import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd

hep.style.use("CMS")


def plot_loss_histories(
    baseline_df: pd.DataFrame,
    alternative_df: pd.DataFrame,
    output=None
):

    print(f"baseline_df =\n{baseline_df}")
    print(f"alternative_df =\n{alternative_df}")
    print(f"output = {output}")

    n_baseline_runs = len(np.unique(baseline_df["run"]))
    n_alternative_runs = len(np.unique(alternative_df["run"]))
    # Generate colors using colormaps
    baseline_colors = plt.cm.viridis(np.linspace(0.6, 1.0, n_baseline_runs))
    alternative_colors = plt.cm.plasma(np.linspace(0.1, 0.6, n_alternative_runs))

    # Define solid colors for averages
    baseline_avg_color = plt.cm.viridis(0.8)  # Single color for baseline average
    alternative_avg_color = plt.cm.plasma(0.35)     # Single color for alternative average

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey="row")

    # Left column: Individual curves
    # Plot baseline curves
    for i_run in range(n_baseline_runs):
        sub_df = baseline_df[baseline_df["run"] == i_run]
        sub_df = sub_df.dropna(subset=["val_loss_train", "val_loss_eval"])
        print(f"sub_df['step'] = {sub_df['step']}")
        print(f"sub_df['val_loss_train'] = {sub_df['val_loss_train']}")
        axs[0, 0].plot(
            sub_df["step"],
            sub_df["val_loss_train"],
            color=baseline_colors[i_run],
            alpha=0.7,
            linewidth=1.5,
            label="Baseline" if i_run == 0 else ""
        )
        axs[1, 0].plot(
            sub_df["step"],
            sub_df["val_loss_eval"],
            color=baseline_colors[i_run],
            alpha=0.7,
            linewidth=1.5,
            label="Baseline" if i_run == 0 else ""
        )

    # Plot alternative curves
    for i_run in range(n_alternative_runs):
        sub_df = alternative_df[alternative_df["run"] == i_run]
        sub_df = sub_df.dropna(subset=["val_loss_train", "val_loss_eval"])
        axs[0, 0].plot(
            sub_df["step"],
            sub_df["val_loss_train"],
            color=alternative_colors[i_run],
            alpha=0.7,
            linewidth=1.5,
            label="Alternative" if i_run == 0 else ""
        )
        axs[1, 0].plot(
            sub_df["step"],
            sub_df["val_loss_eval"],
            color=alternative_colors[i_run],
            alpha=0.7,
            linewidth=1.5,
            label="Alternative" if i_run == 0 else ""
        )

    # Right column: Averages with confidence intervals
    
    # Calculate statistics for BaselineNode
    baseline_train_data = np.array([baseline_df[baseline_df["run"] == i_run]["val_loss_train"] for i_run in range(n_baseline_runs)])
    baseline_eval_data = np.array([baseline_df[baseline_df["run"] == i_run]["val_loss_eval"] for i_run in range(n_baseline_runs)])

    baseline_train_mean = np.mean(baseline_train_data, axis=0)
    baseline_train_std = np.std(baseline_train_data, axis=0)
    baseline_eval_mean = np.mean(baseline_eval_data, axis=0)
    baseline_eval_std = np.std(baseline_eval_data, axis=0)

    # Calculate statistics for alternativeNode
    alternative_train_data = np.array([alternative_df[alternative_df["run"] == i_run]["val_loss_train"] for i_run in range(n_alternative_runs)])
    alternative_eval_data = np.array([alternative_df[alternative_df["run"] == i_run]["val_loss_eval"] for i_run in range(n_alternative_runs)])

    alternative_train_mean = np.mean(alternative_train_data, axis=0)
    alternative_train_std = np.std(alternative_train_data, axis=0)
    alternative_eval_mean = np.mean(alternative_eval_data, axis=0)
    alternative_eval_std = np.std(alternative_eval_data, axis=0)

    # Create x-axis (epochs)
    epochs_train = range(len(baseline_train_mean))
    epochs_eval = range(len(baseline_eval_mean))
    
    # Plot BaselineNode averages with 68% confidence intervals (1 std dev)
    axs[0, 1].plot(epochs_train, baseline_train_mean, color=baseline_avg_color, 
                   linewidth=2, label="Baseline (avg)")
    axs[0, 1].fill_between(epochs_train, 
                           baseline_train_mean - baseline_train_std,
                           baseline_train_mean + baseline_train_std,
                           color=baseline_avg_color, alpha=0.3)
    
    axs[1, 1].plot(epochs_eval, baseline_eval_mean, color=baseline_avg_color, 
                   linewidth=2, label="Baseline (avg)")
    axs[1, 1].fill_between(epochs_eval,
                           baseline_eval_mean - baseline_eval_std,
                           baseline_eval_mean + baseline_eval_std,
                           color=baseline_avg_color, alpha=0.3)
    
    # Plot alternative averages with 68% confidence intervals
    axs[0, 1].plot(epochs_train, alternative_train_mean, color=alternative_avg_color, 
                   linewidth=2, label="Alternative (avg)")
    axs[0, 1].fill_between(epochs_train,
                           alternative_train_mean - alternative_train_std,
                           alternative_train_mean + alternative_train_std,
                           color=alternative_avg_color, alpha=0.3)

    axs[1, 1].plot(epochs_eval, alternative_eval_mean, color=alternative_avg_color, 
                   linewidth=2, label="Alternative (avg)")
    axs[1, 1].fill_between(epochs_eval,
                           alternative_eval_mean - alternative_eval_std,
                           alternative_eval_mean + alternative_eval_std,
                           color=alternative_avg_color, alpha=0.3)

    # Set labels and formatting
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 1].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss (Train Mode)')
    axs[1, 0].set_ylabel('Loss (Eval Mode)')
    axs[0, 0].set_title('Individual Runs')
    axs[0, 1].set_title('Average ± 1σ (68% CI)')

    # Create gradient legend for left column
    # Simpler alternative - create multi-segment legend entries
    legend_elements = []

    # Create Baseline legend with color segments
    baseline_lines = []
    for i in range(n_baseline_runs):
        line = plt.Line2D([0], [0], color=baseline_colors[i], linewidth=4, solid_capstyle='butt')
        baseline_lines.append(line)

    # Create composite legend entry for Baseline
    from matplotlib.legend_handler import HandlerTuple
    legend_elements.append((tuple(baseline_lines), 'Baseline'))

    # Create alternative legend with color segments
    alternative_lines = []
    for i in range(min(5, n_alternative_runs)):
        line = plt.Line2D([0], [0], color=alternative_colors[i], linewidth=4, solid_capstyle='butt')
        alternative_lines.append(line)

    legend_elements.append((tuple(alternative_lines), 'Alternative'))

    # Create legend with custom handler for left column
    axs[0, 0].legend([elem[0] for elem in legend_elements], 
            [elem[1] for elem in legend_elements],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    axs[1, 0].legend([elem[0] for elem in legend_elements], 
            [elem[1] for elem in legend_elements],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})

    # Simple legends for right column (averages)
    axs[0, 1].legend()
    axs[1, 1].legend()

    # Add grids
    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if output is None:
        plt.show()
    else:
        plt.savefig(output)
