#!/usr/bin/env python3
"""Plot training histories from TorchLogix training runs."""

import argparse
from pathlib import Path
import pandas as pd
import glob
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.lines import Line2D
import torch
from torchlogix.models import DlgnCifar10Medium, DlgnCifar10Large, ClgnCifar10SmallRes
from torchlogix.layers import LogicDense


def get_gate_ids(path, param="raw", model_cls=ClgnCifar10SmallRes):
    model =  model_cls(parametrization=param, device="cpu")
    model.load_state_dict(
        torch.load(path, map_location=torch.device('cpu'))
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    gate_ids = [l.get_gate_ids() for l in model if isinstance(l, LogicDense)]
    return torch.cat([ids.flatten() for ids in gate_ids])


gate_names = {
    0: "False",
    1: "A&B",
    2: "A&~B",
    3: "A",
    4: "~A&B",
    5: "B",
    6: "A^B",
    7: "A|B",
    8: "~(A|B)",
    9: "~(A^B)",
    10: "~B",
    11: "B->A",
    12: "~A",
    13: "A->B",
    14: "~(A&B)",
    15: "True"
}

def main():

    hep.style.use("CMS")

    dlgn_cifar10_large_paths = {
        "raw-soft": "results/campaign_1305856/seed_0_td_0.0_arch_DlgnCifar10Large_param_raw_sampling_soft_temp_1.0_weight_residual",
        "raw-gumbelsoft": "results/campaign_1305856/seed_0_td_0.0_arch_DlgnCifar10Large_param_raw_sampling_gumbel_soft_temp_1.0_weight_residual",
        "walsh-gumbelsoft": "results/campaign_1305856/seed_0_td_0.0_arch_DlgnCifar10Large_param_walsh_sampling_gumbel_soft_temp_1.0_weight_residual",
        "walsh-soft": "results/campaign_1305856/seed_0_td_0.0_arch_DlgnCifar10Large_param_walsh_sampling_soft_temp_1.0_weight_residual",
    }

    # wall clock times from SLURM logs
    # DLGN CIFAR10 Large
    wall_times = {
        "raw-soft": "05:46:40",
        "walsh-soft": "03:21:29",
        "raw-gumbelsoft": "05:47:36",
        "walsh-gumbelsoft": " 01:49:39"
    }
    
    wall_times_in_minutes = {k: int(v.split(":")[0]) * 60 + int(v.split(":")[1]) + int(v.split(":")[2])/60 for k, v in wall_times.items()}

    dfs = {}
    for name, dir in dlgn_cifar10_large_paths.items():
        df = pd.read_csv(f"{dir}/training_metrics.csv")
        df = df.dropna()
        config = pd.read_json(f"{dir}/training_config.json", typ='series')
        dfs[name] = df

    ################### Comparison Plot ###################

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

    # scale training steps to wall clock time in minutes
    time_norm = wall_times_in_minutes["raw-soft"] / dfs["raw-soft"]["step"].max()
    # time_norm = 1
    l = ax1.plot(dfs["raw-soft"]["step"] * time_norm, dfs["raw-soft"]["val_acc_eval"], label="DLGN (Petersen et al.)", linewidth=2)
    ax1.plot(dfs["raw-soft"]["step"] * time_norm, dfs["raw-soft"]["val_acc_train"], linewidth=2,
             linestyle='dashed', color=l[0].get_color())

    time_norm = wall_times_in_minutes["walsh-gumbelsoft"] / dfs["walsh-gumbelsoft"]["step"].max()
    l = ax1.plot(dfs["walsh-gumbelsoft"]["step"] * time_norm, dfs["walsh-gumbelsoft"]["val_acc_eval"], label="WARP-LUT (Ours)", linewidth=2)
    ax1.plot(dfs["walsh-gumbelsoft"]["step"] * time_norm, dfs["walsh-gumbelsoft"]["val_acc_train"], linewidth=2,
             linestyle='dashed', color=l[0].get_color())

    ax1.set_xlim(0, 120)
    # set ticks on y-axis
    ax1.set_ylim(bottom=0)
    ax1.set_yticks([0.2, 0.4, 0.6])

    ax1.set_xlabel("Wall Clock Time (minutes)")
    ax1.set_ylabel("Validation Accuracy")

    first_legend = plt.legend(loc='lower right')
    plt.gca().add_artist(first_legend)

    # Add second legend (styles in grey)
    style_handles = [
        Line2D([0], [0], color='grey', linestyle='-', lw=2, alpha=0.6),
        Line2D([0], [0], color='grey', linestyle='--', lw=2, alpha=0.6)
    ]
    style_labels = ["Discrete", "Relaxed"]

    plt.legend(style_handles, style_labels, loc='center right')

    # add fine grid
    ax1.grid(visible=True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
    ax1.grid(visible=True, which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()

    plt.tight_layout()
    fig.savefig("plots/for-paper/dlgn_cifar10_large_comparison.png")


    ################### SAMPLING STRATEGY COMPARISON ###################

    clgn_cifar10_small_paths = {
        "raw-soft-random": "results/campaign_1309951/seed_0_td_0.0_arch_ClgnCifar10SmallRes_param_raw_sampling_soft_temp_1.0_weight_random",
        "raw-gumbelsoft-random": "results/campaign_1309951/seed_0_td_0.0_arch_ClgnCifar10SmallRes_param_raw_sampling_gumbel_soft_temp_1.0_weight_random",
        "walsh-gumbelsoft-random": "results/campaign_1309951/seed_0_td_0.0_arch_ClgnCifar10SmallRes_param_walsh_sampling_gumbel_soft_temp_1.0_weight_random",
        "walsh-soft-random": "results/campaign_1309951/seed_0_td_0.0_arch_ClgnCifar10SmallRes_param_walsh_sampling_soft_temp_1.0_weight_random",
        "raw-soft-residual": "results/campaign_1309951/seed_0_td_0.0_arch_ClgnCifar10SmallRes_param_raw_sampling_soft_temp_1.0_weight_residual",
        "raw-gumbelsoft-residual": "results/campaign_1309951/seed_0_td_0.0_arch_ClgnCifar10SmallRes_param_raw_sampling_gumbel_soft_temp_1.0_weight_residual",
        "walsh-gumbelsoft-residual": "results/campaign_1309951/seed_0_td_0.0_arch_ClgnCifar10SmallRes_param_walsh_sampling_gumbel_soft_temp_1.0_weight_residual",
        "walsh-soft-residual": "results/campaign_1309951/seed_0_td_0.0_arch_ClgnCifar10SmallRes_param_walsh_sampling_soft_temp_1.0_weight_residual",
    }

    dfs = {}
    for name, dir in clgn_cifar10_small_paths.items():
        df = pd.read_csv(f"{dir}/training_metrics.csv")
        df = df.dropna()
        config = pd.read_json(f"{dir}/training_config.json", typ='series')
        dfs[name] = df

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    l = ax1.plot(dfs["raw-gumbelsoft-random"]["step"], dfs["raw-gumbelsoft-random"]["val_acc_eval"], label="Random Init", linewidth=2)
    ax1.plot(dfs["raw-gumbelsoft-random"]["step"], dfs["raw-gumbelsoft-random"]["val_acc_train"], color=l[0].get_color(), linewidth=2, linestyle='dashed')
    l = ax1.plot(dfs["raw-gumbelsoft-residual"]["step"], dfs["raw-gumbelsoft-residual"]["val_acc_eval"], label="Residual Init", linewidth=2)
    ax1.plot(dfs["raw-gumbelsoft-residual"]["step"], dfs["raw-gumbelsoft-residual"]["val_acc_train"], color=l[0].get_color(), linewidth=2, linestyle='dashed')
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Validation Accuracy")
    first_legend = ax1.legend(loc='lower right')
    ax1.add_artist(first_legend)

    ax1.grid(visible=True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
    ax1.grid(visible=True, which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()

    l = ax2.plot(dfs["walsh-gumbelsoft-random"]["step"], dfs["walsh-gumbelsoft-random"]["val_acc_eval"], label="Random Init", linewidth=2)
    ax2.plot(dfs["walsh-gumbelsoft-random"]["step"], dfs["walsh-gumbelsoft-random"]["val_acc_train"], color=l[0].get_color(), linewidth=2, linestyle='dashed')
    l = ax2.plot(dfs["walsh-gumbelsoft-residual"]["step"], dfs["walsh-gumbelsoft-residual"]["val_acc_eval"], label="Residual Init", linewidth=2)
    ax2.plot(dfs["walsh-gumbelsoft-residual"]["step"], dfs["walsh-gumbelsoft-residual"]["val_acc_train"], color=l[0].get_color(), linewidth=2, linestyle='dashed')
    ax2.set_xlabel("Training Steps")
    first_legend = ax2.legend(loc='lower right')
    ax2.add_artist(first_legend)
    ax2.grid(visible=True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
    ax2.grid(visible=True, which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    ax2.minorticks_on()

    # Add second legend (styles in grey)
    style_handles = [
        Line2D([0], [0], color='grey', linestyle='-', lw=2, alpha=0.6),
        Line2D([0], [0], color='grey', linestyle='--', lw=2, alpha=0.6)
    ]
    style_labels = ["Discrete", "Relaxed"]

    ax1.legend(style_handles, style_labels, loc='center right')
    ax2.legend(style_handles, style_labels, loc='center right')

    ax1.set_title("DLGN (Petersen et al.)")
    ax2.set_title("WARP-LUT (Ours)")
    
    plt.tight_layout()
    fig.savefig("plots/for-paper/clgn_cifar10_small_res_samplings.png")


    ################### Gate ID Histogram Comparison ###################

    raw_random_ids = get_gate_ids(f"{clgn_cifar10_small_paths['raw-soft-random']}/best_model.pt", param="raw", model_cls=ClgnCifar10SmallRes)
    raw_residual_ids = get_gate_ids(f"{clgn_cifar10_small_paths['raw-soft-residual']}/best_model.pt", param="raw", model_cls=ClgnCifar10SmallRes)
    walsh_random_ids = get_gate_ids(f"{clgn_cifar10_small_paths['walsh-soft-random']}/best_model.pt", param="walsh", model_cls=ClgnCifar10SmallRes)
    walsh_residual_ids = get_gate_ids(f"{clgn_cifar10_small_paths['walsh-soft-residual']}/best_model.pt", param="walsh", model_cls=ClgnCifar10SmallRes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ax1.hist(raw_random_ids.numpy(), bins=range(0, 17), alpha=0.5, label="Random Init", density=True)
    ax1.hist(raw_residual_ids.numpy(), bins=range(0, 17), alpha=0.5, label="Residual Init", density=True)
    ax1.set_xlabel("Gate")
    ax1.set_ylabel("Density")
    ax1.set_xticks([i + 0.5 for i in range(16)])
    ax1.set_xticklabels([gate_names[i] for i in range(16)], rotation=45, ha='right', fontsize=15)
    ax1.set_xlim(0, 16)
    ax1.minorticks_off()
    ax1.legend()

    ax2.hist(walsh_random_ids.numpy(), bins=range(0, 17), alpha=0.5, label="Random Init", density=True)
    ax2.hist(walsh_residual_ids.numpy(), bins=range(0, 17), alpha=0.5, label="Residual Init", density=True)
    ax2.set_xlabel("Gate")
    ax2.set_xticks([i + 0.5 for i in range(16)])
    ax2.set_xticklabels([gate_names[i] for i in range(16)], rotation=45, ha='right', fontsize=15)
    ax2.set_xlim(0, 16)
    ax2.minorticks_off()
    ax2.legend()

    plt.tight_layout()
    fig.savefig("plots/for-paper/clgn_cifar10_small_res_gate_id_comparison_sampling.png")


if __name__ == "__main__":
    main()