#!/usr/bin/env python3
"""Plot training histories from TorchLogix training runs."""

import argparse
from pathlib import Path
import pandas as pd
import glob

from utils import plot_loss_histories


def make_plots(args):
    baseline_dfs, alternative_dfs = [], []
    for i, df_path in enumerate(glob.glob(str(args.baseline_path))):
        df = pd.read_csv(df_path)
        df['run'] = i  # Add run identifier
        baseline_dfs.append(df)
    baseline_df = pd.concat(baseline_dfs, ignore_index=True)
    for i, df_path in enumerate(glob.glob(str(args.alternative_path))):
        df = pd.read_csv(df_path)
        df['run'] = i  # Add run identifier
        alternative_dfs.append(df)
    alternative_df = pd.concat(alternative_dfs, ignore_index=True)

    print(f"baseline_df =\n{baseline_df}")
    print(f"alternative_df =\n{alternative_df}")

    plot_loss_histories(baseline_df, alternative_df, output=args.output)


def main():
    parser = argparse.ArgumentParser(description="Plot TorchLogix training histories")

    # Input options
    parser.add_argument(
        "--baseline-path", type=Path, required=True,
        help="Path to baseline training run .csv files (can include wildcards for multiple runs)"
    )
    parser.add_argument(
        "--alternative-path", type=Path, required=True,
        help="Path to alternative training run .csv files (can include wildcards for multiple runs)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to output the comparison plot. Interactive display if <None>"
    )

    args = parser.parse_args()
    make_plots(args)


if __name__ == "__main__":
    main()
