#!/usr/bin/env python3
import argparse

from campaigns import load_studies, study_to_campaign_name
from train import run_training, get_parser, CallbackContext, save_best_model
from utils import save_metrics_csv, save_thresholds_csv


def _trigger_actions(parser, args):
    """Manually trigger all custom actions after programmatically setting args."""
    for action in parser._actions:
        # Skip default argparse actions (help, version, store, etc.)
        if action.__class__.__module__ == 'argparse':
            continue
        
        # Trigger any custom action subclass
        if hasattr(args, action.dest):
            value = getattr(args, action.dest)
            if value is not None:
                # Call the action as if argparse called it
                action(parser, args, value, option_string=None)
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaigns-yaml", required=True)
    parser.add_argument("--campaign-index", type=int, required=True)
    parser.add_argument("--override", nargs='*', default=[], help="Override study parameters in the form key=value")
    parser.add_argument("--append", nargs='*', default=[], help="Append values to study parameters in the form key=value1")
    args = parser.parse_args()

    overrides = args.override
    appends = args.append
    delattr(args, 'override')
    delattr(args, 'append')

    studies = load_studies(args.campaigns_yaml)
    study = studies[args.campaign_index]

    study_name = study_to_campaign_name(study)

    print(f"Starting study: {study_name}")
    print(f"With study:\n{study}")

    parser = get_parser()
    args = parser.parse_args([])

    # update all args from study
    for key, value in study.items():
        if key in args.__dict__:
            setattr(args, key, value)

    for key, value in args.__dict__.items():
        print(f"  {key}: {value}")

    # apply overrides
    for override in overrides:
        key, value = override.split('=', 1)
        if key in args.__dict__:
            current_type = type(getattr(args, key))
            if current_type is bool:
                value = value.lower() in ('true', '1', 'yes')
            else:
                value = current_type(value)
            setattr(args, key, value)

    # apply appends
    for append in appends:
        key, value = append.split('=', 1)
        if key in args.__dict__:
            current_value = getattr(args, key)
            if isinstance(current_value, list):
                current_type = type(current_value[0]) if current_value else str
                if current_type is bool:
                    value = value.lower() in ('true', '1', 'yes')
                else:
                    value = current_type(value)
                current_value.append(value)
            elif isinstance(current_value, str):
                current_value += value
            else:
                raise ValueError(f"Cannot append to non-list/non-str argument: {key}")
            setattr(args, key, current_value)

    for key, val in args.__dict__.items():
        print(f"{key}: {val}, type: {type(val)}")

    _trigger_actions(parser, args)

    call_backs = [
        lambda ctx: save_best_model(ctx, args.output),
        lambda ctx: save_metrics_csv(ctx.step, ctx.metrics, args.output),
        lambda ctx: save_thresholds_csv(ctx.step, thresholds=ctx.model[0].get_thresholds().detach(), output_path=args.output) if hasattr(ctx.model[0], "get_thresholds") else None
    ]

    result = run_training(args, callbacks=call_backs)

    print(f"Finished study: {study_name}")
    print(f"With result:\n{result}")


if __name__ == "__main__":
    main()
