#!/usr/bin/env python3
import argparse
import optuna

from campaigns import load_campaigns, campaign_to_study_name
from train import run_training, get_parser, CallbackContext


def condition_met(spec, params):
    if "when" in spec:
        for k, allowed in spec["when"].items():
            if params.get(k) not in allowed:
                return False

    if "when_not" in spec:
        for k, forbidden in spec["when_not"].items():
            if params.get(k) in forbidden:
                return False

    return True


def suggest_from_space(trial, space):
    params = {}

    for name, spec in space.items():

        # 1️⃣ Check condition
        if not condition_met(spec, params):
            continue

        # 2️⃣ Resolve bounds
        low = spec.get("low")
        high = spec.get("high")

        if "low_ref" in spec:
            low = params[spec["low_ref"]]
        if "high_ref" in spec:
            high = params[spec["high_ref"]]

        # 3️⃣ Sample
        ptype = spec["type"]

        print(f"Suggesting param {name} of type {ptype}, low={low}, high={high}, lowtype={type(low)}, hitype={type(high)}")

        if ptype == "float":
            low, high = float(low), float(high)
            params[name] = trial.suggest_float(
                name, low, high, log=spec.get("log", False)
            )

        elif ptype == "int":
            low, high = int(low), int(high)
            params[name] = trial.suggest_int(
                name, low, high, log=spec.get("log", False)
            )

        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(
                name, spec["choices"]
            )

        else:
            raise ValueError(f"Unknown param type: {ptype}")

    return params


def objective(trial, campaign):

    suggest_from_space(trial, campaign["params"])

    parser = get_parser()
    args = parser.parse_args([])

    # update all args from campaign
    for key, value in campaign.items():
        setattr(args, key, value)

    for key, value in trial.params.items():
        setattr(args, key, value)

    for key, val in args.__dict__.items():
        print(f"{key}: {val}, type: {type(val)}")

    def pruning_callback(ctx: CallbackContext):
        step = ctx.step
        m = ctx.metrics

        # 🔴 pruning metric
        trial.report(m["val_loss_discrete"], step)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # 🟢 auxiliary metrics
        offset = args.num_iterations
        trial.report(m["train_loss"], step + offset)
        trial.report(m["val_acc_discrete"], step + 2 * offset)
        trial.report(m["val_loss_relaxed"], step + 3 * offset)
        trial.report(m["val_acc_relaxed"], step + 4 * offset)

        return True

    result = run_training(args, callbacks=[pruning_callback])
    return result["val_loss_discrete"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", required=True)
    parser.add_argument("--campaigns-yaml", required=True)
    parser.add_argument("--campaign-index", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=100)
    args = parser.parse_args()

    campaigns = load_campaigns(args.campaigns_yaml)
    campaign = campaigns[args.campaign_index]

    study_name = campaign_to_study_name(campaign)

    print(f"Starting study: {study_name}")
    print(f"With campaign:\n{campaign}")

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
    )

    # Store campaign metadata (very useful)
    study.set_user_attr("campaign", campaign)

    study.optimize(
        lambda trial: objective(trial, campaign),
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    if study.best_trials:
        print("Best value:", study.best_value)
        print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
