#!/usr/bin/env python3
import argparse
import optuna
from optuna.storages import RDBStorage
from sqlalchemy import event

from campaigns import load_studies, study_to_campaign_name
from train import run_training, get_parser, CallbackContext


def _enable_wal(dbapi_conn, conn_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.close()


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


def objective(trial, study):

    suggest_from_space(trial, study["params"])

    parser = get_parser()
    args = parser.parse_args([])

    # update all args from study
    for key, value in study.items():
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

    studies = load_studies(args.campaigns_yaml)
    study = studies[args.campaign_index]

    study_name = study_to_campaign_name(study)

    print(f"Starting study: {study_name}")
    print(f"With study:\n{study}")

    storage = RDBStorage(
        url=args.storage,
        engine_kwargs={
            "connect_args": {
                "timeout": 10,  # seconds to wait on DB lock
            }
        },
    )

    event.listen(storage.engine, "connect", _enable_wal)

    optuna_study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    # Store campaign metadata (very useful)
    optuna_study.set_user_attr("study", study)

    optuna_study.optimize(
        lambda trial: objective(trial, study),
        n_trials=args.n_trials,
        catch=(optuna.TrialPruned,)
    )

    if optuna_study.best_trials:
        print("Best value:", optuna_study.best_value)
        print("Best params:", optuna_study.best_params)


if __name__ == "__main__":
    main()
