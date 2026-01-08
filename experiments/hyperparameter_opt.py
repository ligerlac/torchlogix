import optuna
from typing import Any
import torch


def sample_all_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Sample hyperparameters.

    :param trial:
    :return:
    """
    # Parametrization hyperparameters
    # log=True: value sampled from the log domain
    residual_param = trial.suggest_float("residual_param", 4.0, 30.0, log=True)
    parametrization_temperature = trial.suggest_float("parametrization_temperature", 1e-5, 30.0, log=True)
    parametrization_temperature_decay = trial.suggest_categorical("parametrization_temperature_decay", ["constant", "linear", "exponential"])
    parametrization_temperature_end = trial.suggest_float("parametrization_temperature_end", 1e-5, parametrization_temperature, log=True)

    # Connection hyperparameters
    connnections_temperature = trial.suggest_float("connnections_temperature", 1e-5, 30.0, log=True)
    connnections_temperature_decay = trial.suggest_categorical("connnections_temperature_decay", ["constant", "linear", "exponential"])
    connnections_temperature_end = trial.suggest_float("connnections_temperature_end", 1e-5, connnections_temperature, log=True)

    # Train hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.1, log=True)
    learning_rate_decay = trial.suggest_categorical("learning_rate_decay", ["constant", "linear", "exponential"])
    learning_rate_end = trial.suggest_float("learning_rate_end", 1e-5, learning_rate, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # Regularization hyperparameters
    weight_rescale = trial.suggest_categorical("weight_rescale", ["clip", "abs_sum", "L2", None])
    regularization_weight = trial.suggest_float("regularization_weight", 1e-5, 30.0, log=True)
    regularization_method = trial.suggest_categorical("regularization_method", [None, "abs_sum", "L2"])
    regularization_weight_increase = trial.suggest_categorical("regularization_weight_increase", ["constant", "linear", "exponential"])
    regularization_weight_end = trial.suggest_float("regularization_weight_end", regularization_weight, 30.0, log=True)


    # Display true values
    #trial.set_user_attr("gamma", 1 - one_minus_gamma)
    #trial.set_user_attr("n_steps", 2**n_steps_pow)
    #trial.set_user_attr("batch_size", 2**batch_size_pow)
    sampled_params = {
        "residual_param": residual_param,
        "parametrization_temperature": parametrization_temperature,
        "parametrization_temperature_decay": parametrization_temperature_decay,
        "parametrization_temperature_end": parametrization_temperature_end,
        "connnections_temperature": connnections_temperature,
        "connnections_temperature_decay": connnections_temperature_decay,
        "connnections_temperature_end": connnections_temperature_end,
        "learning_rate": learning_rate,
        "learning_rate_decay": learning_rate_decay,
        "learning_rate_end": learning_rate_end,
        "batch_size": batch_size,
        "weight_rescale": weight_rescale,
        "regularization_weight": regularization_weight,
        "regularization_method": regularization_method,
        "regularization_weight_increase": regularization_weight_increase,
        "regularization_weight_end": regularization_weight_end
    }
    return sampled_params
