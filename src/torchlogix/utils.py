from torch import nn


def set_export_mode(model: nn.Module, enabled: bool = True) -> None:
    for module in model.modules():
        if hasattr(module, "set_export_mode"):
            module.set_export_mode(enabled)
