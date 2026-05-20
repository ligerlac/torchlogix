from torch import nn


def set_export_mode(model: nn.Module, enabled: bool = True, batch_size: int = 1) -> None:
    for module in model.modules():
        if hasattr(module, "set_export_mode"):
            module.set_export_mode(enabled, batch_size=batch_size)
