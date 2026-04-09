from collections import OrderedDict

import torch
from torch.nn.modules.module import _IncompatibleKeys


INFERENCE_STATE_KEY = "__torchlogix_inference_only__"
INFERENCE_STATE_VERSION_KEY = "__torchlogix_inference_version__"
INFERENCE_STATE_VERSION = 1


def set_persistent_buffer(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    """Create or update a persistent buffer without changing its registration."""
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value, persistent=True)


def get_inference_state_dict(module: torch.nn.Module) -> OrderedDict:
    """Collect the minimal discrete state needed for inference-only execution."""
    state = OrderedDict()
    state[INFERENCE_STATE_KEY] = torch.tensor(True)
    state[INFERENCE_STATE_VERSION_KEY] = torch.tensor(INFERENCE_STATE_VERSION, dtype=torch.int64)

    for module_name, submodule in module.named_modules():
        if not hasattr(submodule, "_torchlogix_get_inference_state"):
            continue

        prefix = f"{module_name}." if module_name else ""
        state.update(submodule._torchlogix_get_inference_state(prefix))

    return state


def load_inference_state_dict(
    module: torch.nn.Module,
    state_dict: dict,
    strict: bool = True,
) -> _IncompatibleKeys:
    """Load an inference-only state dict into an existing module graph."""
    missing_keys: list[str] = []
    consumed_keys = set()

    for special_key in (INFERENCE_STATE_KEY, INFERENCE_STATE_VERSION_KEY):
        if special_key in state_dict:
            consumed_keys.add(special_key)

    for module_name, submodule in module.named_modules():
        if not hasattr(submodule, "_torchlogix_load_inference_state"):
            continue

        prefix = f"{module_name}." if module_name else ""
        submodule._torchlogix_load_inference_state(
            state_dict=state_dict,
            prefix=prefix,
            missing_keys=missing_keys,
            consumed_keys=consumed_keys,
        )

    unexpected_keys = sorted(key for key in state_dict.keys() if key not in consumed_keys)

    if strict and (missing_keys or unexpected_keys):
        lines = ["Error(s) in loading inference-only state_dict:"]
        if missing_keys:
            lines.append(f"\tMissing key(s): {', '.join(repr(key) for key in missing_keys)}.")
        if unexpected_keys:
            lines.append(f"\tUnexpected key(s): {', '.join(repr(key) for key in unexpected_keys)}.")
        raise RuntimeError("\n".join(lines))

    module.eval()
    return _IncompatibleKeys(missing_keys, unexpected_keys)


class InferenceStateDictMixin:
    """Adds an inference-only serialization path alongside the regular state_dict API."""

    def state_dict(self, *args, inference_only: bool = False, **kwargs):
        if inference_only:
            if args:
                raise TypeError("Positional state_dict arguments are not supported with inference_only=True.")
            return get_inference_state_dict(self)
        return super().state_dict(*args, **kwargs)

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
        inference_only: bool = False,
    ):
        if inference_only:
            del assign
            return load_inference_state_dict(self, state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
