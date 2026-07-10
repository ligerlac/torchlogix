"""
alkaid plugin for torchlogix export-mode models.

Uses make_fx (aten-level tracing) instead of strict torch.fx.Tracer, so all
Python-level control flow in the export paths is supported — no need to
rewrite layers for fx-compatibility.

This module has no dependency on the `torchlogix` package beyond registering
itself under its entry-point group: tracing (make_fx + constant folding) and
aten-graph replay operate purely on `torch.fx`/aten IR and alkaid's `FVArray`.
The only torchlogix-specific behavior is an opt-in duck-typed
`set_export_mode(enabled)` call on submodules that define it, so the tracer
works for any PyTorch model built from pure-boolean/integer ops, not just
torchlogix models.

To register with alkaid (add to torchlogix's pyproject.toml):
    [project.entry-points."alir_tracer.plugins"]
    torchlogix = "torchlogix._alkaid_plugin:TorchLogixALIRTracer"

Usage (after registration, or with explicit framework=):
    from alkaid.converter import trace_model
    from alkaid.trace import FVArrayInput, trace

    model = TorchLogixModel()
    inp = FVArrayInput((1, *model.input_shape)).quantize(0, 1, 0)
    inp2, out = trace_model(model, inputs=inp, framework='torchlogix')
    comb = trace(inp2, out)
"""
from __future__ import annotations
import operator

import numpy as np
import torch
from torch.fx import Node

aten = torch.ops.aten
from torch.fx.experimental.proxy_tensor import make_fx

from alkaid.trace import FVArray
from alkaid.converter.plugin import ALIRTracerPluginBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_np(t):
    """Convert a torch.Tensor to a numpy array, or pass through otherwise."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def _resolve(obj, env: dict):
    """Replace torch.fx.Node references with values from env, recursively."""
    if isinstance(obj, Node):
        return env[obj.name]
    if isinstance(obj, (list, tuple)):
        resolved = [_resolve(v, env) for v in obj]
        return type(obj)(resolved)
    if isinstance(obj, dict):
        return {k: _resolve(v, env) for k, v in obj.items()}
    if isinstance(obj, slice):
        return slice(
            _resolve(obj.start, env),
            _resolve(obj.stop, env),
            _resolve(obj.step, env),
        )
    return obj


# ---------------------------------------------------------------------------
# Constant folding (generic torch.fx / aten IR pass, no torchlogix dependency)
#
# Duplicated from torchlogix.circuit.constant_fold_views on purpose: the two
# copies are allowed to diverge independently, since this one backs a
# standalone tracer while the other backs torchlogix's own minimal Circuit IR.
# ---------------------------------------------------------------------------

def _fold_constant_views(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Pre-evaluate shape/index ops (movedim, reshape, select, slice, unbind,
    lift_fresh_copy) that operate on constant weight tensors.

    This is *required* (not optional) before replaying the graph, because the
    wiring step (aten.index.Tensor) needs concrete integer index tensors.
    Without folding, those tensors remain as unevaluated call_function nodes
    whose result is not available at replay time.
    """
    env: dict = {}

    def get_attr_value(gm, target: str):
        obj = gm
        for attr in target.split('.'):
            obj = getattr(obj, attr)
        return obj

    VIEW_OPS = {
        torch.ops.aten.movedim.int,
        torch.ops.aten.reshape.default,
        torch.ops.aten.permute.default,   # needed for conv wiring (permute → unbind chain)
        torch.ops.aten.select.int,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.moveaxis.int,
        torch.ops.aten.unbind.int,
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.aten.eq.Scalar,         # folds lut_ids == k → concrete bool mask
    }

    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            continue
        if node.op == 'get_attr':
            env[node] = get_attr_value(gm, node.target)
            continue
        if node.op == 'call_function' and node.target in VIEW_OPS:
            args_resolved = []
            all_const = True
            for a in node.args:
                if isinstance(a, torch.fx.Node):
                    if a in env:
                        args_resolved.append(env[a])
                    else:
                        all_const = False
                        break
                else:
                    args_resolved.append(a)
            if all_const:
                result = node.target(*args_resolved, **node.kwargs)
                env[node] = result
        # aten.ones.default / aten.zeros.default: constant tensor creation
        elif node.op == 'call_function' and node.target in (
                torch.ops.aten.ones.default, torch.ops.aten.zeros.default):
            size = node.args[0]
            if all(isinstance(s, int) for s in size):
                dtype  = node.kwargs.get('dtype', None)
                device = node.kwargs.get('device', torch.device('cpu'))
                if node.target == torch.ops.aten.ones.default:
                    env[node] = torch.ones(size, dtype=dtype, device=device)
                else:
                    env[node] = torch.zeros(size, dtype=dtype, device=device)
        # aten.fill_.Tensor: in-place fill through a (possibly sliced) view
        elif node.op == 'call_function' and node.target == torch.ops.aten.fill_.Tensor:
            target_arg, fill_val_arg = node.args[0], node.args[1]
            if isinstance(target_arg, torch.fx.Node) and target_arg in env:
                if isinstance(fill_val_arg, torch.fx.Node) and fill_val_arg in env:
                    fill_val = env[fill_val_arg]
                else:
                    fill_val = fill_val_arg
                env[target_arg].fill_(fill_val)
                env[node] = env[target_arg]
        # aten.triu.default: upper-triangular mask of a constant tensor
        elif node.op == 'call_function' and node.target == torch.ops.aten.triu.default:
            input_node = node.args[0]
            if isinstance(input_node, torch.fx.Node) and input_node in env:
                diag = node.args[1] if len(node.args) > 1 else 0
                env[node] = torch.triu(env[input_node], diagonal=diag)
        # aten.index.Tensor has a list-of-(None|Node) as second arg;
        # fold it when every element is also a constant.
        elif node.op == 'call_function' and node.target == torch.ops.aten.index.Tensor:
            tensor_arg = node.args[0]
            indices    = node.args[1]
            if isinstance(tensor_arg, torch.fx.Node) and tensor_arg in env:
                idx_vals   = []
                all_idx_const = True
                for idx in indices:
                    if idx is None:
                        idx_vals.append(None)
                    elif isinstance(idx, torch.fx.Node):
                        if idx in env:
                            idx_vals.append(env[idx])
                        else:
                            all_idx_const = False
                            break
                    else:
                        idx_vals.append(idx)
                if all_idx_const:
                    src = env[tensor_arg]
                    env[node] = src[tuple(
                        slice(None) if v is None else v for v in idx_vals
                    )]

    for node, value in env.items():
        if node.op in ('placeholder', 'get_attr'):
            continue

        const_name = f"_folded_{node.name}"

        if isinstance(value, torch.Tensor):
            gm.register_buffer(const_name, value)
            with gm.graph.inserting_before(node):
                new_node = gm.graph.get_attr(const_name)
            node.replace_all_uses_with(new_node)

        elif isinstance(value, (tuple, list)):
            for user in list(node.users):
                if user.op == 'call_function' and user.target is operator.getitem:
                    idx = user.args[1]
                    item = value[idx]
                    item_name = f"_folded_{node.name}_{idx}"
                    if isinstance(item, torch.Tensor):
                        gm.register_buffer(item_name, item)
                        with gm.graph.inserting_before(user):
                            new_node = gm.graph.get_attr(item_name)
                        user.replace_all_uses_with(new_node)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


# ---------------------------------------------------------------------------
# aten → FVArray/numpy dispatcher
# ---------------------------------------------------------------------------

_DISPATCH: dict = {}


def _register(*targets):
    def decorator(fn):
        for t in targets:
            _DISPATCH[t] = fn
        return fn
    return decorator


# ---- Shape / indexing ops (no new circuit nodes) --------------------------

@_register(aten.index.Tensor)
def _index(src, indices):
    idx_tuple = tuple(
        slice(None) if idx is None
        else _as_np(idx) if isinstance(idx, torch.Tensor)
        else idx
        for idx in indices
    )
    return src[idx_tuple]


@_register(aten.select.int)
def _select(src, dim, idx):
    slices = [slice(None)] * src.ndim
    slices[dim] = idx
    return src[tuple(slices)]


@_register(aten.reshape.default, aten.view.default, aten._unsafe_view.default)
def _reshape(src, new_shape):
    return src.reshape(new_shape)


@_register(aten.permute.default)
def _permute(src, dims):
    return np.transpose(src, dims)


@_register(aten.flatten.using_ints)
def _flatten(src, start=0, end=-1):
    if end < 0:
        end = src.ndim + end
    new_shape = src.shape[:start] + (-1,) + src.shape[end + 1:]
    return src.reshape(new_shape)


@_register(aten.cat.default)
def _cat(tensors, dim=0):
    return np.concatenate(list(tensors), axis=dim)


@_register(aten.clone.default)
def _clone(src, memory_format=None):
    return src


@_register(aten.alias.default)
def _alias(src):
    return src


@_register(aten.unsqueeze.default)
def _unsqueeze(src, dim):
    return np.expand_dims(src, axis=dim)


@_register(aten.squeeze.dim)
def _squeeze(src, dim=None):
    if dim is None:
        return np.squeeze(src)
    return np.squeeze(src, axis=dim)


@_register(aten.flip.default)
def _flip(src, dims):
    return np.flip(src, axis=list(dims))


@_register(aten.slice.Tensor)
def _slice_tensor(src, dim=0, start=None, end=None, step=1):
    if end == 9223372036854775807:  # sys.maxsize sentinel for "end of dim"
        end = None
    slices = [slice(None)] * src.ndim
    slices[dim] = slice(start, end, step)
    return src[tuple(slices)]


@_register(aten.pad.default, aten.constant_pad_nd.default)
def _pad(src, pad_list, mode='constant', value=None):
    if all(p == 0 for p in pad_list):
        return src
    n_pairs = len(pad_list) // 2
    np_pad = [(0, 0)] * src.ndim
    for i in range(n_pairs):
        np_pad[-(i + 1)] = (int(pad_list[2 * i]), int(pad_list[2 * i + 1]))
    return np.pad(src, np_pad, mode='constant',
                  constant_values=int(value) if value is not None else 0)


# @_register(aten.unfold.default)
# def _unfold(src, dim, size, step):
#     # Build sliding-window index array
#     n = src.shape[dim]
#     n_windows = (n - size) // step + 1
#     starts = np.arange(n_windows) * step            # (n_windows,)
#     offsets = np.arange(size)                        # (size,)
#     idx = starts[:, None] + offsets[None, :]        # (n_windows, size)
#     # Index along `dim`, then move the window dim to the end
#     slices = [slice(None)] * src.ndim
#     slices[dim] = idx                               # broadcasting-safe for ndarray
#     result = src[tuple(slices)]
#     # result.shape: (..., n_windows, size, ...)  — size dim is at position dim+1
#     # torch.unfold convention: (..., n_windows, ..., size)  — size at the END
#     ndim_r = result.ndim
#     size_ax = dim + 1
#     perm = list(range(size_ax)) + list(range(size_ax + 1, ndim_r)) + [size_ax]
#     return np.transpose(result, perm)


@_register(aten.unfold.default)
def _unfold(src, dim, size, step):
    n = src.shape[dim]
    n_windows = (n - size) // step + 1

    # Guarantee C-contiguous layout so as_strided strides are standard.
    # Non-contiguous inputs (e.g. the output of a previous permute/index op)
    # would produce wrong windows if we used src.strides directly.
    src_contig = np.ascontiguousarray(src)

    new_shape = src_contig.shape[:dim] + (n_windows, size) + src_contig.shape[dim + 1:]
    s = src_contig.strides
    new_strides = s[:dim] + (s[dim] * step, s[dim]) + s[dim + 1:]

    windowed = np.lib.stride_tricks.as_strided(src_contig, shape=new_shape, strides=new_strides)

    ndim = windowed.ndim
    size_ax = dim + 1
    perm = list(range(size_ax)) + list(range(size_ax + 1, ndim)) + [size_ax]
    result = np.transpose(windowed, perm)

    # Reconstruct FVArray (as_strided / np.transpose lose the subclass).
    # Matches the pattern used in _gather_patches.
    if isinstance(src, FVArray):
        result = FVArray(np.asarray(result), src.solver_options)
    return result


@_register(aten.to.dtype)
def _to_dtype(src, dtype=None, *args, **kwargs):
    return src  # identity: boolean circuits stay boolean


# ---- Boolean binary / unary ops ------------------------------------------

@_register(aten.__and__.Tensor, aten.bitwise_and.Tensor)
def _band(a, b): return a & b

@_register(aten.__or__.Tensor, aten.bitwise_or.Tensor)
def _bor(a, b): return a | b

@_register(aten.__xor__.Tensor, aten.bitwise_xor.Tensor)
def _bxor(a, b): return a ^ b

@_register(aten.bitwise_not.default)
def _bnot(a): return ~a


@_register(aten.bitwise_and.Scalar)
def _band_scalar(a, s):
    return np.zeros_like(a) if not bool(s) else a

@_register(aten.bitwise_or.Scalar)
def _bor_scalar(a, s):
    return np.ones_like(a) if bool(s) else a


# ---- Tensor creation ------------------------------------------------------

@_register(aten.zeros_like.default)
def _zeros_like(ref, *args, **kwargs):
    return np.zeros_like(ref, dtype=bool)

@_register(aten.ones_like.default)
def _ones_like(ref, *args, **kwargs):
    return np.ones_like(ref, dtype=bool)

@_register(aten.empty_like.default)
def _empty_like(ref, *args, **kwargs):
    return np.zeros_like(ref, dtype=bool)  # initialise buffer to CONST_FALSE


# ---- Comparison / conditional ---------------------------------------------

@_register(aten.eq.Scalar)
def _eq_scalar(a, scalar):
    a_np = _as_np(a) if isinstance(a, torch.Tensor) else a
    return a_np == scalar


@_register(aten.where.self)
def _where(condition, x, y):
    cond = _as_np(condition) if isinstance(condition, torch.Tensor) else condition
    return np.where(cond, x, y)


# ---- Reductions -----------------------------------------------------------

@_register(aten.sum.dim_IntList)
def _sum(src, dim_list, keepdim=False):
    axes = tuple(dim_list) if isinstance(dim_list, (list, tuple)) else (dim_list,)
    return np.sum(src, axis=axes, keepdims=keepdim)


# ---- Scalar arithmetic (tau/beta adjustments in GroupSum) ----------------
# These operate on the FVArray output of a sum reduction, which by then holds
# integer-valued symbolic scalars rather than pure boolean bits.

@_register(aten.add.Tensor, aten.add.Scalar)
def _add(a, b): return a + b

@_register(aten.mul.Tensor, aten.mul.Scalar)
def _mul(a, b): return a * b

@_register(aten.div.Tensor, aten.div.Scalar)
def _div(a, b): return a / b

@_register(aten.sub.Tensor, aten.sub.Scalar)
def _sub(a, b): return a - b


# ---- In-place scatter (for-loop LUT dispatch path) -----------------------
# When apply_luts_export_mode is written as a for-loop (rather than the
# torch.where cascade), make_fx emits index_put_ nodes. We implement this
# by writing into a mutable copy of result.

@_register(aten.index_put_.default)
def _index_put_(result, indices, value, accumulate=False):
    result_out = result if isinstance(result, np.ndarray) else np.array(result)
    idx_tuple = tuple(
        slice(None) if idx is None
        else _as_np(idx) if isinstance(idx, torch.Tensor)
        else idx
        for idx in indices
    )
    result_out[idx_tuple] = value
    return result_out


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class TorchLogixALIRTracer(ALIRTracerPluginBase):
    """
    Top-level alkaid tracer for torchlogix export-mode models.

    Traces via make_fx (aten-level IR) rather than strict torch.fx.Tracer,
    so dynamic shapes and Python control flow in export-mode forward passes
    are fully supported. Despite the name, this tracer has no dependency on
    torchlogix's own code (see module docstring) — it works for any model
    whose export-mode forward pass lowers to the aten ops handled by
    `_DISPATCH` below.
    """

    def get_input_shapes(self) -> list[tuple[int, ...]] | None:
        if hasattr(self.model, 'input_shape'):
            return [tuple(self.model.input_shape)]
        return None

    def apply_model(
        self,
        verbose: bool,
        inputs: tuple[FVArray, ...],
    ) -> tuple[dict, list[str]]:
        assert len(inputs) == 1, 'TorchLogixALIRTracer expects a single input FVArray'
        inp_fv = inputs[0]

        # --- Trace with make_fx -----------------------------------------
        self.model.eval()
        # Opt-in duck-typed export-mode switch (torchlogix layers define
        # `set_export_mode`; models that don't are traced as-is).
        for module in self.model.modules():
            if hasattr(module, 'set_export_mode'):
                module.set_export_mode(True)

        x_dummy = torch.zeros(inp_fv.shape, dtype=torch.bool)
        gm = make_fx(self.model)(x_dummy)

        # Fold constant weight / connection tensors so index ops see concrete data
        gm = _fold_constant_views(gm)

        if verbose:
            print('[TorchLogixALIRTracer] aten graph:')
            gm.print_readable(print_output=True)
            aten_ops_not_in_dispatch = {
                node.target for node in gm.graph.nodes
                if node.op == 'call_function' and node.target not in _DISPATCH
            }
            print(f'[TorchLogixALIRTracer] aten ops in graph not in dispatch table: {aten_ops_not_in_dispatch}')

        # --- Replay aten graph on FVArray --------------------------------
        env: dict[str, object] = {}
        out_fv: FVArray | None = None

        for node in gm.graph.nodes:
            if node.op == 'placeholder':
                env[node.name] = inp_fv
                continue

            if node.op == 'get_attr':
                obj = gm
                for part in node.target.split('.'):
                    obj = getattr(obj, part)
                env[node.name] = obj  # concrete torch.Tensor (weights, indices, lut_ids, …)
                continue

            if node.op == 'output':
                ret = node.args[0]
                if isinstance(ret, Node):
                    out_fv = env[ret.name]
                elif isinstance(ret, (list, tuple)):
                    out_fv = next(env[r.name] for r in ret if isinstance(r, Node))
                continue

            if node.op == 'call_function':
                args = _resolve(node.args, env)
                kwargs = _resolve(node.kwargs, env)
                handler = _DISPATCH.get(node.target)
                if handler is not None:
                    result = handler(*args, **kwargs)
                    env[node.name] = result
                elif verbose:
                    print(f'  [skip] {node.target}')
                continue

            # call_module / call_method: skip (should not appear after make_fx)

        assert out_fv is not None, (
            'TorchLogixALIRTracer: failed to produce an output FVArray. '
            'Check that the model is in export mode and that all aten ops are handled.'
        )

        return {'final': out_fv}, ['final']