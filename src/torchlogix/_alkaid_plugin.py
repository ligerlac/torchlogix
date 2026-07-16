"""
alkaid plugin for logic neural networks.

Uses make_fx (aten-level tracing) instead of strict torch.fx.Tracer, so all
Python-level control flow in the export paths is supported — no need to
rewrite layers for fx-compatibility.

This module has no dependency on the `torchlogix` package beyond registering
itself under its entry-point group: tracing (make_fx + constant folding) and
aten-graph replay operate purely on `torch.fx`/aten IR and alkaid's `FVArray`.
So the tracer works for any PyTorch model built from pure-boolean/integer ops,
not just torchlogix models.

To register with alkaid (add to torchlogix's pyproject.toml):
    [project.entry-points."alir_tracer.plugins"]
    logic = "torchlogix._alkaid_plugin:LogicALIRTracer"

Usage (after registration, or with explicit framework=):
    from alkaid.converter import trace_model
    from alkaid.trace import FVArrayInput, trace

    model = TorchLogixModel()
    inp = FVArrayInput((1, *model.input_shape)).quantize(0, 1, 0)
    inp2, out = trace_model(model, inputs=inp, framework='logic')
    comb = trace(inp2, out)
"""
from __future__ import annotations
import operator

import numpy as np
import torch
from torch.fx import Node

aten = torch.ops.aten
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.const_fold import split_const_subgraphs

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
# Constant folding
#
# Delegates to torch's own generic constant-folding pass rather than
# hand-rolling a per-op-type whitelist: split_const_subgraphs does a proper
# dependency-closure walk ("foldable if it's a get_attr, or all its inputs
# are already foldable") and partitions the graph accordingly, so it covers
# any foldable op automatically - including ones this file doesn't otherwise
# know about - rather than only the handful of aten ops a hand-rolled pass
# happened to special-case.
# ---------------------------------------------------------------------------

def _fold_constant_views(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Pre-evaluate the parts of the graph that depend only on constant weight/
    connection tensors (not on the placeholder input), replacing them with
    folded attributes.

    This is *required* (not optional) before replaying the graph, because the
    wiring step (aten.index.Tensor) needs concrete integer index tensors.
    Without folding, those tensors remain as unevaluated call_function nodes
    whose result is not available at replay time.
    """
    folded = split_const_subgraphs(gm)
    folded.run_folding()
    return folded


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

@_register(operator.getitem)
def _getitem(seq, idx):
    # split_const_subgraphs bundles folded constants into one container
    # attribute (e.g. a ParameterList) and unpacks individual values via
    # plain Python getitem, rather than one get_attr node per constant.
    return seq[idx]


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


@_register(aten.unfold.default)
def _unfold(src, dim, size, step):
    # Used by OrPooling2d/OrPooling3d's export-mode forward (pool.py), which
    # calls x.unfold(...) to extract the pooling kernel's receptive-field
    # windows before OR-reducing them (F.max_pool2d only supports floats,
    # so boolean pooling needs its own window extraction).
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

class LogicALIRTracer(ALIRTracerPluginBase):
    """
    Top-level alkaid tracer for logic neural networks.

    Traces via make_fx (aten-level IR) rather than strict torch.fx.Tracer,
    so dynamic shapes and Python control flow in forward passes are fully
    supported. It works for any model whose export-mode forward pass lowers
    to the aten ops handled by `_DISPATCH` below.
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
        assert len(inputs) == 1, 'LogicALIRTracer expects a single input FVArray'
        inp_fv = inputs[0]

        # --- Trace with make_fx -----------------------------------------
        x_dummy = torch.zeros(inp_fv.shape, dtype=torch.bool)
        gm = make_fx(self.model)(x_dummy)

        # Fold constant weight / connection tensors so index ops see concrete data
        gm = _fold_constant_views(gm)

        if verbose:
            print('[LogicALIRTracer] aten graph:')
            gm.print_readable(print_output=True)

        unhandled = sorted(
            {str(node.target) for node in gm.graph.nodes
             if node.op == 'call_function' and node.target not in _DISPATCH}
        )
        if unhandled:
            raise NotImplementedError(
                f'LogicALIRTracer: no dispatch handler for aten op(s): {unhandled}. '
                f'Add a handler to _DISPATCH in _alkaid_plugin.py, or run with '
                f'verbose=True to print the full aten graph.'
            )

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
                # Every call_function target is guaranteed to be in _DISPATCH
                # at this point - checked upfront, above.
                env[node.name] = _DISPATCH[node.target](*args, **kwargs)
                continue

            # call_module / call_method: skip (should not appear after make_fx)

        assert out_fv is not None, (
            'LogicALIRTracer: failed to produce an output FVArray. '
            'Check that the model is in export mode and that all aten ops are handled.'
        )

        return {'final': out_fv}, ['final']