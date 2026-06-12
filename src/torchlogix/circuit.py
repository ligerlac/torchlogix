"""
circuit_ir.py
=============
Serialize a folded FX graph (from constant_fold_views) into:
  1. A flat gate-list IR with globally unique IDs
  2. A Lisp-like text representation of that IR
  3. A self-contained C file that evaluates the circuit

Gate ID space
-------------
  0 .. n_inputs-1          : input wires  (flat index into bool input array)
  n_inputs .. n_inputs+G-1 : gate outputs (in topological order)

Each gate:
  GateOp  : AND | OR | XOR | NOT | WIRE | CONST_FALSE | CONST_TRUE
  in0, in1: gate IDs for the two inputs (-1 if unused, e.g. NOT / WIRE / CONST)
  out_id  : this gate's unique ID
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar
from datetime import datetime
import operator
import json
import numpy as np

import torch
import torch.fx


# ---------------------------------------------------------------------------
# Gate model
# ---------------------------------------------------------------------------

class GateOp(Enum):
    CONST_FALSE = auto()
    CONST_TRUE  = auto()
    WIRE        = auto()   # pass-through of one input
    NOT         = auto()
    AND         = auto()
    OR          = auto()
    XOR         = auto()
    NAND        = auto()
    NOR         = auto()
    XNOR        = auto()
    AND_NOT_B   = auto()   # a & ~b
    AND_NOT_A   = auto()   # ~a & b
    OR_NOT_B    = auto()   # a | ~b   (b->a)
    OR_NOT_A    = auto()   # ~a | b   (a->b)
    NOT_A       = auto()   # alias for NOT with explicit "which input"
    NOT_B       = auto()


LUT_ID_TO_GATE = {
    0:  (GateOp.CONST_FALSE, False, False),  # (op, uses_a, uses_b)
    1:  (GateOp.AND,         True,  True),
    2:  (GateOp.AND_NOT_B,   True,  True),
    3:  (GateOp.WIRE,        True,  False),
    4:  (GateOp.AND_NOT_A,   True,  True),
    5:  (GateOp.WIRE,        False, True),   # wire b
    6:  (GateOp.XOR,         True,  True),
    7:  (GateOp.OR,          True,  True),
    8:  (GateOp.NOR,         True,  True),
    9:  (GateOp.XNOR,        True,  True),
    10: (GateOp.NOT,         False, True),   # not b
    11: (GateOp.OR_NOT_B,    True,  True),
    12: (GateOp.NOT,         True,  False),  # not a
    13: (GateOp.OR_NOT_A,    True,  True),
    14: (GateOp.NAND,        True,  True),
    15: (GateOp.CONST_TRUE,  False, False),
}

GATE_OP_LISP = {
    GateOp.CONST_FALSE: "(const false)",
    GateOp.CONST_TRUE:  "(const true)",
    GateOp.WIRE:        "(wire {a})",
    GateOp.NOT:         "(not {a})",
    GateOp.AND:         "(and {a} {b})",
    GateOp.OR:          "(or {a} {b})",
    GateOp.XOR:         "(xor {a} {b})",
    GateOp.NAND:        "(nand {a} {b})",
    GateOp.NOR:         "(nor {a} {b})",
    GateOp.XNOR:        "(xnor {a} {b})",
    GateOp.AND_NOT_B:   "(and {a} (not {b}))",
    GateOp.AND_NOT_A:   "(and (not {a}) {b})",
    GateOp.OR_NOT_B:    "(or {a} (not {b}))",
    GateOp.OR_NOT_A:    "(or (not {a}) {b})",
    GateOp.NOT_A:       "(not {a})",
    GateOp.NOT_B:       "(not {b})",
}

# Verilog expression templates; a/b are Verilog signal name strings
GATE_OP_VERILOG = {
    GateOp.CONST_FALSE: "1'b0",
    GateOp.CONST_TRUE:  "1'b1",
    GateOp.WIRE:        "{a}",
    GateOp.NOT:         "~{a}",
    GateOp.AND:         "({a} & {b})",
    GateOp.OR:          "({a} | {b})",
    GateOp.XOR:         "({a} ^ {b})",
    GateOp.NAND:        "~({a} & {b})",
    GateOp.NOR:         "~({a} | {b})",
    GateOp.XNOR:        "~({a} ^ {b})",
    GateOp.AND_NOT_B:   "({a} & ~{b})",
    GateOp.AND_NOT_A:   "(~{a} & {b})",
    GateOp.OR_NOT_B:    "({a} | ~{b})",
    GateOp.OR_NOT_A:    "(~{a} | {b})",
    GateOp.NOT_A:       "~{a}",
    GateOp.NOT_B:       "~{b}",
}

# C expression templates; a/b are C identifier strings
GATE_OP_C = {
    GateOp.CONST_FALSE: "false",
    GateOp.CONST_TRUE:  "true",
    GateOp.WIRE:        "{a}",
    GateOp.NOT:         "(!{a})",
    GateOp.AND:         "({a} & {b})",
    GateOp.OR:          "({a} | {b})",
    GateOp.XOR:         "({a} ^ {b})",
    GateOp.NAND:        "(!({a} & {b}))",
    GateOp.NOR:         "(!({a} | {b}))",
    GateOp.XNOR:        "(!({a} ^ {b}))",
    GateOp.AND_NOT_B:   "({a} & !{b})",
    GateOp.AND_NOT_A:   "((!{a}) & {b})",
    GateOp.OR_NOT_B:    "({a} | !{b})",
    GateOp.OR_NOT_A:    "((!{a}) | {b})",
    GateOp.NOT_A:       "(!{a})",
    GateOp.NOT_B:       "(!{b})",
}

# Bit-packed C expression templates (uint{N}_t words, bitwise ops process N samples in parallel).
# Uses ~ (bitwise NOT) instead of ! (logical NOT) so all N bits are inverted.
# {T} is replaced with the concrete C type (e.g. "uint32_t") before use.
GATE_OP_C_PACKED = {
    GateOp.CONST_FALSE: "({T})0",
    GateOp.CONST_TRUE:  "~({T})0",
    GateOp.WIRE:        "{a}",
    GateOp.NOT:         "~{a}",
    GateOp.AND:         "({a} & {b})",
    GateOp.OR:          "({a} | {b})",
    GateOp.XOR:         "({a} ^ {b})",
    GateOp.NAND:        "~({a} & {b})",
    GateOp.NOR:         "~({a} | {b})",
    GateOp.XNOR:        "~({a} ^ {b})",
    GateOp.AND_NOT_B:   "({a} & ~{b})",
    GateOp.AND_NOT_A:   "(~{a} & {b})",
    GateOp.OR_NOT_B:    "({a} | ~{b})",
    GateOp.OR_NOT_A:    "(~{a} | {b})",
    GateOp.NOT_A:       "~{a}",
    GateOp.NOT_B:       "~{b}",
}

def _eval_gate_op(op: GateOp, a: bool, b: bool) -> bool:
    if op == GateOp.CONST_FALSE: return False
    if op == GateOp.CONST_TRUE:  return True
    if op == GateOp.WIRE:        return a
    if op == GateOp.NOT:         return not a
    if op == GateOp.NOT_A:       return not a
    if op == GateOp.NOT_B:       return not b
    if op == GateOp.AND:         return a and b
    if op == GateOp.OR:          return a or b
    if op == GateOp.XOR:         return a != b
    if op == GateOp.NAND:        return not (a and b)
    if op == GateOp.NOR:         return not (a or b)
    if op == GateOp.XNOR:        return a == b
    if op == GateOp.AND_NOT_B:   return a and not b
    if op == GateOp.AND_NOT_A:   return not a and b
    if op == GateOp.OR_NOT_B:    return a or not b
    if op == GateOp.OR_NOT_A:    return not a or b
    return False  # unreachable


@dataclass
class CircuitNode:
    """Base class for all circuit IR nodes.

    Every node has a unique ID, a list of input node IDs, and a list of output
    node IDs it produces.  All IDs live in the same shared counter space so the
    graph can be traversed uniformly without type-specific dispatch.
    """
    node_id:    int
    input_ids:  list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.output_ids:
            self.output_ids = [self.node_id]


@dataclass(kw_only=True)
class Gate(CircuitNode):
    """Boolean logic gate.  input_ids has 0–2 elements (no -1 sentinels).
    output_ids is always [node_id].
    """
    op:       GateOp
    layer:    int = -1
    node_idx: int = -1


@dataclass(kw_only=True)
class SumReduction(CircuitNode):
    """Reduces N boolean inputs to one integer/float scalar.
    input_ids = boolean gate IDs being summed.
    output_ids = [node_id] (one scalar output).
    """
    tau:  float = 1.0
    beta: float = 0.0

    @classmethod
    def infer_c_dtype(cls, reductions: list['SumReduction']) -> str:
        """Narrowest C output type that cannot overflow for this set of reductions."""
        for sr in reductions:
            if sr.tau != 1.0 or sr.beta != round(sr.beta):
                return "float"
        max_val = max(
            (len(sr.input_ids) + int(round(sr.beta)) for sr in reductions),
            default=0,
        )
        if max_val <= 0xFF:       return "uint8_t"
        if max_val <= 0xFFFF:     return "uint16_t"
        if max_val <= 0xFFFFFFFF: return "uint32_t"
        return "uint64_t"


@dataclass(kw_only=True)
class Binarization(CircuitNode):
    """Fans one scalar input out to n_bits boolean gate IDs via threshold comparisons.
    input_ids  = [scalar_node_id]         (exactly one scalar input)
    output_ids = n_bits boolean gate IDs  (fan-out; referenceable by downstream gates)
    thresholds = one per output bit; output_ids[k] = 1 iff input > thresholds[k]
    """
    thresholds: list[float] = field(default_factory=list)


@dataclass
class Circuit:
    n_inputs:           int
    input_shape:        list[int]
    gates:              list[Gate]         = field(default_factory=list)
    outputs:            list[int]          = field(default_factory=list)
    output_shape:       list[int]          = field(default_factory=list)
    sum_nodes:          list[SumReduction] = field(default_factory=list)
    binarization_nodes: list[Binarization] = field(default_factory=list)

    @property
    def _sum_by_id(self) -> dict[int, SumReduction]:
        return {sr.node_id: sr for sr in self.sum_nodes}

    @property
    def _node_by_output_id(self) -> dict[int, CircuitNode]:
        """Maps every output gate ID to its owning CircuitNode."""
        result: dict[int, CircuitNode] = {}
        for g in self.gates:
            result[g.node_id] = g
        for sr in self.sum_nodes:
            result[sr.node_id] = sr
        for b in self.binarization_nodes:
            for oid in b.output_ids:
                result[oid] = b
        return result
    
    def __repr__(self) -> str:
        return (
            f"Circuit(\n"
            f"  n_inputs={self.n_inputs},\n"
            f"  input_shape={self.input_shape},\n"
            f"  logic_nodes={len(self.gates)},\n"
            f"  sum_nodes={len(self.sum_nodes)},\n"
            f"  outputs={self.outputs},\n"
            f"  output_shape={self.output_shape}\n"
            f")"
        )

    # Registry for custom op handlers registered by downstream code (e.g. torchlogix layers).
    # Each entry: op_target -> fn(node, wire_map, circuit, next_id, gm, layer_idx) -> (next_id, layer_idx)
    _op_handlers: ClassVar[dict] = {}

    @classmethod
    def register_op_handler(cls, op_target, fn) -> None:
        """Register a handler for a custom FX op so from_fx_graph can process it.

        fn signature: (node, wire_map, circuit, next_id, gm, layer_idx) -> (next_id, layer_idx)
        The handler should update wire_map and circuit.gates in-place and return
        the new (next_id, layer_idx) pair.  Return None to leave them unchanged.
        """
        cls._op_handlers[op_target] = fn

    @classmethod
    def from_model(cls, model: torch.nn.Module, input_shape: list[int]) -> Circuit:
        """
        Build a Circuit from a PyTorch model by tracing and folding it.

        The model should be in export mode (if applicable) and should have
        been traced and folded with the appropriate utilities to ensure the
        FX graph is in the expected form.
        """
        from torchlogix.utils import set_export_mode  # local import — keeps Circuit standalone
        from torch.fx.experimental.proxy_tensor import make_fx
        model.eval()
        set_export_mode(model, enabled=True)
        x_dummy = torch.zeros(1, *input_shape, dtype=torch.bool)
        gm = make_fx(model)(x_dummy)
        # Trace with (1, *input_shape) so the FX graph's batch-dimension ops
        # resolve correctly; then strip the leading 1 from the stored shape.
        circuit = cls.from_fx_graph(gm, [1, *input_shape])
        circuit.input_shape = list(input_shape)
        return circuit

    @classmethod
    def from_fx_graph(cls, gm: torch.fx.GraphModule, input_shape: list[int]) -> Circuit:
        """
        Build a Circuit directly from a folded FX graph.

        This is the core logic for walking the FX graph and constructing the
        flat gate list. Assumptions (satisfied after constant_fold_views):
        - Exactly one placeholder node ('input')
        - Wiring is done via aten.index.Tensor with folded constant index tensors
        - LUT dispatch is a cascade of aten.eq + aten.where nodes
        - Layers are connected by further aten.index.Tensor nodes
        """

        gm = constant_fold_views(gm)

        n_inputs = 1
        for d in input_shape:
            n_inputs *= d

        circuit = Circuit(n_inputs=n_inputs, input_shape=list(input_shape))
        next_id = n_inputs   # gate IDs start after input IDs

        # node_name -> list[int]  (flat list of gate/input IDs produced by that node)
        wire_map: dict[str, list[int]] = {}
        # Maps node name -> list of node IDs (gate IDs or SumReduction node_ids).
        # Used to track which nodes contribute to the circuit's final outputs,
        # including mixed boolean + reduction chains.
        _output_chain: dict[str, list[int]] = {}
        # Maps SumReduction node_id -> SumReduction object for tau/beta mutation.
        _sum_by_chain_id: dict[int, SumReduction] = {}

        nodes = list(gm.graph.nodes)

        def resolve(fx_node: torch.fx.Node) -> list[int]:
            return wire_map[fx_node.name]

        # ------------------------------------------------------------------
        # Pass 1: find placeholder and seed wire_map with flat input indices
        # ------------------------------------------------------------------
        for node in nodes:
            if node.op == 'placeholder':
                wire_map[node.name] = list(range(n_inputs))
                wire_map[f'__shape_{node.name}'] = list(input_shape)
                break

        # ------------------------------------------------------------------
        # Pass 2: walk nodes in order
        # ------------------------------------------------------------------
        layer_idx = 0

        i = 0
        while i < len(nodes):
            node = nodes[i]

            # ---- get_attr: fold constant bool tensors into CONST gates; skip others ----
            if node.op == 'get_attr':
                val = _get_attr_val(gm, node)
                if isinstance(val, torch.Tensor) and val.dtype == torch.bool:
                    # Skip boolean tensors used exclusively as index masks —
                    # they are routing metadata, not circuit data. The index.Tensor
                    # and index_put_ handlers retrieve them via _get_attr_val directly.
                    _index_targets = {
                        torch.ops.aten.index.Tensor,
                        torch.ops.aten.index_put_.default,
                    }
                    if node.users and all(
                        u.op == 'call_function' and u.target in _index_targets
                        for u in node.users
                    ):
                        i += 1
                        continue
                    flat = val.flatten().tolist()
                    gate_ids = []
                    for b in flat:
                        op = GateOp.CONST_TRUE if b else GateOp.CONST_FALSE
                        g = Gate(node_id=next_id, input_ids=[], op=op, layer=layer_idx)
                        circuit.gates.append(g)
                        gate_ids.append(next_id)
                        next_id += 1
                    wire_map[node.name] = gate_ids
                    wire_map[f'__shape_{node.name}'] = list(val.shape)
                i += 1
                continue

            # ---- placeholder already handled ----
            if node.op == 'placeholder':
                i += 1
                continue

            # ---- call_module (_guards_fn etc.) : skip ----
            if node.op == 'call_module':
                i += 1
                continue

            # ---- output ----
            if node.op == 'output':
                ret = node.args[0]
                def _collect(n):
                    if not isinstance(n, torch.fx.Node):
                        return []
                    if n.name in _output_chain:
                        return list(_output_chain[n.name])
                    if n.name in wire_map:
                        return list(wire_map[n.name])
                    return []
                if isinstance(ret, torch.fx.Node):
                    circuit.outputs = _collect(ret)
                elif isinstance(ret, (tuple, list)):
                    for r in ret:
                        circuit.outputs.extend(_collect(r))
                circuit.output_shape = [len(circuit.outputs)]
                i += 1
                continue

            if node.op != 'call_function':
                i += 1
                continue

            tgt = node.target

            # ----------------------------------------------------------------
            # aten.index.Tensor  ->  wiring step
            # Gathers inputs for the next layer. The result is a tensor whose
            # first non-batch dimension is 2 (the a/b inputs), so we split it.
            # After folding, the index tensors are flat integer arrays telling
            # us which upstream gate ID to use for each position.
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.index.Tensor:
                src_node = node.args[0]
                if not isinstance(src_node, torch.fx.Node) or src_node.name not in wire_map:
                    i += 1
                    continue
                src_ids   = resolve(src_node)
                idx_list  = node.args[1]   # list of None | fx.Node

                has_non_none = any(idx is not None for idx in idx_list)
                if not has_non_none:
                    wire_map[node.name] = src_ids
                    wire_map[f'__shape_{node.name}'] = wire_map.get(
                        f'__shape_{src_node.name}', list(input_shape))
                else:
                    src_shape = wire_map.get(f'__shape_{src_node.name}', list(input_shape))
                    # Build the index tuple directly from idx_list, replacing None
                    # with slice(None).  Using the raw list avoids accidentally
                    # padding extra slice(None) args after a multi-dimensional
                    # boolean mask (which implicitly spans several dimensions).
                    index_args = []
                    ok = True
                    for idx_node in idx_list:
                        if idx_node is None:
                            index_args.append(slice(None))
                        elif isinstance(idx_node, torch.fx.Node):
                            if idx_node.op == 'get_attr':
                                index_args.append(_get_attr_val(gm, idx_node))
                            elif idx_node.name in wire_map:
                                index_args.append(torch.tensor(
                                    resolve(idx_node), dtype=torch.long))
                            else:
                                ok = False
                                break
                        else:
                            index_args.append(idx_node)
                    if ok:
                        id_tensor = torch.tensor(src_ids, dtype=torch.long).reshape(src_shape)
                        gathered  = id_tensor[tuple(index_args)]
                        wire_map[node.name] = [int(x) for x in gathered.flatten().tolist()]
                        wire_map[f'__shape_{node.name}'] = list(gathered.shape)

                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.select.int  ->  slice one dimension
            # Used to split the a/b inputs after gather
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.select.int:
                src_ids = resolve(node.args[0])
                dim = node.args[1]
                idx = node.args[2]

                src_shape = wire_map.get(f'__shape_{node.args[0].name}')
                if src_shape is None:
                    i += 1
                    continue

                id_tensor = torch.tensor(src_ids, dtype=torch.long).reshape(src_shape)
                selected  = id_tensor.select(dim, idx)
                wire_map[node.name] = [int(x) for x in selected.flatten().tolist()]
                wire_map[f'__shape_{node.name}'] = list(selected.shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.clone  ->  identity (no new gates)
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.clone.default:
                src_node = node.args[0]
                if isinstance(src_node, torch.fx.Node) and src_node.name in wire_map:
                    wire_map[node.name] = wire_map[src_node.name]
                    shape_key = f'__shape_{src_node.name}'
                    if shape_key in wire_map:
                        wire_map[f'__shape_{node.name}'] = wire_map[shape_key]
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.unsqueeze  ->  insert a size-1 dim in the shape (no new gates)
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.unsqueeze.default:
                src_node = node.args[0]
                dim      = int(node.args[1])
                if isinstance(src_node, torch.fx.Node) and src_node.name in wire_map:
                    src_ids   = resolve(src_node)
                    src_shape = list(wire_map.get(f'__shape_{src_node.name}', [len(src_ids)]))
                    new_shape = src_shape[:]
                    if dim < 0:
                        dim = len(new_shape) + 1 + dim
                    new_shape.insert(dim, 1)
                    wire_map[node.name] = src_ids
                    wire_map[f'__shape_{node.name}'] = new_shape
                    if src_node.name in _output_chain:
                        _output_chain[node.name] = _output_chain[src_node.name]
                elif isinstance(src_node, torch.fx.Node) and src_node.name in _output_chain:
                    # Float-domain unsqueeze (e.g. before FixedBinarization comparison)
                    _output_chain[node.name] = _output_chain[src_node.name]
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.squeeze.dim  ->  remove a size-1 dim (no new gates)
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.squeeze.dim:
                src_node = node.args[0]
                dim      = int(node.args[1])
                if isinstance(src_node, torch.fx.Node) and src_node.name in wire_map:
                    src_ids   = resolve(src_node)
                    src_shape = list(wire_map.get(f'__shape_{src_node.name}', [len(src_ids)]))
                    if dim < 0:
                        dim = len(src_shape) + dim
                    new_shape = [s for idx, s in enumerate(src_shape) if idx != dim or s != 1]
                    wire_map[node.name] = src_ids
                    wire_map[f'__shape_{node.name}'] = new_shape
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.flip  ->  reorder gate IDs along the flipped dims
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.flip.default:
                src_node = node.args[0]
                dims     = node.args[1]
                if isinstance(src_node, torch.fx.Node) and src_node.name in wire_map:
                    src_ids   = resolve(src_node)
                    src_shape = list(wire_map.get(f'__shape_{src_node.name}', [len(src_ids)]))
                    id_tensor = torch.tensor(src_ids, dtype=torch.long).reshape(src_shape)
                    flipped   = torch.flip(id_tensor, dims=dims)
                    wire_map[node.name] = [int(x) for x in flipped.flatten().tolist()]
                    wire_map[f'__shape_{node.name}'] = list(flipped.shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.reshape / aten.view  ->  just reshape the ID list (no new gates)
            # ----------------------------------------------------------------
            if tgt in (torch.ops.aten.reshape.default, torch.ops.aten.view.default,
                       torch.ops.aten._unsafe_view.default):
                src_ids  = resolve(node.args[0])
                new_shape = node.args[1]
                wire_map[node.name] = src_ids
                wire_map[f'__shape_{node.name}'] = list(new_shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.permute  ->  reorder axes of the ID tensor (no new gates)
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.permute.default:
                src_node = node.args[0]
                dims     = node.args[1]
                if isinstance(src_node, torch.fx.Node) and src_node.name in wire_map:
                    src_ids   = resolve(src_node)
                    src_shape = list(wire_map.get(f'__shape_{src_node.name}', [len(src_ids)]))
                    id_tensor = torch.tensor(src_ids, dtype=torch.long).reshape(src_shape)
                    permuted  = id_tensor.permute(dims)
                    wire_map[node.name] = [int(x) for x in permuted.contiguous().flatten().tolist()]
                    wire_map[f'__shape_{node.name}'] = list(permuted.shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.flatten.using_ints  ->  flatten a range of dims (no new gates)
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.flatten.using_ints:
                src_ids   = resolve(node.args[0])
                src_shape = wire_map.get(f'__shape_{node.args[0].name}')
                if src_shape is None:
                    i += 1
                    continue
                start_dim = node.args[1] if len(node.args) > 1 else 0
                end_dim   = node.args[2] if len(node.args) > 2 else -1
                id_tensor = torch.tensor(src_ids, dtype=torch.long).reshape(src_shape)
                flattened = id_tensor.flatten(start_dim, end_dim)
                wire_map[node.name] = [int(x) for x in flattened.flatten().tolist()]
                wire_map[f'__shape_{node.name}'] = list(flattened.shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.unfold  ->  sliding-window view (no new gates, just a remap)
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.unfold.default:
                src_ids   = resolve(node.args[0])
                src_shape = wire_map.get(f'__shape_{node.args[0].name}')
                if src_shape is None:
                    i += 1
                    continue
                dim  = node.args[1]
                size = node.args[2]
                step = node.args[3]
                id_tensor = torch.tensor(src_ids, dtype=torch.long).reshape(src_shape)
                unfolded  = id_tensor.unfold(dim, size, step)
                wire_map[node.name] = [int(x) for x in unfolded.flatten().tolist()]
                wire_map[f'__shape_{node.name}'] = list(unfolded.shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.pad / aten.constant_pad_nd  ->  pad (inserts const gates)
            # make_fx decomposes F.pad to aten.constant_pad_nd.default
            # ----------------------------------------------------------------
            if tgt in (torch.ops.aten.pad.default,
                       torch.ops.aten.constant_pad_nd.default):
                src_ids   = resolve(node.args[0])
                src_shape = wire_map.get(f'__shape_{node.args[0].name}')
                if src_shape is None:
                    i += 1
                    continue
                pad_list = list(node.args[1])
                if all(p == 0 for p in pad_list):
                    # No-op: zero-size padding
                    wire_map[node.name] = src_ids
                    wire_map[f'__shape_{node.name}'] = list(src_shape)
                else:
                    value    = float(node.args[3]) if len(node.args) > 3 else 0.0
                    const_op = GateOp.CONST_TRUE if value != 0.0 else GateOp.CONST_FALSE
                    id_tensor = torch.tensor(src_ids, dtype=torch.float).reshape(src_shape)
                    padded    = torch.nn.functional.pad(id_tensor, pad_list,
                                                        mode='constant', value=-1.0)
                    result_ids = []
                    for v in padded.flatten().tolist():
                        if v < 0:
                            g = Gate(node_id=next_id, input_ids=[], op=const_op, layer=layer_idx)
                            circuit.gates.append(g)
                            result_ids.append(next_id)
                            next_id += 1
                        else:
                            result_ids.append(int(v))
                    wire_map[node.name] = result_ids
                    wire_map[f'__shape_{node.name}'] = list(padded.shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.slice.Tensor  ->  slice one dimension of the ID tensor
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.slice.Tensor:
                src_ids   = resolve(node.args[0])
                src_shape = wire_map.get(f'__shape_{node.args[0].name}')
                if src_shape is None:
                    i += 1
                    continue
                dim   = node.args[1] if len(node.args) > 1 else 0
                start = node.args[2] if len(node.args) > 2 else None
                end   = node.args[3] if len(node.args) > 3 else None
                step  = node.args[4] if len(node.args) > 4 else 1
                # torch.export uses 9223372036854775807 (sys.maxsize) to mean "end of dim"
                if end == 9223372036854775807:
                    end = None
                id_tensor = torch.tensor(src_ids, dtype=torch.long).reshape(src_shape)
                slices = [slice(None)] * len(src_shape)
                slices[dim] = slice(start, end, step)
                sliced = id_tensor[tuple(slices)]
                wire_map[node.name] = [int(x) for x in sliced.flatten().tolist()]
                wire_map[f'__shape_{node.name}'] = list(sliced.shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.cat  ->  concatenate ID tensors along a dimension.
            # Both wire_map (gate IDs) and _output_chain (any node IDs) are
            # dict[str, list[int]], so mixing is handled uniformly: collect IDs
            # from whichever dict knows each input, in order.
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.cat.default:
                cat_nodes = node.args[0]
                dim = node.args[1] if len(node.args) > 1 else 0
                # Check whether any input is in _output_chain (has sum nodes)
                has_output_chain = any(
                    isinstance(n2, torch.fx.Node) and n2.name in _output_chain
                    for n2 in cat_nodes
                )
                if has_output_chain:
                    # Unified path: mix gate IDs and sum-node IDs freely
                    combined: list[int] = []
                    ok = True
                    for n2 in cat_nodes:
                        if not isinstance(n2, torch.fx.Node):
                            ok = False; break
                        if n2.name in _output_chain:
                            combined.extend(_output_chain[n2.name])
                        elif n2.name in wire_map:
                            combined.extend(wire_map[n2.name])
                        else:
                            ok = False; break
                    if ok:
                        _output_chain[node.name] = combined
                else:
                    id_tensors = []
                    ok = True
                    for n2 in cat_nodes:
                        if not (isinstance(n2, torch.fx.Node) and n2.name in wire_map):
                            ok = False
                            break
                        shape = wire_map.get(f'__shape_{n2.name}')
                        if shape is None:
                            ok = False
                            break
                        id_tensors.append(
                            torch.tensor(resolve(n2), dtype=torch.long).reshape(shape))
                    if ok and id_tensors:
                        catted = torch.cat(id_tensors, dim=dim)
                        wire_map[node.name] = [int(x) for x in catted.flatten().tolist()]
                        wire_map[f'__shape_{node.name}'] = list(catted.shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.alias  ->  identity / pass-through (no new gates)
            # Appears when a native-ops lambda returns its input unchanged,
            # e.g. WIRE A: lambda a, b: a
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.alias.default:
                src = node.args[0]
                if isinstance(src, torch.fx.Node) and src.name in wire_map:
                    wire_map[node.name] = wire_map[src.name]
                    shape = wire_map.get(f'__shape_{src.name}')
                    if shape is not None:
                        wire_map[f'__shape_{node.name}'] = list(shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.empty_like  ->  initialise result buffer (filled by index_put_)
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.empty_like.default:
                ref = node.args[0]
                if isinstance(ref, torch.fx.Node) and ref.name in wire_map:
                    ref_shape = wire_map.get(f'__shape_{ref.name}', [len(resolve(ref))])
                    n = 1
                    for d in ref_shape:
                        n *= d
                    wire_map[node.name] = [-1] * n
                    wire_map[f'__shape_{node.name}'] = list(ref_shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.bitwise_{and,or}.Scalar  ->  constant gate or pass-through
            # Emitted by _map[0] (a & False) and _map[15] (True | a).
            # ----------------------------------------------------------------
            if tgt in (torch.ops.aten.bitwise_and.Scalar,
                       torch.ops.aten.bitwise_or.Scalar):
                x_arg = node.args[0]
                scalar = node.args[1]
                if isinstance(x_arg, torch.fx.Node) and x_arg.name in wire_map:
                    x_ids = resolve(x_arg)
                    shape = wire_map.get(f'__shape_{x_arg.name}', [len(x_ids)])
                    scalar_bool = bool(scalar)
                    # Determine if this is an identity or a constant op
                    if tgt == torch.ops.aten.bitwise_and.Scalar:
                        const_op = GateOp.CONST_FALSE if not scalar_bool else None
                    else:  # bitwise_or.Scalar
                        const_op = GateOp.CONST_TRUE if scalar_bool else None
                    if const_op is None:
                        # Identity: scalar is 1 for AND or 0 for OR
                        wire_map[node.name] = x_ids
                        wire_map[f'__shape_{node.name}'] = shape
                    else:
                        gate_ids = []
                        for _ in x_ids:
                            g = Gate(node_id=next_id, input_ids=[], op=const_op, layer=layer_idx)
                            circuit.gates.append(g)
                            gate_ids.append(next_id)
                            next_id += 1
                        wire_map[node.name] = gate_ids
                        wire_map[f'__shape_{node.name}'] = shape
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.index_put_  ->  scatter gate IDs back into the result buffer
            # Used by apply_luts_vectorized_export_mode to write per-LUT results.
            # Assumes the non-None index is a bool mask (folded from aten.eq.Scalar).
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.index_put_.default:
                result_arg = node.args[0]
                indices    = node.args[1]
                value_arg  = node.args[2]
                if (isinstance(result_arg, torch.fx.Node) and result_arg.name in wire_map
                        and isinstance(value_arg, torch.fx.Node) and value_arg.name in wire_map):
                    result_ids   = list(resolve(result_arg))
                    result_shape = list(wire_map.get(f'__shape_{result_arg.name}',
                                                     [len(result_ids)]))
                    value_ids    = resolve(value_arg)
                    # Find the boolean mask among index args
                    for idx_node in indices:
                        if idx_node is None:
                            continue
                        if not isinstance(idx_node, torch.fx.Node):
                            continue
                        mask_t = (_get_attr_val(gm, idx_node) if idx_node.op == 'get_attr'
                                  else None)
                        if mask_t is None or mask_t.dtype != torch.bool:
                            continue
                        # Flatten multi-dim mask to get 1-D positions within the
                        # non-batch dims of the result buffer.
                        positions = mask_t.reshape(-1).nonzero(as_tuple=False).reshape(-1).tolist()
                        if positions:
                            batch   = result_shape[0] if result_shape else 1
                            out_dim = len(result_ids) // batch   # total non-batch elements
                            n_pos   = len(positions)
                            for b in range(batch):
                                for j, pos in enumerate(positions):
                                    result_ids[b * out_dim + pos] = value_ids[b * n_pos + j]
                        break
                    wire_map[node.name] = [int(x) for x in result_ids]
                    wire_map[f'__shape_{node.name}'] = result_shape
                i += 1
                continue

            # ----------------------------------------------------------------
            # Direct boolean binary/unary ops (e.g. OrPooling2d)
            # These appear outside LUT cascades with both operands in wire_map.
            # ----------------------------------------------------------------
            _DIRECT_BINARY_GATE = {
                torch.ops.aten.__or__.Tensor:       GateOp.OR,
                torch.ops.aten.__and__.Tensor:      GateOp.AND,
                torch.ops.aten.__xor__.Tensor:      GateOp.XOR,
                torch.ops.aten.bitwise_or.Tensor:   GateOp.OR,
                torch.ops.aten.bitwise_and.Tensor:  GateOp.AND,
                torch.ops.aten.bitwise_xor.Tensor:  GateOp.XOR,
            }
            if tgt in _DIRECT_BINARY_GATE:
                a_node_arg = node.args[0]
                b_node_arg = node.args[1]
                if (isinstance(a_node_arg, torch.fx.Node) and a_node_arg.name in wire_map
                        and isinstance(b_node_arg, torch.fx.Node) and b_node_arg.name in wire_map):
                    a_ids   = resolve(a_node_arg)
                    b_ids   = resolve(b_node_arg)
                    gate_op = _DIRECT_BINARY_GATE[tgt]
                    shape   = wire_map.get(f'__shape_{a_node_arg.name}', [len(a_ids)])
                    gate_ids = []
                    for a_id, b_id in zip(a_ids, b_ids):
                        g = Gate(node_id=next_id, input_ids=[a_id, b_id], op=gate_op, layer=layer_idx)
                        circuit.gates.append(g)
                        gate_ids.append(next_id)
                        next_id += 1
                    wire_map[node.name] = gate_ids
                    wire_map[f'__shape_{node.name}'] = list(shape)
                i += 1
                continue

            if tgt == torch.ops.aten.bitwise_not.default:
                a_node_arg = node.args[0]
                if isinstance(a_node_arg, torch.fx.Node) and a_node_arg.name in wire_map:
                    a_ids = resolve(a_node_arg)
                    shape = wire_map.get(f'__shape_{a_node_arg.name}', [len(a_ids)])
                    gate_ids = []
                    for a_id in a_ids:
                        g = Gate(node_id=next_id, input_ids=[a_id], op=GateOp.NOT, layer=layer_idx)
                        circuit.gates.append(g)
                        gate_ids.append(next_id)
                        next_id += 1
                    wire_map[node.name] = gate_ids
                    wire_map[f'__shape_{node.name}'] = list(shape)
                i += 1
                continue

            # ----------------------------------------------------------------
            # zeros_like / ones_like  ->  constant gate per position
            # ----------------------------------------------------------------
            if tgt in (torch.ops.aten.zeros_like.default,
                    torch.ops.aten.ones_like.default):
                ref_ids = resolve(node.args[0])
                op = GateOp.CONST_FALSE if tgt == torch.ops.aten.zeros_like.default \
                    else GateOp.CONST_TRUE
                gate_ids = []
                for _ in ref_ids:
                    g = Gate(node_id=next_id, input_ids=[], op=op, layer=layer_idx)
                    circuit.gates.append(g)
                    gate_ids.append(next_id)
                    next_id += 1
                wire_map[node.name] = gate_ids
                src_shape = wire_map.get(f'__shape_{node.args[0].name}', [len(ref_ids)])
                wire_map[f'__shape_{node.name}'] = src_shape
                i += 1
                continue

            # ----------------------------------------------------------------
            # SumReduction: any last-dim reduction -> one SumReduction per row.
            # The reshape before the sum is already absorbed into wire_map, so
            # x_shape encodes the full structure without needing explicit k/g.
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.sum.dim_IntList:
                x_node = node.args[0]
                dim_list = node.args[1]
                if isinstance(x_node, torch.fx.Node) and x_node.name in wire_map:
                    x_shape = wire_map.get(f'__shape_{x_node.name}', [])
                    last_dim = len(x_shape) - 1
                    if dim_list in ([-1], [last_dim]) and len(x_shape) >= 2:
                        x_ids = resolve(x_node)
                        id_tensor = torch.tensor(x_ids, dtype=torch.long).reshape(x_shape)
                        flat_outer = id_tensor.reshape(-1, x_shape[-1])
                        new_sr_ids = []
                        for row in flat_outer:
                            sr = SumReduction(
                                node_id=next_id,
                                input_ids=[int(v) for v in row],
                            )
                            circuit.sum_nodes.append(sr)
                            _sum_by_chain_id[next_id] = sr
                            new_sr_ids.append(next_id)
                            next_id += 1
                        _output_chain[node.name] = new_sr_ids
                i += 1
                continue

            if tgt in (torch.ops.aten.to.dtype, torch.ops.aten._to_copy.default):
                src = node.args[0]
                if isinstance(src, torch.fx.Node) and src.name in wire_map:
                    wire_map[node.name] = wire_map[src.name]
                    shape = wire_map.get(f'__shape_{src.name}')
                    if shape is not None:
                        wire_map[f'__shape_{node.name}'] = list(shape)
                    if src.name in _output_chain:
                        _output_chain[node.name] = _output_chain[src.name]
                elif isinstance(src, torch.fx.Node) and src.name in _output_chain:
                    _output_chain[node.name] = _output_chain[src.name]
                i += 1
                continue

            # ----------------------------------------------------------------
            # aten.gt.Tensor / aten.gt.Scalar → Binarization fan-out nodes.
            # Each SumReduction node_id in _output_chain fans out to n_bits
            # boolean gate IDs, one per threshold.
            # ----------------------------------------------------------------
            _GT_OPS = (torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Scalar)
            if tgt in _GT_OPS:
                x_node = node.args[0]
                if isinstance(x_node, torch.fx.Node) and x_node.name in _output_chain:
                    sr_ids = _output_chain[x_node.name]
                    # Retrieve thresholds
                    thr_arg = node.args[1]
                    if tgt == torch.ops.aten.gt.Scalar:
                        # Single scalar threshold → one output per SumReduction
                        global_thresholds = [float(thr_arg)]
                        per_feature = False
                    else:
                        # Tensor threshold (global shape (n_bits,) or per-feature (n_features, n_bits))
                        if isinstance(thr_arg, torch.fx.Node) and thr_arg.op == 'get_attr':
                            thr_tensor = _get_attr_val(gm, thr_arg)
                        elif isinstance(thr_arg, torch.fx.Node) and thr_arg.name in wire_map:
                            # Already evaluated tensor — skip for now
                            i += 1
                            continue
                        else:
                            i += 1
                            continue
                        thr_tensor = thr_tensor.float()
                        per_feature = thr_tensor.dim() == 2  # (n_features, n_bits)
                        if not per_feature:
                            global_thresholds = thr_tensor.flatten().tolist()

                    n_features = len(sr_ids)
                    n_bits = len(global_thresholds) if not per_feature else thr_tensor.shape[-1]
                    all_out_ids: list[int] = []
                    for f, sr_id in enumerate(sr_ids):
                        thresholds_f = (thr_tensor[f].tolist() if per_feature
                                        else global_thresholds)
                        out_ids = list(range(next_id, next_id + n_bits))
                        next_id += n_bits
                        b_node = Binarization(
                            node_id=next_id,
                            input_ids=[sr_id],
                            output_ids=out_ids,
                            thresholds=thresholds_f,
                        )
                        next_id += 1
                        circuit.binarization_nodes.append(b_node)
                        all_out_ids.extend(out_ids)

                    # Output is boolean → goes into wire_map
                    wire_map[node.name] = all_out_ids
                    wire_map[f'__shape_{node.name}'] = [1, n_features, n_bits]
                    i += 1
                    continue

            # ----------------------------------------------------------------
            # aten.gt applied to a wire_map input (e.g. FixedBinarization as
            # the first layer comparing raw float inputs to thresholds).
            # Creates one Binarization node per input position; the node's
            # input_id points to the gate/wire ID for that position.
            # ----------------------------------------------------------------
            if tgt in _GT_OPS:
                x_node = node.args[0]
                if isinstance(x_node, torch.fx.Node) and x_node.name in wire_map:
                    x_ids   = wire_map[x_node.name]
                    x_shape = wire_map.get(f'__shape_{x_node.name}', [len(x_ids)])
                    thr_arg = node.args[1]
                    if tgt == torch.ops.aten.gt.Scalar:
                        thresholds_val = [float(thr_arg)]
                    elif isinstance(thr_arg, torch.fx.Node) and thr_arg.op == 'get_attr':
                        thresholds_val = _get_attr_val(gm, thr_arg).float().flatten().tolist()
                    else:
                        i += 1; continue

                    n_bits = len(thresholds_val)
                    all_out_ids: list[int] = []
                    for gate_id in x_ids:
                        out_ids = list(range(next_id, next_id + n_bits))
                        next_id += n_bits
                        b_node = Binarization(
                            node_id=next_id,
                            input_ids=[gate_id],
                            output_ids=out_ids,
                            thresholds=thresholds_val,
                        )
                        next_id += 1
                        circuit.binarization_nodes.append(b_node)
                        all_out_ids.extend(out_ids)

                    # Replace the last dimension with n_bits (broadcasting semantics)
                    out_shape = list(x_shape[:-1]) + [n_bits]
                    wire_map[node.name] = all_out_ids
                    wire_map[f'__shape_{node.name}'] = out_shape
                    i += 1
                    continue

            # Scalar add/div/mul following a sum node (tau/beta adjustment).
            # Only the SumReduction nodes from that specific sum are updated.
            _SCALAR_OPS = (
                torch.ops.aten.add.Tensor,  torch.ops.aten.add.Scalar,
                torch.ops.aten.div.Tensor,  torch.ops.aten.div.Scalar,
                torch.ops.aten.mul.Tensor,  torch.ops.aten.mul.Scalar,
            )
            if tgt in _SCALAR_OPS:
                x_node = node.args[0]
                if (isinstance(x_node, torch.fx.Node) and x_node.name in _output_chain):
                    chain_ids = _output_chain[x_node.name]
                    scalar = node.args[1]
                    if isinstance(scalar, (int, float)):
                        for sr_id in chain_ids:
                            sr = _sum_by_chain_id.get(sr_id)
                            if sr is None:
                                continue
                            if tgt in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar):
                                sr.beta += float(scalar)
                            elif tgt in (torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar):
                                sr.tau *= float(scalar)
                            elif tgt in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar):
                                sr.tau /= float(scalar)
                    _output_chain[node.name] = chain_ids
                    i += 1
                    continue

            # ---------------------------------------------------------------------
            # Registered custom op handlers (e.g. torchlogix lut_layer / group_sum)
            # ---------------------------------------------------------------------
            if tgt in Circuit._op_handlers:
                result = Circuit._op_handlers[tgt](
                    node, wire_map, circuit, next_id, gm, layer_idx)
                if result is not None:
                    next_id, layer_idx = result
                i += 1
                continue

            # ---- skip everything else (sym_size, asserts, etc.) ----
            i += 1

        return circuit
    

    def simplify(self, n_max=1000) -> None:
        for _ in range(n_max):
            before_gates = len(self.gates)
            before_sr    = sum(len(sr.input_ids) for sr in self.sum_nodes)
            self.constant_fold_gates()
            self.constant_fold_sum_reductions()
            self.bypass_wires()
            self.fuse_not_inputs()
            self.dedup()
            self.eliminate_dead_gates()
            after_gates = len(self.gates)
            after_sr    = sum(len(sr.input_ids) for sr in self.sum_nodes)
            if after_gates == before_gates and after_sr == before_sr:
                break

    def fuse_not_inputs(self) -> None:
        """Absorb NOT gates into their single downstream consumer.

        Recognises patterns that arise when native-torch boolean ops are used
        instead of all 16 gates, and folds them into the equivalent single-gate form:
            AND(x, NOT_1use(y))  -> AND_NOT_B(x, y)
            AND(NOT_1use(x), y)  -> AND_NOT_A(x, y)
            OR(x,  NOT_1use(y))  -> OR_NOT_B(x, y)
            OR(NOT_1use(x), y)   -> OR_NOT_A(x, y)
            NOT(AND(x, y))_1use  -> NAND(x, y)
            NOT(OR(x, y))_1use   -> NOR(x, y)
            NOT(XOR(x, y))_1use  -> XNOR(x, y)

        After fusion the absorbed NOT gates become dead and are removed by the
        next eliminate_dead_gates() call (which simplify() already calls).
        """
        gate_by_id = {g.node_id: g for g in self.gates}

        # Count uses of each gate so we only absorb single-use NOTs.
        sum_by_id = self._sum_by_id
        use_count: dict[int, int] = {}
        for out_id in self.outputs:
            if out_id not in sum_by_id:
                use_count[out_id] = use_count.get(out_id, 0) + 1
        for sr in self.sum_nodes:
            for gid in sr.input_ids:
                use_count[gid] = use_count.get(gid, 0) + 1
        for g in self.gates:
            for inp in g.input_ids:
                use_count[inp] = use_count.get(inp, 0) + 1

        for g in self.gates:
            # --- AND / OR: absorb a NOT on one input ---
            if g.op in (GateOp.AND, GateOp.OR):
                in0_g = gate_by_id.get(g.input_ids[0]) if g.input_ids else None
                in1_g = gate_by_id.get(g.input_ids[1]) if len(g.input_ids) > 1 else None
                if in1_g is not None and in1_g.op == GateOp.NOT and use_count.get(g.input_ids[1], 0) == 1:
                    g.op = GateOp.AND_NOT_B if g.op == GateOp.AND else GateOp.OR_NOT_B
                    g.input_ids[1] = in1_g.input_ids[0]
                    in1_g.op = GateOp.WIRE
                elif in0_g is not None and in0_g.op == GateOp.NOT and use_count.get(g.input_ids[0], 0) == 1:
                    g.op = GateOp.AND_NOT_A if g.op == GateOp.AND else GateOp.OR_NOT_A
                    g.input_ids[0] = in0_g.input_ids[0]
                    in0_g.op = GateOp.WIRE

            # --- NOT(binary): fold into NAND / NOR / XNOR ---
            elif g.op == GateOp.NOT and g.input_ids and use_count.get(g.node_id, 0) >= 1:
                src = gate_by_id.get(g.input_ids[0])
                if src is not None and use_count.get(g.input_ids[0], 0) == 1:
                    if src.op == GateOp.AND:
                        g.op = GateOp.NAND; g.input_ids = list(src.input_ids)
                        src.op = GateOp.WIRE
                    elif src.op == GateOp.OR:
                        g.op = GateOp.NOR; g.input_ids = list(src.input_ids)
                        src.op = GateOp.WIRE
                    elif src.op == GateOp.XOR:
                        g.op = GateOp.XNOR; g.input_ids = list(src.input_ids)
                        src.op = GateOp.WIRE


    def eliminate_dead_gates(self) -> None:
        """
        Remove gates that do not contribute to the output (i.e. not on a path
        from any output ID back to an input ID).
        """
        # Backward BFS using the unified node map — all node types handled uniformly.
        node_map = self._node_by_output_id
        visited: set[int] = set()
        queue: list[int] = list(self.outputs)
        while queue:
            oid = queue.pop()
            if oid in visited or oid < self.n_inputs:
                continue
            visited.add(oid)
            node = node_map.get(oid)
            if node is None:
                continue
            queue.extend(node.input_ids)

        self.gates = [g for g in self.gates if g.node_id in visited]


    def constant_fold_sum_reductions(self) -> None:
        """
        For each SumReduction, fold CONST_TRUE / CONST_FALSE inputs directly into
        beta, leaving only genuinely variable inputs in input_ids.

        After this pass, a fully-folded reduction has input_ids == [] and beta
        encodes the entire sum; codegen emits a constant rather than a loop.
        output_ids is rebuilt from the remaining live inputs so that
        eliminate_dead_gates can remove the constant gates that were folded away.
        """
        if not self.sum_nodes:
            return
        gate_by_id = {g.node_id: g for g in self.gates}
        for sr in self.sum_nodes:
            live = []
            for gid in sr.input_ids:
                g = gate_by_id.get(gid)
                if g is not None and g.op == GateOp.CONST_TRUE:
                    sr.beta += 1.0
                elif g is not None and g.op == GateOp.CONST_FALSE:
                    pass  # contributes 0 — drop silently
                else:
                    live.append(gid)
            sr.input_ids = live
        # No output_ids to rebuild: eliminate_dead_gates reads from self.outputs directly.

    def constant_fold_gates(self) -> None:
        """
        Evaluate gates that have constant inputs and replace them with CONST_TRUE
        or CONST_FALSE gates as appropriate. This can simplify the circuit and
        reduce the number of gates.
        """
        const_val: dict[int, bool] = {}
        new_gates = []
        for g in self.gates:
            in0 = g.input_ids[0] if g.input_ids else -1
            in1 = g.input_ids[1] if len(g.input_ids) > 1 else -1
            a_c = const_val.get(in0) if in0 >= 0 else None
            b_c = const_val.get(in1) if in1 >= 0 else None
            new_op, new_in0, new_in1, known = _simplify_gate(g.op, in0, in1, a_c, b_c)
            if known is not None:
                const_val[g.node_id] = known
            new_inputs = [x for x in [new_in0, new_in1] if x >= 0]
            new_gates.append(Gate(node_id=g.node_id, input_ids=new_inputs, op=new_op, layer=g.layer, node_idx=g.node_idx))
        self.gates = new_gates


    def bypass_wires(self) -> None:
        """
        Eliminate trivial aliases:
            WIRE(x)    -> x
            NOT(NOT(x)) -> x

        Rewrites all fanins/output IDs transitively and removes dead alias gates.

        This pass is intentionally conservative and cheap.
        """

        gate_by_id = {g.node_id: g for g in self.gates}

        # ------------------------------------------------------------------
        # Build alias map
        #
        # alias[gid] = replacement_gid
        # ------------------------------------------------------------------
        alias: dict[int, int] = {}

        changed = True
        while changed:
            changed = False

            for g in self.gates:

                # ----------------------------------------------------------
                # WIRE(x) -> x
                # ----------------------------------------------------------
                if g.op == GateOp.WIRE and g.input_ids:
                    target = alias.get(g.input_ids[0], g.input_ids[0])
                    if alias.get(g.node_id) != target:
                        alias[g.node_id] = target
                        changed = True

                # ----------------------------------------------------------
                # NOT(NOT(x)) -> x
                # ----------------------------------------------------------
                elif g.op == GateOp.NOT and g.input_ids:
                    src = gate_by_id.get(g.input_ids[0])
                    if src is not None and src.op == GateOp.NOT and src.input_ids:
                        target = alias.get(src.input_ids[0], src.input_ids[0])
                        if alias.get(g.node_id) != target:
                            alias[g.node_id] = target
                            changed = True

        # ------------------------------------------------------------------
        # Resolve aliases transitively
        # ------------------------------------------------------------------
        def resolve(gid: int) -> int:
            while gid in alias:
                nxt = alias[gid]

                if nxt == gid:
                    break

                gid = nxt

            return gid

        # ------------------------------------------------------------------
        # Rewrite all fanins, outputs, and sum-node input_ids.
        # resolve() only aliases gate IDs; SumReduction/Binarization IDs pass through.
        # ------------------------------------------------------------------
        for g in self.gates:
            g.input_ids = [resolve(x) for x in g.input_ids]

        self.outputs = [resolve(oid) for oid in self.outputs]
        for sr in self.sum_nodes:
            sr.input_ids = [resolve(gid) for gid in sr.input_ids]
        # Binarization input_ids reference SumReduction node_ids → not in alias map → unchanged.

        # ------------------------------------------------------------------
        # Remove aliased gates themselves
        # ------------------------------------------------------------------
        self.gates = [g for g in self.gates if g.node_id not in alias]


    def dedup(self) -> None:
        """
        Structural hashing / common-subexpression elimination.

        Deduplicates gates with identical:
            (op, in0, in1)

        Commutative ops are canonicalized so:
            AND(a,b) == AND(b,a)

        After deduplication, all fanins/output IDs are rewritten to the
        canonical representative and duplicate gates are removed.
        """

        COMMUTATIVE = {
            GateOp.AND,
            GateOp.OR,
            GateOp.XOR,
            GateOp.NAND,
            GateOp.NOR,
            GateOp.XNOR,
        }

        # ------------------------------------------------------------------
        # Canonical representative for each structural key
        # ------------------------------------------------------------------
        canonical: dict[tuple, int] = {}

        # duplicate_gid -> canonical_gid
        replace: dict[int, int] = {}

        # ------------------------------------------------------------------
        # Build replacement map
        # ------------------------------------------------------------------
        for g in self.gates:
            inputs = list(g.input_ids)
            # Normalize commutative ops
            if g.op in COMMUTATIVE and len(inputs) == 2 and inputs[0] > inputs[1]:
                inputs[0], inputs[1] = inputs[1], inputs[0]

            key = tuple([g.op] + inputs)

            if key in canonical:
                replace[g.node_id] = canonical[key]
            else:
                canonical[key] = g.node_id

        # ------------------------------------------------------------------
        # Resolve transitively
        # ------------------------------------------------------------------
        def resolve(gid: int) -> int:
            while gid in replace:
                nxt = replace[gid]

                if nxt == gid:
                    break

                gid = nxt

            return gid

        # ------------------------------------------------------------------
        # Rewrite fanins, outputs, and sum-node input_ids.
        # ------------------------------------------------------------------
        for g in self.gates:
            g.input_ids = [resolve(x) for x in g.input_ids]
            # Keep commutative gates normalized afterward
            if g.op in COMMUTATIVE and len(g.input_ids) == 2 and g.input_ids[0] > g.input_ids[1]:
                g.input_ids[0], g.input_ids[1] = g.input_ids[1], g.input_ids[0]

        self.outputs = [resolve(oid) for oid in self.outputs]
        for sr in self.sum_nodes:
            sr.input_ids = [resolve(gid) for gid in sr.input_ids]

        # ------------------------------------------------------------------
        # Remove duplicate gates
        # ------------------------------------------------------------------
        self.gates = [g for g in self.gates if g.node_id not in replace]
        

    def compile(self, opt_level: int = 1, pack_bits: int = None) -> None:
        """
        Write C code to a temp file, compile to a shared library, and load it.
        After calling this, circuit(x) will use the compiled implementation.
        """
        import ctypes
        import subprocess
        import tempfile

        # Packed mode requires boolean inputs; circuits with input-binarization
        # take float inputs, so packing is not supported for them.
        if pack_bits is not None and any(
            b.input_ids and b.input_ids[0] < self.n_inputs
            for b in self.binarization_nodes
        ):
            pack_bits = None

        c_code = self.get_c_code(pack_bits=pack_bits)
        tmp_c = tempfile.NamedTemporaryFile(suffix='.c', delete=False, mode='w')
        tmp_c.write(c_code)
        tmp_c.close()
        so_path = tmp_c.name.replace('.c', '.so')

        result = subprocess.run(
            ['gcc', f"-O{opt_level}", '-shared', '-fPIC', '-o', so_path, tmp_c.name],
            capture_output=True, text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")

        _ctype_map = {
            None: ctypes.c_bool,
            8:    ctypes.c_uint8,
            16:   ctypes.c_uint16,
            32:   ctypes.c_uint32,
            64:   ctypes.c_uint64,
        }
        _reduction_ctype_map = {
            "float":    ctypes.c_float,
            "uint8_t":  ctypes.c_uint8,
            "uint16_t": ctypes.c_uint16,
            "uint32_t": ctypes.c_uint32,
            "uint64_t": ctypes.c_uint64,
        }
        red_outs  = [sr for sr in self.sum_nodes if sr.node_id in set(self.outputs)]
        # Use float input type when Binarization nodes take raw input wires.
        has_input_binarization = (pack_bits is None and
            any(b.input_ids and b.input_ids[0] < self.n_inputs for b in self.binarization_nodes))
        in_ctype  = ctypes.c_float if has_input_binarization else _ctype_map[pack_bits]
        # Output type depends on reductions, not on input type.
        packed_ctype = _ctype_map[pack_bits]  # packed output type (bool for None → c_bool)
        out_ctype = _reduction_ctype_map[SumReduction.infer_c_dtype(red_outs)] if red_outs else packed_ctype
        lib = ctypes.CDLL(so_path)
        lib.circuit.argtypes = [
            ctypes.POINTER(in_ctype),
            ctypes.POINTER(out_ctype),
        ]
        lib.circuit.restype = None
        lib.circuit_bench.argtypes = [
            ctypes.POINTER(in_ctype),
            ctypes.POINTER(out_ctype),
            ctypes.c_int,
        ]
        lib.circuit_bench.restype = None
        if pack_bits is not None:
            # Bool-input wrapper: packing/unpacking happens inside C, no Python loop needed.
            bool_out_ctype = out_ctype if red_outs else ctypes.c_bool
            lib.circuit_bench_bool.argtypes = [
                ctypes.POINTER(ctypes.c_bool),
                ctypes.POINTER(bool_out_ctype),
                ctypes.c_int,
            ]
            lib.circuit_bench_bool.restype = None
        self._lib = lib
        self._pack_bits = pack_bits


    def __call__(self, input: torch.Tensor, use_compiled: bool = False) -> torch.Tensor:
        """
        Evaluate the circuit on a given input tensor (shape: batch x n_inputs).

        Uses the compiled C library when available (after compile()), otherwise
        evaluates the gate list in Python.

        Returns a tensor of shape (batch, len(outputs)). The dtype is determined
        by SumReduction.infer_c_dtype over the output sum nodes: uint{N}_t when
        all tau=1, float when tau≠1, bool when there are no sum nodes in outputs.

        Attention: For a fair performance comparison, the compiled code does not
        do type conversions or looping over batches. Instead, it expects numpy
        inputs of the correct shape (batch dim must match number of packed bits)
        """
        import ctypes

        batch_size = input.shape[0]
        sum_by_id  = self._sum_by_id
        red_outs   = [sr for sr in self.sum_nodes if sr.node_id in set(self.outputs)]
        has_reductions = bool(red_outs)
        has_input_binarization = any(
            b.input_ids and b.input_ids[0] < self.n_inputs
            for b in self.binarization_nodes
        )

        if use_compiled:

            pack = self._pack_bits

            _reduction_dtype_map = {
                "float":    (np.float32,  ctypes.c_float),
                "uint8_t":  (np.uint8,    ctypes.c_uint8),
                "uint16_t": (np.uint16,   ctypes.c_uint16),
                "uint32_t": (np.uint32,   ctypes.c_uint32),
                "uint64_t": (np.uint64,   ctypes.c_uint64),
            }
            n_out = len(self.outputs)
            if not has_reductions:
                np_dtype_out = np.bool_
                c_dtype_out = ctypes.c_bool
            else:
                np_dtype_out, c_dtype_out = _reduction_dtype_map[SumReduction.infer_c_dtype(red_outs)]

            if not hasattr(self, '_lib'):
                raise RuntimeError("Circuit not compiled yet. Call compile() first.")

            # Input must be a numpy array; dtype is bool unless input-binarization nodes
            # are present, in which case float32 is expected.
            assert isinstance(input, np.ndarray), (
                "Compiled circuit expects a numpy array input, got {t}".format(t=type(input)))
            _in_np_dtype = np.float32 if has_input_binarization else np.bool_
            _in_c_type   = ctypes.c_float if has_input_binarization else ctypes.c_bool

            assert batch_size % pack == 0 if pack is not None else True, (
                f"batch_size={batch_size} must be a multiple of pack_bits={pack}"
            )

            if pack is None:
                n_iter = batch_size
                flat = input.reshape(batch_size, -1).astype(_in_np_dtype)
                in_arr = flat.ctypes.data_as(ctypes.POINTER(_in_c_type))
                out_np = np.zeros(batch_size * n_out, dtype=np_dtype_out)
                out_arr = out_np.ctypes.data_as(ctypes.POINTER(c_dtype_out))
                self._lib.circuit_bench(in_arr, out_arr, ctypes.c_int(n_iter))
                return out_np.reshape(batch_size, n_out)

            else:
                # circuit_bench_bool handles packing, circuit evaluation, and unpacking
                # entirely in C — no Python loops or intermediate arrays needed.
                flat = input.reshape(batch_size, -1)
                in_arr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
                if has_reductions:
                    out_np = np.zeros(batch_size * n_out, dtype=np_dtype_out)
                    out_arr = out_np.ctypes.data_as(ctypes.POINTER(c_dtype_out))
                else:
                    out_np = np.zeros(batch_size * n_out, dtype=np.bool_)
                    out_arr = out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
                self._lib.circuit_bench_bool(in_arr, out_arr, ctypes.c_int(batch_size))
                return out_np.reshape(batch_size, n_out)

        else:
            if self.binarization_nodes:
                # General topological evaluation — covers all three node types.
                # Node IDs are assigned from a shared counter in topological order,
                # so sorting by ID gives a correct evaluation sequence.
                node_map    = self._node_by_output_id
                gate_by_id  = {g.node_id: g for g in self.gates}
                bin_out_map = {oid: b
                               for b in self.binarization_nodes
                               for oid in b.output_ids}
                all_ids = sorted(node_map.keys())

                _torch_dtype_map = {
                    "float":    torch.float32,
                    "uint8_t":  torch.uint8,
                    "uint16_t": torch.int32,
                    "uint32_t": torch.int32,
                    "uint64_t": torch.int64,
                }
                out_dtype = _torch_dtype_map.get(
                    SumReduction.infer_c_dtype(red_outs) if red_outs else "bool",
                    torch.float32,
                ) if has_reductions else torch.bool

                results = torch.zeros(batch_size, len(self.outputs), dtype=out_dtype)
                for i in range(batch_size):
                    gate_vals: dict[int, bool] = {
                        gid: bool(input[i].flatten()[gid].item())
                        for gid in range(self.n_inputs)
                    }
                    scalar_vals: dict[int, float] = {}
                    for oid in all_ids:
                        node = node_map[oid]
                        if isinstance(node, Gate):
                            a = gate_vals.get(node.input_ids[0], False) if node.input_ids else False
                            b = gate_vals.get(node.input_ids[1], False) if len(node.input_ids) > 1 else False
                            gate_vals[oid] = _eval_gate_op(node.op, a, b)
                        elif isinstance(node, SumReduction) and oid == node.node_id:
                            s = sum(int(gate_vals.get(gid, False)) for gid in node.input_ids)
                            scalar_vals[oid] = (s + node.beta) / node.tau
                        elif isinstance(node, Binarization) and oid in bin_out_map:
                            b_node = bin_out_map[oid]
                            bit_idx = b_node.output_ids.index(oid)
                            inp = b_node.input_ids[0]
                            if inp < self.n_inputs:
                                # Raw input wire — read the actual (float) value
                                sv = float(input[i].flatten()[inp].item())
                            else:
                                sv = scalar_vals.get(inp, 0.0)
                            gate_vals[oid] = bool(sv > b_node.thresholds[bit_idx])
                    for j, out_id in enumerate(self.outputs):
                        sr = sum_by_id.get(out_id)
                        if sr is not None:
                            results[i, j] = scalar_vals.get(out_id, 0.0)
                        else:
                            results[i, j] = int(gate_vals.get(out_id, False))
                return results

            def _eval_gates(inp_row):
                gate_vals: dict[int, bool] = {
                    gid: bool(inp_row.flatten()[gid].item())
                    for gid in range(self.n_inputs)
                }
                for g in self.gates:
                    a = gate_vals.get(g.input_ids[0], False) if g.input_ids else False
                    b = gate_vals.get(g.input_ids[1], False) if len(g.input_ids) > 1 else False
                    gate_vals[g.node_id] = _eval_gate_op(g.op, a, b)
                return gate_vals

            if has_reductions:
                _torch_dtype_map = {
                    "float":    torch.float32,
                    "uint8_t":  torch.uint8,
                    "uint16_t": torch.int32,   # torch has no uint16
                    "uint32_t": torch.int32,
                    "uint64_t": torch.int64,
                }
                out_dtype = _torch_dtype_map[SumReduction.infer_c_dtype(red_outs)]
                is_int = out_dtype != torch.float32
                results = torch.zeros(batch_size, len(self.outputs), dtype=out_dtype)
                for i in range(batch_size):
                    gate_vals = _eval_gates(input[i])
                    for j, out_id in enumerate(self.outputs):
                        sr = sum_by_id.get(out_id)
                        if sr is not None:
                            s = sum(int(gate_vals.get(gid, False)) for gid in sr.input_ids)
                            results[i, j] = s + int(round(sr.beta)) if is_int else (s + sr.beta) / sr.tau
                        else:
                            results[i, j] = int(gate_vals.get(out_id, False))
                return results

            raw = torch.zeros(batch_size, len(self.outputs), dtype=torch.bool)
            for i in range(batch_size):
                gate_vals = _eval_gates(input[i])
                for k, out_id in enumerate(self.outputs):
                    raw[i, k] = gate_vals.get(out_id, False)
            return raw


    def get_c_code(self, inline_single_use: bool = False, pack_bits=None) -> str:
        """
        Generate a self-contained C function that evaluates the circuit.

        inline_single_use=True (default): gates used by only one other gate are
        inlined into their parent's expression rather than emitted as variables.
        This eliminates most temporaries and makes each output a single expression
        tree rooted at the inputs.
        """

        assert pack_bits in (None, 8, 16, 32, 64), "pack_bits must be one of None, 8, 16, 32, or 64"

        if pack_bits is not None:
            ctype     = f"uint{pack_bits}_t"
            gate_ops  = {op: tmpl.replace("{T}", ctype)
                         for op, tmpl in GATE_OP_C_PACKED.items()}
            const_false = f"({ctype})0"
        else:
            ctype     = "bool"
            gate_ops  = GATE_OP_C
            const_false = "false"

        n_in  = self.n_inputs
        n_g   = len(self.gates)
        # Derived output quantities
        sum_by_id  = self._sum_by_id
        n_total    = len(self.outputs)   # total output slots

        # Build raw[] layout: sum-reduction input gate IDs in outputs order
        raw_ids: list[int] = []
        sum_raw_offset: dict[int, tuple[int, int]] = {}  # node_id -> (start, end) in raw[]
        for out_id in self.outputs:
            sr = sum_by_id.get(out_id)
            if sr is not None and out_id not in sum_raw_offset:
                start = len(raw_ids)
                raw_ids.extend(sr.input_ids)
                sum_raw_offset[out_id] = (start, len(raw_ids))
        n_raw = len(raw_ids)

        red_outs      = [sum_by_id[oid] for oid in self.outputs if oid in sum_by_id]
        has_reductions = bool(red_outs)
        red_dtype      = SumReduction.infer_c_dtype(red_outs) if has_reductions else "bool"
        out_ctype      = red_dtype if has_reductions else ctype
        is_int_red     = has_reductions and red_dtype != "float"
        # Packed boolean-only circuits use bit-packed output (one word per output slot).
        # Circuits with reductions use per-sample layout (pack_bits values per slot).
        if has_reductions and pack_bits is not None:
            out_n = n_total * pack_bits
        else:
            out_n = n_total

        gate_by_id = {g.node_id: g for g in self.gates}

        # ------------------------------------------------------------------ #
        # Count how many times each gate's output is consumed                 #
        # ------------------------------------------------------------------ #
        use_count: dict[int, int] = {}
        if inline_single_use:
            for g in self.gates:
                use_count.setdefault(g.node_id, 0)
                for dep in g.input_ids:
                    if dep >= n_in:
                        use_count[dep] = use_count.get(dep, 0) + 1
            for out_id in self.outputs:
                sr = sum_by_id.get(out_id)
                if sr is not None:
                    for gid in sr.input_ids:
                        if gid >= n_in:
                            use_count[gid] = use_count.get(gid, 0) + 1
                elif out_id >= n_in:
                    use_count[out_id] = use_count.get(out_id, 0) + 1

        def should_inline(gid: int) -> bool:
            return inline_single_use and use_count.get(gid, 0) <= 1

        # Build expression strings for inlined gates (memoised)
        _expr_cache: dict[int, str] = {}

        def expr_of(gid: int) -> str:
            """Return C expression for gate gid (inlined if single-use)."""
            if gid < 0:
                return const_false
            if gid < n_in:
                return f"in[{gid}]"
            if not should_inline(gid):
                return f"g{gid}"
            if gid in _expr_cache:
                return _expr_cache[gid]
            g = gate_by_id.get(gid)
            if g is None:
                return f"g{gid}"
            a = expr_of((g.input_ids[0] if g.input_ids else -1))
            b = expr_of((g.input_ids[1] if len(g.input_ids) > 1 else -1))
            result = gate_ops[g.op].format(a=a, b=b)
            _expr_cache[gid] = result
            return result

        pack_str = f"  pack_bits={pack_bits}" if pack_bits else ""

        def _c_float(v: float) -> str:
            s = f"{v:.9g}"
            if '.' not in s and 'e' not in s and 'E' not in s:
                s += '.0'
            return s + 'f'

        # Determine if any Binarization takes a raw input wire (input_id < n_in).
        # Those comparisons need the input as float rather than bool.
        input_binarizations = [b for b in self.binarization_nodes if b.input_ids and b.input_ids[0] < n_in]
        in_ctype = "float" if (pack_bits is None and input_binarizations) else ctype

        # Build a map from Binarization output_id → (Binarization, bit_index)
        bin_out_map = {oid: (b, k)
                       for b in self.binarization_nodes
                       for k, oid in enumerate(b.output_ids)}

        # Emit gates and Binarization outputs interleaved in topological (node_id) order.
        # Gates that have inline_single_use are skipped; Binarization outputs are always explicit.
        gate_lines = []
        current_layer = -1

        # Build a unified sorted emit list
        emit_list: list[tuple[int, object]] = []
        for gate in self.gates:
            emit_list.append((gate.node_id, gate))
        for b in self.binarization_nodes:
            for k, oid in enumerate(b.output_ids):
                emit_list.append((oid, (b, k)))
        emit_list.sort(key=lambda x: x[0])

        for node_id, obj in emit_list:
            if isinstance(obj, Gate):
                gate = obj
                if should_inline(gate.node_id):
                    continue
                if gate.layer != current_layer:
                    if current_layer >= 0:
                        gate_lines.append("")
                    current_layer = gate.layer
                    gate_lines.append(f"    // --- layer {current_layer} ---")
                a = expr_of((gate.input_ids[0] if gate.input_ids else -1))
                b_val = expr_of((gate.input_ids[1] if len(gate.input_ids) > 1 else -1))
                expr = gate_ops[gate.op].format(a=a, b=b_val)
                gate_lines.append(f"    {ctype} g{gate.node_id} = {expr};")
            else:
                b_node, bit_idx = obj
                inp_id = b_node.input_ids[0]
                threshold = b_node.thresholds[bit_idx]
                if inp_id < n_in:
                    inp_expr = f"in[{inp_id}]"
                else:
                    inp_expr = f"sum_{inp_id}"
                gate_lines.append(f"    bool g{node_id} = {inp_expr} > {_c_float(threshold)};")

        gates_str = "\n".join(gate_lines)

        def _sr_assign(dest: str, sr: SumReduction, s_expr: str) -> str:
            if is_int_red:
                b = int(round(sr.beta))
                return f"{dest} = {s_expr}{f' + {b}' if b else ''};"
            return f"{dest} = ((float){s_expr} + {_c_float(sr.beta)}) / {_c_float(sr.tau)};"

        if pack_bits is None:
            # Non-packed: one output value per output node
            out_lines = []
            for j, out_id in enumerate(self.outputs):
                sr = sum_by_id.get(out_id)
                if sr is not None:
                    start, end = sum_raw_offset[out_id]
                    if start == end:
                        val = int(round(sr.beta)) if is_int_red else _c_float(sr.beta / sr.tau)
                        out_lines.append(f"    out[{j}] = {val};")
                    else:
                        assign = _sr_assign(f"out[{j}]", sr, "s")
                        out_lines.append(
                            f"    {{\n"
                            f"        int s = 0;\n"
                            f"        for (int i = {start}; i < {end}; i++) s += (int)raw[i];\n"
                            f"        {assign}\n"
                            f"    }}"
                        )
                else:
                    cast = f"({out_ctype})" if has_reductions else ""
                    out_lines.append(f"    out[{j}] = {cast}{expr_of(out_id)};")
            out_section = "\n".join(out_lines)

            if n_raw > 0:
                raw_assigns = "\n".join(
                    f"    raw[{k}] = {expr_of(gid)};"
                    for k, gid in enumerate(raw_ids)
                )
                raw_section = f"\n    // --- raw inputs to sum reductions ---\n    bool raw[{n_raw}];\n{raw_assigns}\n"
            else:
                raw_section = ""
            output_section = f"""
{raw_section}
    // --- outputs ---
{out_section}"""
        elif not has_reductions:
            # Packed boolean-only: each out[j] is a bit-packed word (N samples in N bits).
            # This is the classical SIMD circuit evaluation format.
            out_assigns = "\n".join(
                f"    out[{j}] = {expr_of(out_id)};"
                for j, out_id in enumerate(self.outputs)
            )
            output_section = f"""

    // --- packed outputs ---
{out_assigns}"""
        else:
            # Packed with reductions (or mixed): per-sample layout out[p * n_total + j].
            pack_lines = []
            for j, out_id in enumerate(self.outputs):
                sr = sum_by_id.get(out_id)
                if sr is not None:
                    start, end = sum_raw_offset[out_id]
                    if start == end:
                        val = int(round(sr.beta)) if is_int_red else sr.beta / sr.tau
                        pack_lines.append(f"        out[p * {n_total} + {j}] = {val if is_int_red else _c_float(val)};")
                    else:
                        assign = _sr_assign(f"out[p * {n_total} + {j}]", sr, "s")
                        pack_lines.append(
                            f"        {{\n"
                            f"            int s = 0;\n"
                            f"            for (int i = {start}; i < {end}; i++)"
                            f" s += (int)((raw[i] >> p) & 1);\n"
                            f"            {assign}\n"
                            f"        }}"
                        )
                else:
                    pack_lines.append(
                        f"        out[p * {n_total} + {j}] = ({out_ctype})(({expr_of(out_id)} >> p) & 1);"
                    )
            pack_section = "\n".join(pack_lines)

            if n_raw > 0:
                raw_assigns = "\n".join(
                    f"    raw[{k}] = {expr_of(gid)};"
                    for k, gid in enumerate(raw_ids)
                )
                raw_section = f"\n    // --- raw packed inputs to sum reductions ---\n    {ctype} raw[{n_raw}];\n{raw_assigns}\n"
            else:
                raw_section = ""
            output_section = f"""
{raw_section}
    // --- outputs: sample p at out[p * {n_total} + j] ---
    for (int p = 0; p < {pack_bits}; p++) {{
{pack_section}
    }}"""

        return f"""\
// Auto-generated circuit — do not edit
// Gate IDs 0..{n_in - 1} are inputs, {n_in}..{n_in + n_g - 1} are gates

#include <stdbool.h>
#include <stdint.h>

// Input shape:  {self.input_shape}
// Output shape: {self.output_shape}
// n_inputs={n_in}  n_gates={n_g}  n_outputs={n_total}{pack_str}

void circuit(
    const {in_ctype} in[{n_in}],
    {out_ctype}   out[{out_n}])
{{
{gates_str}{output_section}
}}

void circuit_bench(
    const {in_ctype} in[{n_in}],
    {out_ctype}   out[{out_n}],
    int           n_iter)
{{
    for (int i = 0; i < n_iter; i++)
        circuit(in + i * {n_in}, out + i * {out_n});
}}

{"" if pack_bits is None else f"""
// Packs raw bool input, runs packed circuit, unpacks output — no Python packing needed.
void circuit_bench_bool(
    const bool   *in_bool,  // (batch_size, {n_in}) bool, row-major
    {'bool' if not has_reductions else out_ctype}  *out,  // (batch_size, {n_total}) {'bool' if not has_reductions else out_ctype}, row-major
    int           batch_size)
{{
    int n_iter = batch_size / {pack_bits};
    {ctype} packed_in[{n_in}];
    {out_ctype} packed_out[{out_n}];

    for (int iter = 0; iter < n_iter; iter++) {{

        // Pack {pack_bits} bool samples per input wire into one {ctype} word
        for (int k = 0; k < {n_in}; k++) {{
            {ctype} w = ({ctype})0;
            for (int b = 0; b < {pack_bits}; b++)
                w |= ({ctype})in_bool[(iter * {pack_bits} + b) * {n_in} + k] << b;
            packed_in[k] = w;
        }}

        circuit(packed_in, packed_out);

        // {'Unpack packed-word outputs to individual bools' if not has_reductions else 'Copy per-sample outputs (already unpacked by circuit)'}
        for (int b = 0; b < {pack_bits}; b++)
            for (int j = 0; j < {n_total}; j++)
                out[(iter * {pack_bits} + b) * {n_total} + j] =
                    {f'(packed_out[j] >> b) & 1' if not has_reductions else f'packed_out[b * {n_total} + j]'};
    }}
}}"""}"""


    def turn_group_sum_into_argmax(self) -> None:
        """
        Replace the GroupSum output reduction with pure gate logic that computes
        the argmax of the k group sums. Ties are broken in favour of the
        lowest-index class. After this call output_reduction is None and the
        circuit has k boolean outputs (one-hot: winning class bit is 1).
        """
        red_outs = [sr for sr in self.sum_nodes if sr.node_id in set(self.outputs)]
        if not red_outs:
            raise ValueError("Circuit has no sum-reduction outputs to convert")

        k = len(red_outs)
        argmax_layer = max((g.layer for g in self.gates), default=-1) + 1
        next_id = max((g.node_id for g in self.gates), default=self.n_inputs - 1) + 1

        # A lazily-created CONST_TRUE gate (needed only if NOT(always-false) appears).
        _const_true_id: list[int] = []

        def get_const_true() -> int:
            nonlocal next_id
            if not _const_true_id:
                gid = next_id
                self.gates.append(Gate(node_id=gid, input_ids=[], op=GateOp.CONST_TRUE, layer=argmax_layer))
                next_id += 1
                _const_true_id.append(gid)
            return _const_true_id[0]

        def emit(op: GateOp, a: int = -1, b: int = -1) -> int:
            """Emit a gate with constant folding for -1 (constant false) inputs."""
            nonlocal next_id
            if op == GateOp.WIRE:      return a
            if op == GateOp.NOT:
                if a == -1:            return get_const_true()
            if op == GateOp.AND:
                if a == -1 or b == -1: return -1
            if op == GateOp.OR:
                if a == -1 and b == -1: return -1
                if a == -1:            return b
                if b == -1:            return a
            if op == GateOp.XOR:
                if a == -1:            return b
                if b == -1:            return a
            if op == GateOp.AND_NOT_B:   # a AND NOT b
                if a == -1:            return -1
                if b == -1:            return a   # a AND true = a
            if op == GateOp.AND_NOT_A:   # NOT a AND b
                if b == -1:            return -1
                if a == -1:            return b   # true AND b = b
            if op == GateOp.XNOR:
                if a == -1 and b == -1: return get_const_true()
                if a == -1:            return emit(GateOp.NOT, b)
                if b == -1:            return emit(GateOp.NOT, a)
            gid = next_id
            self.gates.append(Gate(node_id=gid, input_ids=[a, b], op=op, layer=argmax_layer))
            next_id += 1
            return gid

        def add_bits(a_bits: list[int], b_bits: list[int]) -> list[int]:
            """Add two unsigned binary numbers (LSB first). Returns sum bits."""
            n = max(len(a_bits), len(b_bits))
            a = a_bits + [-1] * (n - len(a_bits))
            b = b_bits + [-1] * (n - len(b_bits))
            result = []
            carry = -1
            for ai, bi in zip(a, b):
                xab   = emit(GateOp.XOR, ai, bi)
                s     = emit(GateOp.XOR, xab, carry)
                c_ab  = emit(GateOp.AND, ai, bi)
                c_xc  = emit(GateOp.AND, xab, carry)
                carry = emit(GateOp.OR, c_ab, c_xc)
                result.append(s)
            result.append(carry)
            return result

        def popcount(bit_ids: list[int]) -> list[int]:
            """Binary-tree popcount. Returns count as binary number (LSB first)."""
            if not bit_ids:
                return [-1]
            numbers = [[b] for b in bit_ids]
            while len(numbers) > 1:
                new_numbers = []
                for i in range(0, len(numbers) - 1, 2):
                    new_numbers.append(add_bits(numbers[i], numbers[i + 1]))
                if len(numbers) % 2 == 1:
                    new_numbers.append(numbers[-1])
                numbers = new_numbers
            return numbers[0]

        def compare_gt(a_bits: list[int], b_bits: list[int]) -> int:
            """Return gate ID for (unsigned(a) > unsigned(b)).

            Standard ripple comparator: scan MSB→LSB tracking 'greater so far'
            and 'equal so far'. Equal is true until any bit differs.
              greater_new = greater OR (equal AND a_wins)
              equal_new   = equal AND (a[i] XNOR b[i])
            """
            n = max(len(a_bits), len(b_bits))
            a = a_bits + [-1] * (n - len(a_bits))
            b = b_bits + [-1] * (n - len(b_bits))
            greater = -1   # false: a not greater so far
            equal   = None  # None = implicitly true (before first bit)
            for i in range(n - 1, -1, -1):
                ai, bi = a[i], b[i]
                a_wins = emit(GateOp.AND_NOT_B, ai, bi)
                a_eq   = emit(GateOp.XNOR, ai, bi)
                if equal is None:
                    # First bit: equal is implicitly true, so AND(true, a_wins) = a_wins
                    greater = emit(GateOp.OR, greater, a_wins)
                    equal   = a_eq
                else:
                    greater = emit(GateOp.OR, greater, emit(GateOp.AND, equal, a_wins))
                    equal   = emit(GateOp.AND, equal, a_eq)
            return greater

        # 1. Popcount for each SumReduction's input bits, including any constant
        # contribution that was folded into beta by constant_fold_sum_reductions.
        # We add int(beta) CONST_TRUE wires so the comparators see the full sum;
        # the subsequent simplify() will fold those constants back out of the adder.
        def _bits_with_const(sr: SumReduction) -> list[int]:
            bits = list(sr.input_ids)
            ct = get_const_true()
            bits.extend([ct] * int(round(sr.beta)))
            return bits

        counts = [popcount(_bits_with_const(sr)) for sr in red_outs]

        # 2. Pairwise comparisons: gt[(a, b)] = gate for "a strictly beats b", a > b.
        gt: dict[tuple[int, int], int] = {}
        for a in range(k):
            for b in range(a):
                gt[(a, b)] = compare_gt(counts[a], counts[b])

        # 3. One-hot output.
        # out[j] = AND(j > m for all m < j) AND AND(NOT (m > j) for all m > j)
        out_ids = []
        for j in range(k):
            conditions = []
            for m in range(j):
                conditions.append(gt[(j, m)])
            for m in range(j + 1, k):
                conditions.append(emit(GateOp.NOT, gt[(m, j)]))
            if not conditions:
                out_ids.append(get_const_true())
            else:
                result = conditions[0]
                for c in conditions[1:]:
                    result = emit(GateOp.AND, result, c)
                out_ids.append(result)

        # Replace reduction output IDs in self.outputs with argmax gate IDs
        red_id_to_argmax = {sr.node_id: out_ids[j] for j, sr in enumerate(red_outs)}
        self.outputs = [red_id_to_argmax.get(oid, oid) for oid in self.outputs]
        self.output_shape = [k]
        self.sum_nodes = []



    def to_dict(self) -> dict:
        """
        Convert the circuit representation to a JSON-serializable format.
        """
        d = {
            'n_inputs': self.n_inputs,
            'input_shape': self.input_shape,
            'outputs': self.outputs,
            'output_shape': self.output_shape,
            'gates': [
                {
                    'node_id': g.node_id,
                    'op': g.op.name,
                    'input_ids': g.input_ids,
                    'layer': g.layer,
                    'node_idx': g.node_idx,
                }
                for g in self.gates
            ],
        }
        if self.sum_nodes:
            d['sum_nodes'] = [
                {'node_id': sr.node_id, 'input_ids': sr.input_ids,
                 'tau': sr.tau, 'beta': sr.beta}
                for sr in self.sum_nodes
            ]
        if self.binarization_nodes:
            d['binarization_nodes'] = [
                {'node_id': b.node_id, 'input_ids': b.input_ids,
                 'output_ids': b.output_ids, 'thresholds': b.thresholds}
                for b in self.binarization_nodes
            ]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Circuit:
        """
        Create a Circuit instance from a (JSON-deserialized) dictionary.
        """
        circuit = cls(n_inputs=data['n_inputs'], input_shape=data['input_shape'])
        circuit.outputs = data.get('outputs', [])
        circuit.output_shape = data.get('output_shape', [])
        for g_data in data.get('gates', []):
            # Support both new format (input_ids) and old format (in0/in1)
            if 'input_ids' in g_data:
                inputs = g_data['input_ids']
            else:
                inputs = [x for x in [g_data.get('in0', -1), g_data.get('in1', -1)] if x >= 0]
            g = Gate(
                node_id=g_data.get('gate_id', g_data.get('node_id')),
                input_ids=inputs,
                op=GateOp[g_data['op']],
                layer=g_data.get('layer', -1),
                node_idx=g_data.get('node_idx', -1),
            )
            circuit.gates.append(g)
        if 'sum_nodes' in data:
            circuit.sum_nodes = [
                SumReduction(
                    node_id=sr['node_id'],
                    input_ids=sr['input_ids'],
                    tau=sr.get('tau', 1.0),
                    beta=sr.get('beta', 0.0),
                )
                for sr in data['sum_nodes']
            ]
        if 'binarization_nodes' in data:
            circuit.binarization_nodes = [
                Binarization(
                    node_id=b['node_id'],
                    input_ids=b['input_ids'],
                    output_ids=b['output_ids'],
                    thresholds=b['thresholds'],
                )
                for b in data['binarization_nodes']
            ]
        return circuit
    
    @classmethod
    def from_json_file(cls, file_path: str) -> Circuit:
        """
        Load a Circuit instance from a JSON file.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


    def get_verilog_code(self, inline_single_use: bool) -> str:
        """
        Generate a Verilog module that implements the circuit.

        Each gate becomes a continuous assignment:
            wire g<id> = <expr>;
        Outputs are assigned to an output bus:
            assign out[k] = <expr>;

        inline_single_use=True: single-use gates are folded into their parent
        expression rather than named wires (same semantics as in emit_c).
        """
        n_in      = self.n_inputs
        n_total   = len(self.outputs)
        sum_by_id = self._sum_by_id
        gate_by_id = {g.node_id: g for g in self.gates}

        # Build raw[] layout for sum reductions (same as get_c_code)
        raw_ids: list[int] = []
        sum_raw_offset_v: dict[int, tuple[int, int]] = {}
        for out_id in self.outputs:
            sr = sum_by_id.get(out_id)
            if sr is not None and out_id not in sum_raw_offset_v:
                start = len(raw_ids)
                raw_ids.extend(sr.input_ids)
                sum_raw_offset_v[out_id] = (start, len(raw_ids))
        n_raw = len(raw_ids)

        red_outs  = [sum_by_id[oid] for oid in self.outputs if oid in sum_by_id]
        has_red   = bool(red_outs)

        # ---- use-count for optional inlining --------------------------------
        use_count: dict[int, int] = {}
        if inline_single_use:
            for g in self.gates:
                use_count.setdefault(g.node_id, 0)
                for dep in g.input_ids:
                    if dep >= n_in:
                        use_count[dep] = use_count.get(dep, 0) + 1
            for out_id in self.outputs:
                sr = sum_by_id.get(out_id)
                if sr is not None:
                    for gid in sr.input_ids:
                        if gid >= n_in:
                            use_count[gid] = use_count.get(gid, 0) + 1
                elif out_id >= n_in:
                    use_count[out_id] = use_count.get(out_id, 0) + 1

        def should_inline(gid: int) -> bool:
            return inline_single_use and use_count.get(gid, 0) <= 1

        _expr_cache: dict[int, str] = {}

        def vexpr(gid: int) -> str:
            """Return a Verilog expression for gate gid (inlined if single-use)."""
            if gid < 0:
                return "1'b0"
            if gid < n_in:
                return f"inp[{gid}]"
            if not should_inline(gid):
                return f"g{gid}"
            if gid in _expr_cache:
                return _expr_cache[gid]
            g = gate_by_id.get(gid)
            if g is None:
                return f"g{gid}"
            a = vexpr((g.input_ids[0] if g.input_ids else -1))
            b = vexpr((g.input_ids[1] if len(g.input_ids) > 1 else -1))
            result = GATE_OP_VERILOG[g.op].format(a=a, b=b)
            _expr_cache[gid] = result
            return result

        gate_lines = []
        current_layer = -1
        for gate in self.gates:
            if should_inline(gate.node_id):
                continue
            if gate.layer != current_layer:
                gate_lines.append("")
                current_layer = gate.layer
                gate_lines.append(f"    // --- layer {current_layer} ---")
            a = vexpr((gate.input_ids[0] if gate.input_ids else -1))
            b = vexpr((gate.input_ids[1] if len(gate.input_ids) > 1 else -1))
            expr = GATE_OP_VERILOG[gate.op].format(a=a, b=b)
            gate_lines.append(f"    wire g{gate.node_id} = {expr};")
        gates_str = "\n".join(gate_lines)

        if has_red:
            _dtype_bits = {
                "float": 32, "uint8_t": 8, "uint16_t": 16,
                "uint32_t": 32, "uint64_t": 64,
            }
            score_bits = _dtype_bits[SumReduction.infer_c_dtype(red_outs)]
            reduction_comment = (
                f"// {n_total} output(s) — scores_flat = {n_total} x {score_bits}-bit values\n"
            )
            module_port = f"    output reg  [{n_total * score_bits - 1}:0] scores_flat"

            # Build always block body: one entry per output in order
            sv_lines = []
            sv_vars = set()
            for j, out_id in enumerate(self.outputs):
                sr = sum_by_id.get(out_id)
                slot = f"scores_flat[{j}*{score_bits} +: {score_bits}]"
                if sr is not None:
                    sv_vars.add(f"s_{j}")
                    start, end = sum_raw_offset_v[out_id]
                    if start == end:
                        val = int(round(sr.beta)) if sr.tau == 1.0 else sr.beta / sr.tau
                        sv_lines.append(f"        s_{j} = 0;\n        {slot} = {val};")
                    else:
                        sv_lines.append(
                            f"        s_{j} = 0;\n"
                            f"        for (i = {start}; i < {end}; i = i + 1)"
                            f" s_{j} = s_{j} + raw[i];\n"
                            f"        {slot} = s_{j};"
                        )
                else:
                    sv_lines.append(f"        {slot} = {vexpr(out_id)};")

            sum_vars_decl = ", ".join(sorted(sv_vars)) + ", i" if sv_vars else "i"
            sum_body = "\n".join(sv_lines)

            if n_raw > 0:
                raw_assigns = "\n".join(
                    f"    assign raw[{k}] = {vexpr(gid)};"
                    for k, gid in enumerate(raw_ids)
                )
                raw_section = f"\n    // --- raw inputs to sum reductions ---\n    wire [{n_raw - 1}:0] raw;\n{raw_assigns}\n"
            else:
                raw_section = ""

            output_section = f"""
{raw_section}
    // --- outputs (behavioral — synthesizer maps to carry chain) ---
    integer {sum_vars_decl};
    always @(*) begin
{sum_body}
    end"""
        else:
            reduction_comment = ""
            module_port = f"    output wire [{n_total - 1}:0] out"
            out_assigns = "\n".join(
                f"    assign out[{k}] = {vexpr(out_id)};"
                for k, out_id in enumerate(self.outputs)
            )
            output_section = f"""

    // --- outputs ---
{out_assigns}"""

        return f"""\
// Auto-generated by circuit_ir — do not edit
// Input shape:  {self.input_shape}
// Output shape: {self.output_shape}
// n_inputs={n_in}  n_gates={len(self.gates)}  n_outputs={n_total}

{reduction_comment}module circuit (
    input  wire [{n_in - 1}:0] inp,
{module_port}
);
{gates_str}{output_section}

endmodule"""


    def write_c_code(self, path: str) -> None:
        """
        Write the generated C code to a file.
        """
        c_code = self.get_c_code()
        with open(path, 'w') as f:
            f.write(c_code)

    def write_verilog_code(self, path: str) -> None:
        """
        Write the generated Verilog code to a file.
        """
        verilog_code = self.get_verilog_code()
        with open(path, 'w') as f:
            f.write(verilog_code)

    def write_json(self, path: str) -> None:
        """
        Write the circuit representation to a JSON file.
        """
        json_data = self.to_dict()
        with open(path, 'w') as f:
            json.dump(json_data, f)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_attr_val(gm: torch.fx.GraphModule, node: torch.fx.Node):
    obj = gm
    for part in node.target.split('.'):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Constant-fold view ops on weight tensors
# ---------------------------------------------------------------------------

def constant_fold_views(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Pre-evaluate shape/index ops (movedim, reshape, select, slice, unbind,
    lift_fresh_copy) that operate on constant weight tensors.

    This is *required* (not optional) before build_circuit because the wiring
    step (aten.index.Tensor) needs concrete integer index tensors.  Without
    folding, those tensors remain as unevaluated call_function nodes whose
    result is not available at graph-build time; the fallback in build_circuit
    would use gate IDs instead of actual index values and produce wrong wiring.
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
# Constant-gate folding (algebraic simplification)
# ---------------------------------------------------------------------------

def _simplify_gate(op, in0, in1, a_const, b_const):
    """
    Given a gate op and the known constant values (True/False/None) of its
    two inputs, return (new_op, new_in0, new_in1, known_output_or_None).
    """
    CF = GateOp.CONST_FALSE
    CT = GateOp.CONST_TRUE

    if op == GateOp.CONST_FALSE:
        return op, -1, -1, False
    if op == GateOp.CONST_TRUE:
        return op, -1, -1, True

    # Single-input ops
    if op in (GateOp.WIRE, GateOp.NOT, GateOp.NOT_A):
        if a_const is not None:
            v = a_const if op == GateOp.WIRE else not a_const
            return (CT if v else CF), -1, -1, v
        return op, in0, in1, None

    if op == GateOp.NOT_B:
        if b_const is not None:
            v = not b_const
            return (CT if v else CF), -1, -1, v
        return op, in0, in1, None

    # Two-input ops
    if op == GateOp.AND:
        if a_const is False or b_const is False:  return CF, -1, -1, False
        if a_const is True and b_const is True:   return CT, -1, -1, True
        if a_const is True:   return GateOp.WIRE, in1, -1, None
        if b_const is True:   return GateOp.WIRE, in0, -1, None

    elif op == GateOp.OR:
        if a_const is True or b_const is True:    return CT, -1, -1, True
        if a_const is False and b_const is False:  return CF, -1, -1, False
        if a_const is False:  return GateOp.WIRE, in1, -1, None
        if b_const is False:  return GateOp.WIRE, in0, -1, None

    elif op == GateOp.XOR:
        if a_const is False and b_const is False:  return CF, -1, -1, False
        if a_const is True  and b_const is True:   return CF, -1, -1, False
        if a_const is False:  return GateOp.WIRE, in1, -1, None
        if b_const is False:  return GateOp.WIRE, in0, -1, None
        if a_const is True:   return GateOp.NOT,  in1, -1, None
        if b_const is True:   return GateOp.NOT,  in0, -1, None

    elif op == GateOp.NAND:
        if a_const is False or b_const is False:   return CT, -1, -1, True
        if a_const is True and b_const is True:    return CF, -1, -1, False
        if a_const is True:   return GateOp.NOT, in1, -1, None
        if b_const is True:   return GateOp.NOT, in0, -1, None

    elif op == GateOp.NOR:
        if a_const is True  or b_const is True:    return CF, -1, -1, False
        if a_const is False and b_const is False:   return CT, -1, -1, True
        if a_const is False:  return GateOp.NOT, in1, -1, None
        if b_const is False:  return GateOp.NOT, in0, -1, None

    elif op == GateOp.XNOR:
        if a_const is False and b_const is False:   return CT, -1, -1, True
        if a_const is True  and b_const is True:    return CT, -1, -1, True
        if a_const is False:  return GateOp.NOT,  in1, -1, None
        if b_const is False:  return GateOp.NOT,  in0, -1, None
        if a_const is True:   return GateOp.WIRE, in1, -1, None
        if b_const is True:   return GateOp.WIRE, in0, -1, None

    elif op == GateOp.AND_NOT_B:   # a & !b
        if b_const is True:   return CF, -1, -1, False   # a & false
        if b_const is False:  return GateOp.WIRE, in0, -1, None  # a & true
        if a_const is False:  return CF, -1, -1, False
        if a_const is True:   return GateOp.NOT, in1, -1, None

    elif op == GateOp.AND_NOT_A:   # !a & b
        if a_const is True:   return CF, -1, -1, False
        if a_const is False:  return GateOp.WIRE, in1, -1, None
        if b_const is False:  return CF, -1, -1, False
        if b_const is True:   return GateOp.NOT, in0, -1, None

    elif op == GateOp.OR_NOT_B:    # a | !b
        if b_const is True:   return GateOp.WIRE, in0, -1, None  # a | false
        if b_const is False:  return CT, -1, -1, True              # a | true
        if a_const is True:   return CT, -1, -1, True
        if a_const is False:  return GateOp.NOT, in1, -1, None

    elif op == GateOp.OR_NOT_A:    # !a | b
        if a_const is True:   return GateOp.WIRE, in1, -1, None
        if a_const is False:  return CT, -1, -1, True
        if b_const is True:   return CT, -1, -1, True
        if b_const is False:  return GateOp.NOT, in0, -1, None

    return op, in0, in1, None
