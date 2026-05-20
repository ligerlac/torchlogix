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
import operator
import json

import torch
import torch.fx

from torchlogix.utils import set_export_mode


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
    GateOp.NOT:         "!{a}",
    GateOp.AND:         "({a} & {b})",
    GateOp.OR:          "({a} | {b})",
    GateOp.XOR:         "({a} ^ {b})",
    GateOp.NAND:        "!({a} & {b})",
    GateOp.NOR:         "!({a} | {b})",
    GateOp.XNOR:        "!({a} ^ {b})",
    GateOp.AND_NOT_B:   "({a} & !{b})",
    GateOp.AND_NOT_A:   "(!{a} & {b})",
    GateOp.OR_NOT_B:    "({a} | !{b})",
    GateOp.OR_NOT_A:    "(!{a} | {b})",
    GateOp.NOT_A:       "!{a}",
    GateOp.NOT_B:       "!{b}",
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
class Gate:
    gate_id: int
    op:      GateOp
    in0:     int = -1   # -1 = unused
    in1:     int = -1
    # metadata
    layer:   int = -1
    node_idx: int = -1  # index within the layer's flat node list


@dataclass
class Circuit:
    n_inputs:    int
    input_shape: list[int]          # original shape of the input tensor
    gates:       list[Gate] = field(default_factory=list)
    output_ids:   list[int] = field(default_factory=list)   # flat, in order
    output_shape: list[int] = field(default_factory=list)

    @classmethod
    def from_model(cls, model: torch.nn.Module, input_shape: list[int]) -> Circuit:
        """
        Build a Circuit from a PyTorch model by tracing and folding it.

        The model should be in export mode (if applicable) and should have
        been traced and folded with the appropriate utilities to ensure the
        FX graph is in the expected form.
        """
        model.eval()
        set_export_mode(model, enabled=True, batch_size=1)
        x_dummy = torch.zeros(1, *input_shape, dtype=torch.bool)
        exported = torch.export.export(model, (x_dummy,), strict=False)
        gm = exported.module()
        return cls.from_fx_graph(gm, [1, *input_shape])

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

            # ---- get_attr: lut_ids tensors we need later; skip others ----
            if node.op == 'get_attr':
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
                if isinstance(ret, (tuple, list)):
                    out_ids = []
                    for r in ret:
                        if isinstance(r, torch.fx.Node) and r.name in wire_map:
                            out_ids.extend(wire_map[r.name])
                    if out_ids:
                        circuit.output_ids = out_ids
                        # use stored shape if available, else flat
                        if len(ret) == 1 and isinstance(ret[0], torch.fx.Node):
                            shape = wire_map.get(f'__shape_{ret[0].name}', [len(out_ids)])
                            circuit.output_shape = list(shape)
                        else:
                            circuit.output_shape = [len(out_ids)]
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
                src_ids = resolve(node.args[0])
                idx_list = node.args[1]   # list of None | fx.Node

                # Find the non-None index tensors
                active = [(dim, idx) for dim, idx in enumerate(idx_list) if idx is not None]

                if len(active) == 0:
                    # Identity gather — pass through
                    wire_map[node.name] = src_ids
                else:
                    # Evaluate the gathered IDs.
                    # Each index tensor contains integer positions into src_ids.
                    # We need to compute the cartesian result.
                    #
                    # Strategy: build the full gathered ID tensor by evaluating
                    # the index operation on the *ID* array (treating gate IDs
                    # as the "values" to gather).
                    src_shape = []
                    # Reconstruct src shape from input_shape or prior layer shape
                    # We store this alongside wire_map
                    src_shape = wire_map.get(f'__shape_{node.args[0].name}',
                                            list(input_shape))

                    # Reconstruct index tensors as concrete tensors
                    idx_tensors = []
                    for dim, idx_node in active:
                        idx_tensors.append((dim, _get_attr_val(gm, idx_node)
                                            if idx_node.op == 'get_attr'
                                            else torch.tensor(resolve(idx_node))))

                    # Build a tensor of gate IDs with src_shape, then index it
                    id_tensor = torch.tensor(src_ids, dtype=torch.long).reshape(src_shape)
                    # Apply each index in turn (simplified: single multi-dim index)
                    index_args = [slice(None)] * len(src_shape)
                    for dim, t in idx_tensors:
                        index_args[dim] = t
                    gathered = id_tensor[tuple(index_args)]   # shape: gathered output shape

                    flat_ids = gathered.flatten().tolist()
                    wire_map[node.name] = [int(x) for x in flat_ids]
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
            # aten.reshape / aten.view  ->  just reshape the ID list (no new gates)
            # ----------------------------------------------------------------
            if tgt in (torch.ops.aten.reshape.default, torch.ops.aten.view.default):
                src_ids  = resolve(node.args[0])
                new_shape = node.args[1]
                wire_map[node.name] = src_ids
                wire_map[f'__shape_{node.name}'] = list(new_shape)
                circuit.output_ids   = src_ids
                circuit.output_shape = list(new_shape)
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
                circuit.output_ids   = wire_map[node.name]
                circuit.output_shape = list(flattened.shape)
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
            # aten.pad  ->  zero-pad (inserts CONST_FALSE/CONST_TRUE gates)
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.pad.default:
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
                            g = Gate(gate_id=next_id, op=const_op, layer=layer_idx)
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
            # aten.cat  ->  concatenate ID tensors along a dimension
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.cat.default:
                cat_nodes = node.args[0]
                dim = node.args[1] if len(node.args) > 1 else 0
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
            # Direct boolean binary/unary ops (e.g. OrPooling2d)
            # These appear outside LUT cascades with both operands in wire_map.
            # ----------------------------------------------------------------
            _DIRECT_BINARY_GATE = {
                torch.ops.aten.__or__.Tensor:  GateOp.OR,
                torch.ops.aten.__and__.Tensor: GateOp.AND,
                torch.ops.aten.__xor__.Tensor: GateOp.XOR,
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
                        g = Gate(gate_id=next_id, op=gate_op, in0=a_id, in1=b_id,
                                layer=layer_idx)
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
                        g = Gate(gate_id=next_id, op=GateOp.NOT, in0=a_id, layer=layer_idx)
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
                    g = Gate(gate_id=next_id, op=op, layer=layer_idx)
                    circuit.gates.append(g)
                    gate_ids.append(next_id)
                    next_id += 1
                wire_map[node.name] = gate_ids
                src_shape = wire_map.get(f'__shape_{node.args[0].name}', [len(ref_ids)])
                wire_map[f'__shape_{node.name}'] = src_shape
                i += 1
                continue

            # ----------------------------------------------------------------
            # LUT cascade — two possible graph patterns:
            #
            # 1. "where" cascade (dense):
            #      eq = aten.eq.Scalar(lut_ids, K)
            #      result_k = <op>(a, b)
            #      where_k = aten.where.self(eq, result_k, prev)
            #
            # 2. "index_put_" scatter (sparse):
            #      eq = aten.eq.Scalar(lut_ids, K)
            #      a_k  = aten.index.Tensor(select_a, [eq])   # boolean gather
            #      b_k  = aten.index.Tensor(select_b, [eq])
            #      val  = <op>(a_k, b_k)
            #      buf  = aten.index_put_.default(prev, [eq], val)
            #
            # Both carry identical per-gate info (lut_ids + a/b select nodes),
            # so the gate-emission logic is shared.
            # ----------------------------------------------------------------
            if tgt == torch.ops.aten.eq.Scalar:
                # Walk back through unsqueeze/view wrappers to the underlying get_attr.
                arg0 = node.args[0]
                lut_ids_node = arg0
                while (lut_ids_node.op == 'call_function'
                    and lut_ids_node.target in (
                        torch.ops.aten.unsqueeze.default,
                        torch.ops.aten.reshape.default,
                        torch.ops.aten.view.default,
                    )):
                    lut_ids_node = lut_ids_node.args[0]
                if not (lut_ids_node.op == 'get_attr' and 'lut_ids' in lut_ids_node.name):
                    i += 1
                    continue
                lut_ids = _get_attr_val(gm, lut_ids_node).flatten().tolist()
                n_nodes = len(lut_ids)

                # Walk backward to find the two select nodes that supply a/b inputs.
                a_ids = None
                b_ids = None
                a_node = None
                b_node = None
                for k in range(i - 1, -1, -1):
                    n2 = nodes[k]
                    if (n2.op == 'call_function'
                            and n2.target == torch.ops.aten.select.int
                            and n2.name in wire_map):
                        if b_ids is None:
                            b_ids = resolve(n2)
                            b_node = n2
                        elif a_ids is None:
                            a_ids = resolve(n2)
                            a_node = n2
                            break

                if a_ids is None or b_ids is None:
                    i += 1
                    continue

                # Names of the select nodes — used to identify boolean gathers
                # (index.Tensor nodes inside the scatter cascade) vs wiring steps.
                select_names: set[str] = set()
                if a_node is not None:
                    select_names.add(a_node.name)
                if b_node is not None:
                    select_names.add(b_node.name)

                # Scan forward over the cascade, stopping when we reach a node that
                # begins the next wiring step or layer boundary.
                j = i
                last_output_node = None   # last where or index_put_ seen
                while j < len(nodes):
                    n2 = nodes[j]
                    j += 1   # advance past n2

                    if n2.op == 'output':
                        # Step back so the outer loop processes the output node.
                        j -= 1
                        break

                    if n2.op == 'get_attr':
                        # Constant tensor used inside the cascade (e.g. folded
                        # ones/zeros tensor for LUT 15/0) — skip, don't stop.
                        continue

                    if n2.op != 'call_function':
                        # Unexpected non-call node — stop conservatively.
                        j -= 1
                        break

                    if n2.target == torch.ops.aten.reshape.default:
                        # Reshape ends the cascade; step back so the reshape handler runs.
                        j -= 1
                        break

                    if n2.target == torch.ops.aten.index.Tensor:
                        # Distinguish boolean gather (inside scatter cascade) from
                        # wiring index (takes the cascade output as source).
                        # Boolean gathers: src is a select node, index list has
                        # exactly one non-None entry (the bool mask). Nones can
                        # appear as prefix for batch dims (e.g. [None, eq_K]).
                        src_name = (n2.args[0].name
                                    if isinstance(n2.args[0], torch.fx.Node) else None)
                        idx_list = n2.args[1]
                        non_none = [idx for idx in idx_list if idx is not None]
                        is_bool_gather = (
                            src_name in select_names
                            and len(non_none) == 1
                        )
                        if not is_bool_gather:
                            # Wiring step — step back so the index handler runs.
                            j -= 1
                            break
                        # else: boolean gather is part of this cascade, continue

                    if n2.target == torch.ops.aten.where.self:
                        last_output_node = n2

                    if n2.target == torch.ops.aten.index_put_.default:
                        last_output_node = n2

                # Emit one gate per output position using lut_ids + a/b wire IDs.
                result_ids = [None] * n_nodes
                for nidx, lut_id in enumerate(lut_ids):
                    lut_id = int(lut_id)
                    op, uses_a, uses_b = LUT_ID_TO_GATE[lut_id]
                    in0 = a_ids[nidx] if uses_a else -1
                    in1 = b_ids[nidx] if uses_b else -1
                    # LUT 5 (wire b) and LUT 10 (not b): in0 must be b
                    if lut_id == 5:
                        in0 = b_ids[nidx]
                        in1 = -1
                    elif lut_id == 10:
                        in0 = b_ids[nidx]
                        in1 = -1

                    g = Gate(gate_id=next_id, op=op, in0=in0, in1=in1,
                            layer=layer_idx, node_idx=nidx)
                    circuit.gates.append(g)
                    result_ids[nidx] = next_id
                    next_id += 1

                # Register the cascade's output node in wire_map so the next
                # wiring step (index.Tensor / reshape) can resolve it.
                if last_output_node is not None:
                    out_shape = (wire_map.get(f'__shape_{b_node.name}')
                                if b_node is not None else None) or [n_nodes]
                    wire_map[last_output_node.name] = result_ids
                    wire_map[f'__shape_{last_output_node.name}'] = out_shape

                layer_idx += 1
                i = j
                continue

            # ---- skip everything else (sym_size, asserts, eq on non-lut, etc.) ----
            i += 1

        # If the output node was jumped over by the where-cascade skip, resolve it now.
        if not circuit.output_ids:
            for node in nodes:
                if node.op == 'output':
                    ret = node.args[0]
                    if isinstance(ret, (tuple, list)):
                        out_ids = []
                        for r in ret:
                            if isinstance(r, torch.fx.Node) and r.name in wire_map:
                                out_ids.extend(wire_map[r.name])
                        if out_ids:
                            circuit.output_ids = out_ids
                            if len(ret) == 1 and isinstance(ret[0], torch.fx.Node):
                                shape = wire_map.get(f'__shape_{ret[0].name}',
                                                    [len(out_ids)])
                                circuit.output_shape = list(shape)
                            else:
                                circuit.output_shape = [len(out_ids)]
                    break

        return circuit
    

    def simplify(self, n_max=1000) -> None:
        for _ in range(n_max):
            before = len(self.gates)
            self.constant_fold_gates()
            self.bypass_wires()
            self.dedup()
            self.eliminate_dead_gates()
            after = len(self.gates)
            if after == before:
                break


    def eliminate_dead_gates(self) -> None:
        """
        Remove gates that do not contribute to the output (i.e. not on a path
        from any output ID back to an input ID).
        """
        gate_by_id = {g.gate_id: g for g in self.gates}

        # Backward BFS from outputs
        visited: set[int] = set()
        queue = list(self.output_ids)
        while queue:
            gid = queue.pop()
            if gid in visited or gid < self.n_inputs:
                continue
            visited.add(gid)
            g = gate_by_id.get(gid)
            if g is None:
                continue
            if g.in0 >= 0:
                queue.append(g.in0)
            if g.in1 >= 0:
                queue.append(g.in1)

        self.gates = [g for g in self.gates if g.gate_id in visited]


    def constant_fold_gates(self) -> None:
        """
        Evaluate gates that have constant inputs and replace them with CONST_TRUE
        or CONST_FALSE gates as appropriate. This can simplify the circuit and
        reduce the number of gates.
        """
        const_val: dict[int, bool] = {}
        new_gates = []
        for g in self.gates:
            a_c = const_val.get(g.in0) if g.in0 >= 0 else None
            b_c = const_val.get(g.in1) if g.in1 >= 0 else None
            new_op, new_in0, new_in1, known = _simplify_gate(
                g.op, g.in0, g.in1, a_c, b_c)
            if known is not None:
                const_val[g.gate_id] = known
            new_gates.append(Gate(gate_id=g.gate_id, op=new_op,
                                in0=new_in0, in1=new_in1,
                                layer=g.layer, node_idx=g.node_idx))
        self.gates = new_gates


    def bypass_wires(self) -> None:
        """
        Eliminate trivial aliases:
            WIRE(x)    -> x
            NOT(NOT(x)) -> x

        Rewrites all fanins/output IDs transitively and removes dead alias gates.

        This pass is intentionally conservative and cheap.
        """

        gate_by_id = {g.gate_id: g for g in self.gates}

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
                if g.op == GateOp.WIRE and g.in0 >= 0:
                    target = alias.get(g.in0, g.in0)

                    if alias.get(g.gate_id) != target:
                        alias[g.gate_id] = target
                        changed = True

                # ----------------------------------------------------------
                # NOT(NOT(x)) -> x
                # ----------------------------------------------------------
                elif g.op == GateOp.NOT and g.in0 >= 0:
                    src = gate_by_id.get(g.in0)

                    if src is not None and src.op == GateOp.NOT:
                        target = alias.get(src.in0, src.in0)

                        if alias.get(g.gate_id) != target:
                            alias[g.gate_id] = target
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
        # Rewrite all fanins
        # ------------------------------------------------------------------
        for g in self.gates:
            if g.in0 >= 0:
                g.in0 = resolve(g.in0)

            if g.in1 >= 0:
                g.in1 = resolve(g.in1)

        # ------------------------------------------------------------------
        # Rewrite outputs
        # ------------------------------------------------------------------
        self.output_ids = [resolve(gid) for gid in self.output_ids]

        # ------------------------------------------------------------------
        # Remove aliased gates themselves
        # ------------------------------------------------------------------
        self.gates = [g for g in self.gates if g.gate_id not in alias]


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

            in0 = g.in0
            in1 = g.in1

            # Normalize commutative ops
            if g.op in COMMUTATIVE and in0 > in1:
                in0, in1 = in1, in0

            key = (g.op, in0, in1)

            if key in canonical:
                replace[g.gate_id] = canonical[key]
            else:
                canonical[key] = g.gate_id

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
        # Rewrite fanins
        # ------------------------------------------------------------------
        for g in self.gates:

            if g.in0 >= 0:
                g.in0 = resolve(g.in0)

            if g.in1 >= 0:
                g.in1 = resolve(g.in1)

            # Keep commutative gates normalized afterward
            if g.op in COMMUTATIVE and g.in0 > g.in1:
                g.in0, g.in1 = g.in1, g.in0

        # ------------------------------------------------------------------
        # Rewrite outputs
        # ------------------------------------------------------------------
        self.output_ids = [resolve(gid) for gid in self.output_ids]

        # ------------------------------------------------------------------
        # Remove duplicate gates
        # ------------------------------------------------------------------
        self.gates = [g for g in self.gates if g.gate_id not in replace]
        

    def compile(self) -> None:
        """
        Write C code to a temp file, compile to a shared library, and load it.
        After calling this, circuit(x) will use the compiled implementation.
        """
        import ctypes
        import subprocess
        import tempfile

        c_code = self.get_c_code()
        tmp_c = tempfile.NamedTemporaryFile(suffix='.c', delete=False, mode='w')
        tmp_c.write(c_code)
        tmp_c.close()
        so_path = tmp_c.name.replace('.c', '.so')

        result = subprocess.run(
            ['gcc', '-O2', '-shared', '-fPIC', '-o', so_path, tmp_c.name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")

        lib = ctypes.CDLL(so_path)
        lib.circuit.argtypes = [
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_bool),
        ]
        lib.circuit.restype = None
        self._lib = lib


    def __call__(self, input: torch.Tensor, use_compiled: bool = False) -> torch.Tensor:
        """
        Evaluate the circuit on a given input tensor (shape: batch x n_inputs).

        Uses the compiled C library when available (after compile()), otherwise
        evaluates the gate list in Python.

        If the stored output_shape has 3+ dimensions (e.g. [1, k, group_size]
        from a GroupSum layer), the raw boolean outputs are summed along the
        last axis to reproduce the GroupSum reduction.
        """
        import ctypes

        batch_size = input.shape[0]
        n_out = len(self.output_ids)
        use_c = use_compiled or hasattr(self, '_lib')

        if use_c:
            raw = torch.zeros(batch_size, n_out, dtype=torch.bool)
            for i in range(batch_size):
                in_arr  = (ctypes.c_bool * self.n_inputs)(*input[i].flatten().tolist())
                out_arr = (ctypes.c_bool * n_out)()
                self._lib.circuit(in_arr, out_arr)
                raw[i] = torch.tensor(
                    [bool(out_arr[k]) for k in range(n_out)], dtype=torch.bool
                )
        else:
            raw = torch.zeros(batch_size, n_out, dtype=torch.bool)
            for i in range(batch_size):
                gate_vals: dict[int, bool] = {
                    gid: bool(input[i].flatten()[gid].item())
                    for gid in range(self.n_inputs)
                }
                for g in self.gates:
                    a = gate_vals.get(g.in0, False) if g.in0 >= 0 else False
                    b = gate_vals.get(g.in1, False) if g.in1 >= 0 else False
                    gate_vals[g.gate_id] = _eval_gate_op(g.op, a, b)
                for k, gid in enumerate(self.output_ids):
                    raw[i, k] = gate_vals[gid]

        # Reconstruct GroupSum reduction if output_shape indicates it.
        # A 3-D output_shape [traced_batch, k, group_size] means the FX graph
        # contained a reshape-then-sum (GroupSum) that is not a boolean gate.

        os = self.output_shape
        if len(os) >= 3:
            group_size = os[-1]
            k = os[-2]
            return raw.reshape(batch_size, k, group_size).sum(-1).float()

        return raw.float()


    def get_c_code(self, inline_single_use: bool = True) -> str:
        """
        Generate a self-contained C function that evaluates the circuit.

        inline_single_use=True (default): gates used by only one other gate are
        inlined into their parent's expression rather than emitted as variables.
        This eliminates most temporaries and makes each output a single expression
        tree rooted at the inputs.
        """
        n_in  = self.n_inputs
        n_g   = len(self.gates)
        n_out = len(self.output_ids)

        gate_by_id = {g.gate_id: g for g in self.gates}

        # ------------------------------------------------------------------ #
        # Count how many times each gate's output is consumed                 #
        # ------------------------------------------------------------------ #
        use_count: dict[int, int] = {}
        if inline_single_use:
            for g in self.gates:
                use_count.setdefault(g.gate_id, 0)
                for dep in (g.in0, g.in1):
                    if dep >= n_in:
                        use_count[dep] = use_count.get(dep, 0) + 1
            for oid in self.output_ids:
                if oid >= n_in:
                    use_count[oid] = use_count.get(oid, 0) + 1

        def should_inline(gid: int) -> bool:
            return inline_single_use and use_count.get(gid, 0) <= 1

        # Build expression strings for inlined gates (memoised)
        _expr_cache: dict[int, str] = {}

        def expr_of(gid: int) -> str:
            """Return C expression for gate gid (inlined if single-use)."""
            if gid < 0:
                return "false"
            if gid < n_in:
                return f"in[{gid}]"
            if not should_inline(gid):
                return f"g{gid}"
            if gid in _expr_cache:
                return _expr_cache[gid]
            g = gate_by_id.get(gid)
            if g is None:
                return f"g{gid}"
            a = expr_of(g.in0)
            b = expr_of(g.in1)
            result = GATE_OP_C[g.op].format(a=a, b=b)
            _expr_cache[gid] = result
            return result

        lines = []
        lines.append("// Auto-generated circuit — do not edit")
        lines.append("// Gate IDs 0..{} are inputs, {}..{} are gates".format(
            n_in - 1, n_in, n_in + n_g - 1))
        lines.append("")
        lines.append("#include <stdbool.h>")
        lines.append("#include <stdint.h>")
        lines.append("")
        lines.append(f"// Input shape:  {self.input_shape}")
        lines.append(f"// Output shape: {self.output_shape}")
        lines.append(f"// n_inputs={n_in}  n_gates={n_g}  n_outputs={n_out}")
        lines.append("")
        lines.append(f"void circuit(")
        lines.append(f"    const bool in[{n_in}],")
        lines.append(f"    bool       out[{n_out}])")
        lines.append("{")

        current_layer = -1
        for gate in self.gates:
            if should_inline(gate.gate_id):
                continue  # will be inlined at use site

            if gate.layer != current_layer:
                if current_layer >= 0:
                    lines.append("")
                current_layer = gate.layer
                lines.append(f"    // --- layer {current_layer} ---")

            a = expr_of(gate.in0)
            b = expr_of(gate.in1)
            expr = GATE_OP_C[gate.op].format(a=a, b=b)
            lines.append(f"    bool g{gate.gate_id} = {expr};")

        lines.append("")
        lines.append("    // --- outputs ---")
        for k, gid in enumerate(self.output_ids):
            lines.append(f"    out[{k}] = {expr_of(gid)};")

        lines.append("}")
        return "\n".join(lines)


    def to_dict(self) -> dict:
        """
        Convert the circuit representation to a JSON-serializable format.
        """
        return {
            'n_inputs': self.n_inputs,
            'input_shape': self.input_shape,
            'output_ids': self.output_ids,
            'output_shape': self.output_shape,
            'gates': [
                {
                    'gate_id': g.gate_id,
                    'op': g.op.name,
                    'in0': g.in0,
                    'in1': g.in1,
                    'layer': g.layer,
                    'node_idx': g.node_idx,
                }
                for g in self.gates
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Circuit:
        """
        Create a Circuit instance from a (JSON-deserialized) dictionary.
        """
        circuit = cls(n_inputs=data['n_inputs'])
        circuit.input_shape = data.get('input_shape', [])
        circuit.output_ids = data.get('output_ids', [])
        circuit.output_shape = data.get('output_shape', [])
        for g_data in data.get('gates', []):
            g = Gate(
                gate_id=g_data['gate_id'],
                op=GateOp[g_data['op']],
                in0=g_data['in0'],
                in1=g_data['in1'],
                layer=g_data.get('layer', -1),
                node_idx=g_data.get('node_idx', -1),
            )
            circuit.gates.append(g)
        return circuit
    
    @classmethod
    def from_json_file(cls, file_path: str) -> Circuit:
        """
        Load a Circuit instance from a JSON file.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_lisp_code(self, inline_single_use: bool) -> str:
        """
        Generate a Lisp-like text representation of the circuit.
        """
        n_in = self.n_inputs
        gate_by_id = {g.gate_id: g for g in self.gates}

        # ---- use-count for optional inlining --------------------------------
        use_count: dict[int, int] = {}
        if inline_single_use:
            for g in self.gates:
                use_count.setdefault(g.gate_id, 0)
                for dep in (g.in0, g.in1):
                    if dep >= n_in:
                        use_count[dep] = use_count.get(dep, 0) + 1
            for oid in self.output_ids:
                if oid >= n_in:
                    use_count[oid] = use_count.get(oid, 0) + 1

        def should_inline(gid: int) -> bool:
            return inline_single_use and use_count.get(gid, 0) <= 1

        _expr_cache: dict[int, str] = {}

        def lisp_expr(gid: int) -> str:
            if gid < 0:
                return "_"
            if gid < n_in:
                return f"in[{gid}]"
            if not should_inline(gid):
                return f"g{gid}"
            if gid in _expr_cache:
                return _expr_cache[gid]
            g = gate_by_id.get(gid)
            if g is None:
                return f"g{gid}"
            a = lisp_expr(g.in0)
            b = lisp_expr(g.in1)
            result = GATE_OP_LISP[g.op].format(a=a, b=b)
            _expr_cache[gid] = result
            return result

        # ---- build output ---------------------------------------------------
        lines = []
        lines.append(f"(circuit")
        lines.append(f"  (meta")
        lines.append(f"    (n-inputs  {self.n_inputs})")
        lines.append(f"    (n-gates   {len(self.gates)})")
        lines.append(f"    (n-outputs {len(self.output_ids)})")
        lines.append(f"    (input-shape  {self.input_shape})")
        lines.append(f"    (output-shape {self.output_shape}))")
        lines.append("")
        lines.append(f"  (inputs")
        lines.append(f"    ; IDs 0..{self.n_inputs - 1} map flat into input tensor")
        lines.append(f"    (shape {self.input_shape}))")
        lines.append("")

        current_layer = -1
        for gate in self.gates:
            if should_inline(gate.gate_id):
                continue  # will appear inlined at its use site

            if gate.layer != current_layer:
                if current_layer >= 0:
                    lines.append(f"  ) ; end layer {current_layer}")
                    lines.append("")
                current_layer = gate.layer
                lines.append(f"  (layer {current_layer}")

            a_str = lisp_expr(gate.in0)
            b_str = lisp_expr(gate.in1)
            expr  = GATE_OP_LISP[gate.op].format(a=a_str, b=b_str)
            lines.append(f"    (gate g{gate.gate_id} {expr})")

        if current_layer >= 0:
            lines.append(f"  ) ; end layer {current_layer}")

        lines.append("")
        lines.append(f"  (outputs ; shape {self.output_shape}")
        for k, gid in enumerate(self.output_ids):
            lines.append(f"    {lisp_expr(gid)}")
        lines.append("  )")
        lines.append(")")
        return "\n".join(lines)


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
        n_in  = self.n_inputs
        n_out = len(self.output_ids)
        gate_by_id = {g.gate_id: g for g in self.gates}

        # ---- use-count for optional inlining --------------------------------
        use_count: dict[int, int] = {}
        if inline_single_use:
            for g in self.gates:
                use_count.setdefault(g.gate_id, 0)
                for dep in (g.in0, g.in1):
                    if dep >= n_in:
                        use_count[dep] = use_count.get(dep, 0) + 1
            for oid in self.output_ids:
                if oid >= n_in:
                    use_count[oid] = use_count.get(oid, 0) + 1

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
            a = vexpr(g.in0)
            b = vexpr(g.in1)
            result = GATE_OP_VERILOG[g.op].format(a=a, b=b)
            _expr_cache[gid] = result
            return result

        lines = []
        lines.append(f"// Auto-generated by circuit_ir — do not edit")
        lines.append(f"// Input shape:  {self.input_shape}")
        lines.append(f"// Output shape: {self.output_shape}")
        lines.append(f"// n_inputs={n_in}  n_gates={len(self.gates)}  n_outputs={n_out}")
        lines.append("")
        lines.append(f"module circuit (")
        lines.append(f"    input  wire [{n_in - 1}:0] inp,")
        lines.append(f"    output wire [{n_out - 1}:0] out")
        lines.append(f");")

        current_layer = -1
        for gate in self.gates:
            if should_inline(gate.gate_id):
                continue

            if gate.layer != current_layer:
                lines.append("")
                current_layer = gate.layer
                lines.append(f"    // --- layer {current_layer} ---")

            a = vexpr(gate.in0)
            b = vexpr(gate.in1)
            expr = GATE_OP_VERILOG[gate.op].format(a=a, b=b)
            lines.append(f"    wire g{gate.gate_id} = {expr};")

        lines.append("")
        lines.append("    // --- outputs ---")
        for k, gid in enumerate(self.output_ids):
            lines.append(f"    assign out[{k}] = {vexpr(gid)};")

        lines.append("")
        lines.append("endmodule")
        return "\n".join(lines)


    def write_c_code(self, path: str) -> None:
        """
        Write the generated C code to a file.
        """
        c_code = self.get_c_code()
        with open(path, 'w') as f:
            f.write(c_code)

    def write_lisp_code(self, path: str) -> None:
        """
        Write the generated Lisp code to a file.
        """
        lisp_code = self.get_lisp_code()
        with open(path, 'w') as f:
            f.write(lisp_code)

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
        torch.ops.aten.select.int,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.moveaxis.int,
        torch.ops.aten.unbind.int,
        torch.ops.aten.lift_fresh_copy.default,
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
