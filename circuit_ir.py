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
from typing import Optional
import textwrap

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
    output_ids:  list[int]  = field(default_factory=list)   # flat, in order
    output_shape: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _get_attr_val(gm: torch.fx.GraphModule, node: torch.fx.Node):
    obj = gm
    for part in node.target.split('.'):
        obj = getattr(obj, part)
    return obj


def build_circuit(gm: torch.fx.GraphModule, input_shape: list[int]) -> Circuit:
    """
    Walk the folded FX graph and build a flat Circuit.

    Assumptions (satisfied after constant_fold_views):
      - Exactly one placeholder node ('input')
      - Wiring is done via aten.index.Tensor with folded constant index tensors
      - LUT dispatch is a cascade of aten.eq + aten.where nodes
      - Layers are connected by further aten.index.Tensor nodes
    """
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
                    print("Found where in LUT cascade")
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
                if lut_id == 10:
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


# ---------------------------------------------------------------------------
# Lisp IR emitter
# ---------------------------------------------------------------------------

def emit_lisp(circuit: Circuit) -> str:
    lines = []
    lines.append(f"(circuit")
    lines.append(f"  (meta")
    lines.append(f"    (n-inputs  {circuit.n_inputs})")
    lines.append(f"    (n-gates   {len(circuit.gates)})")
    lines.append(f"    (n-outputs {len(circuit.output_ids)})")
    lines.append(f"    (input-shape  {circuit.input_shape})")
    lines.append(f"    (output-shape {circuit.output_shape}))")
    lines.append("")
    lines.append(f"  (inputs")
    lines.append(f"    ; IDs 0..{circuit.n_inputs - 1} map flat into input tensor")
    lines.append(f"    (shape {circuit.input_shape}))")
    lines.append("")

    current_layer = -1
    for gate in circuit.gates:
        if gate.layer != current_layer:
            if current_layer >= 0:
                lines.append(f"  ) ; end layer {current_layer}")
                lines.append("")
            current_layer = gate.layer
            lines.append(f"  (layer {current_layer}")

        def id_str(gid):
            if gid < 0:
                return "_"
            if gid < circuit.n_inputs:
                return f"in[{gid}]"
            return f"g{gid}"

        a_str = id_str(gate.in0)
        b_str = id_str(gate.in1)
        expr  = GATE_OP_LISP[gate.op].format(a=a_str, b=b_str)
        lines.append(f"    (gate g{gate.gate_id} {expr})")

    if current_layer >= 0:
        lines.append(f"  ) ; end layer {current_layer}")

    lines.append("")
    lines.append(f"  (outputs ; shape {circuit.output_shape}")
    out_strs = [f"g{gid}" for gid in circuit.output_ids]
    # wrap at 80 chars
    chunk = 12
    for k in range(0, len(out_strs), chunk):
        lines.append("    " + " ".join(out_strs[k:k+chunk]))
    lines.append("  )")
    lines.append(")")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# C emitter
# ---------------------------------------------------------------------------

def emit_c(circuit: Circuit, func_name: str = "circuit_eval") -> str:
    n_in  = circuit.n_inputs
    n_g   = len(circuit.gates)
    n_out = len(circuit.output_ids)

    def cvar(gid: int) -> str:
        if gid < 0:
            return "false"  # unused input
        if gid < n_in:
            return f"in[{gid}]"
        return f"g{gid}"

    lines = []
    lines.append("// Auto-generated circuit — do not edit")
    lines.append("// Gate IDs 0..{} are inputs, {}..{} are gates".format(
        n_in - 1, n_in, n_in + n_g - 1))
    lines.append("")
    lines.append("#include <stdbool.h>")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"// Input shape:  {circuit.input_shape}")
    lines.append(f"// Output shape: {circuit.output_shape}")
    lines.append(f"// n_inputs={n_in}  n_gates={n_g}  n_outputs={n_out}")
    lines.append("")
    lines.append(f"void {func_name}(")
    lines.append(f"    const bool in[{n_in}],")
    lines.append(f"    bool       out[{n_out}])")
    lines.append("{")

    current_layer = -1
    for gate in circuit.gates:
        if gate.layer != current_layer:
            if current_layer >= 0:
                lines.append("")
            current_layer = gate.layer
            lines.append(f"    // --- layer {current_layer} ---")

        a = cvar(gate.in0)
        b = cvar(gate.in1)
        expr = GATE_OP_C[gate.op].format(a=a, b=b)
        lines.append(f"    bool g{gate.gate_id} = {expr};")

    lines.append("")
    lines.append("    // --- outputs ---")
    for k, gid in enumerate(circuit.output_ids):
        lines.append(f"    out[{k}] = {cvar(gid)};")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def serialize_circuit(gm: torch.fx.GraphModule,
                      input_shape: list[int],
                      lisp_path: str = "circuit.ir",
                      c_path:    str = "circuit.c",
                      func_name: str = "circuit_eval"):
    circuit = build_circuit(gm, input_shape)

    lisp_ir = emit_lisp(circuit)
    with open(lisp_path, "w") as f:
        f.write(lisp_ir)

    c_src = emit_c(circuit, func_name=func_name)
    with open(c_path, "w") as f:
        f.write(c_src)

    print(f"Circuit: {circuit.n_inputs} inputs, "
          f"{len(circuit.gates)} gates, "
          f"{len(circuit.output_ids)} outputs")
    print(f"Wrote Lisp IR -> {lisp_path}")
    print(f"Wrote C       -> {c_path}")

    return circuit
