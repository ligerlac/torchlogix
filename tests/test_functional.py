"""Tests for the primitives in torchlogix.functional.

These are the bottom of the pyramid: each test pins one function against an
independent definition of what it should compute, rather than against another
part of torchlogix. That way a change in the layers cannot mask a change here.

Most of functional.py was previously only exercised indirectly, through layers,
so nothing was checking these contracts on their own.
"""
import pytest
import torch

from torchlogix.functional import (
    bin_op,
    compute_all_logic_ops_vectorized,
    fwht,
    hadamard_matrix,
    id_to_truth_table,
    id_to_walsh_coefficients,
    take_tuples,
    walsh_hadamard_transform,
)

# The 16 two-input boolean functions, in the order compute_all_logic_ops_
# vectorized emits them. Written as plain Python booleans so the reference is
# independent of the tensor implementation under test.
LOGIC_OPS = [
    (0,  "false",     lambda a, b: False),
    (1,  "a and b",   lambda a, b: a and b),
    (2,  "a and !b",  lambda a, b: a and not b),
    (3,  "a",         lambda a, b: a),
    (4,  "b and !a",  lambda a, b: b and not a),
    (5,  "b",         lambda a, b: b),
    (6,  "a xor b",   lambda a, b: a != b),
    (7,  "a or b",    lambda a, b: a or b),
    (8,  "!(a or b)", lambda a, b: not (a or b)),
    (9,  "!(a xor b)", lambda a, b: a == b),
    (10, "!b",        lambda a, b: not b),
    (11, "b -> a",    lambda a, b: a or not b),
    (12, "!a",        lambda a, b: not a),
    (13, "a -> b",    lambda a, b: b or not a),
    (14, "!(a and b)", lambda a, b: not (a and b)),
    (15, "true",      lambda a, b: True),
]


@pytest.mark.parametrize("op_id, name, reference", LOGIC_OPS)
@pytest.mark.parametrize("a", [0.0, 1.0])
@pytest.mark.parametrize("b", [0.0, 1.0])
def test_all_logic_ops_match_boolean_truth_tables(op_id, name, reference, a, b):
    """On binary inputs, op `op_id` must equal the boolean function it stands for."""
    ops = compute_all_logic_ops_vectorized(torch.tensor([a]), torch.tensor([b]))
    expected = float(reference(bool(a), bool(b)))
    assert ops[0, op_id].item() == pytest.approx(expected), (
        f"op {op_id} ({name}) wrong for a={a}, b={b}"
    )


@pytest.mark.parametrize("op_id", range(16))
def test_bin_op_agrees_with_vectorized_ops(op_id):
    """The scalar dispatch and the vectorized stack must not disagree."""
    torch.manual_seed(0)
    a, b = torch.rand(2, 4), torch.rand(2, 4)
    assert torch.allclose(bin_op(a, b, op_id),
                          compute_all_logic_ops_vectorized(a, b)[..., op_id],
                          atol=1e-6)


# ---------------------------------------------------------------------------
# Walsh-Hadamard transform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_hadamard_matrix_is_orthogonal(n):
    """H @ H.T == size * I, which is what makes the transform invertible."""
    size = 1 << n
    H = hadamard_matrix(size)
    assert H.shape == (size, size)
    assert torch.allclose(H @ H.T, size * torch.eye(size), atol=1e-5)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_fwht_matches_dense_hadamard_product(n):
    """The fast transform must equal multiplication by the Hadamard matrix."""
    torch.manual_seed(0)
    x = torch.rand(3, 1 << n)
    assert torch.allclose(fwht(x, n), x @ hadamard_matrix(1 << n), atol=1e-5)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_walsh_transform_fast_and_dense_paths_agree(n):
    """fast=True and fast=False are two implementations of one function."""
    torch.manual_seed(0)
    x = torch.rand(3, 1 << n)
    assert torch.allclose(walsh_hadamard_transform(x, n, fast=True),
                          walsh_hadamard_transform(x, n, fast=False), atol=1e-5)


# ---------------------------------------------------------------------------
# Gate id <-> truth table <-> Walsh coefficients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 2, 3])
def test_id_to_truth_table_expands_bits_most_significant_first(rank):
    """A gate id is its truth table as bits, mapped {0,1} -> {+1,-1}.

    Bit order is most-significant-first, so id 1 sets only the last entry.
    """
    lut_entries = 1 << rank
    ids = torch.arange(1 << lut_entries)
    table = id_to_truth_table(ids, rank, device=None)

    assert table.shape == (1 << lut_entries, lut_entries)
    assert set(table.flatten().tolist()) <= {-1, 1}

    # all-zeros id -> all +1; all-ones id -> all -1
    assert table[0].tolist() == [1] * lut_entries
    assert table[-1].tolist() == [-1] * lut_entries
    # id 1 has only its least significant bit set, i.e. the final entry
    assert table[1].tolist() == [1] * (lut_entries - 1) + [-1]


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_walsh_coefficients_transform_back_to_the_truth_table(rank):
    """The transform is its own inverse up to the 2^rank scaling the coeffs undo.

    id -> truth -> coefficients -> transform must land back on truth.
    """
    ids = torch.arange(1 << (1 << rank))
    truth = id_to_truth_table(ids, rank, device=None).float()
    coeffs = id_to_walsh_coefficients(ids, rank, device=None)
    assert torch.allclose(walsh_hadamard_transform(coeffs, n=rank), truth, atol=1e-5)


# ---------------------------------------------------------------------------
# Strided tuple gather
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tuple_size, start, stride_within, step_between", [
    (2, 0, 1, 1),   # overlapping adjacent pairs
    (2, 0, 1, 2),   # disjoint adjacent pairs
    (3, 0, 1, 3),   # disjoint triples
    (2, 1, 2, 2),   # offset start, gap inside the tuple
    (4, 0, 2, 1),   # dilated tuple, overlapping
])
def test_take_tuples_gathers_the_indices_it_promises(tuple_size, start,
                                                     stride_within, step_between):
    """y[..., i, g] must be x[..., start + g * step_between + i * stride_within].

    Checked against an explicit index-by-index reference rather than another
    torchlogix helper, so the two cannot be wrong in the same way.
    """
    x = torch.arange(12).float()
    y = take_tuples(x, tuple_size, start, stride_within, step_between)

    assert y.shape[0] == tuple_size
    n_groups = y.shape[-1]
    reference = torch.stack([
        torch.stack([x[start + g * step_between + i * stride_within]
                     for g in range(n_groups)])
        for i in range(tuple_size)
    ])
    assert torch.equal(y, reference)


def test_take_tuples_preserves_leading_dims():
    """Only the last dimension is consumed; batch dims pass through."""
    x = torch.arange(2 * 3 * 8).float().reshape(2, 3, 8)
    y = take_tuples(x, 2, 0, 1, 2)
    assert y.shape[:2] == (2, 3)
    assert torch.equal(y[1, 2], take_tuples(x[1, 2], 2, 0, 1, 2))
