"""Per-group GroupSum scales and the opt-in fixed-point Verilog output."""

import shutil
import subprocess

import numpy as np
import pytest
import torch

from torchlogix.circuit import Circuit
from torchlogix.layers import GroupSum, LogicDense


def test_scalar_groupsum_keeps_existing_behavior():
    layer = GroupSum(2, 2.0, 1.0)
    x = torch.tensor([[0.2, 0.5, 0.7, 0.1]], requires_grad=True)
    expected = (x.reshape(1, 2, 2).sum(-1) + 1.0) / 2.0
    torch.testing.assert_close(layer(x), expected)
    layer(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.5))
    assert layer.state_dict() == {}

    # Existing model code resets these public attributes before raw-count export.
    layer.tau = 1.0
    layer.beta = 0.0
    torch.testing.assert_close(layer(x), x.reshape(1, 2, 2).sum(-1))


@pytest.mark.parametrize("tau,beta", [
    ([2.0, 4.0], [-2.0, 1.0]),
    (2.0, [-2.0, 1.0]),
    ([2.0, 4.0], -2.0),
])
def test_groupsum_broadcasts_per_group_scales(tau, beta):
    layer = GroupSum(2, tau=tau, beta=beta)
    x = torch.arange(12, dtype=torch.float32).reshape(2, 6).requires_grad_()
    counts = x.reshape(2, 2, 3).sum(-1)
    expected = (counts + torch.as_tensor(beta)) / torch.as_tensor(tau)
    torch.testing.assert_close(layer(x), expected)
    layer(x).sum().backward()
    expected_grad = (1 / torch.as_tensor(tau)).expand(2, 2).repeat_interleave(3, -1)
    torch.testing.assert_close(x.grad, expected_grad)
    assert set(layer.state_dict()) == (
        ({"_tau_vector"} if isinstance(tau, list) else set())
        | ({"_beta_vector"} if isinstance(beta, list) else set())
    )


def test_groupsum_rejects_bad_vectors():
    with pytest.raises(ValueError, match="length-2"):
        GroupSum(2, tau=[1.0])
    with pytest.raises(ValueError, match="positive"):
        GroupSum(2, tau=[1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        GroupSum(2, beta=[0.0, float("nan")])


def test_vector_scales_follow_module_dtype_and_can_be_reset():
    layer = GroupSum(2, tau=[2.0, 4.0], beta=[-1.0, 1.0]).to(dtype=torch.float64)
    x = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    assert layer.tau.dtype == torch.float64
    assert layer(x).dtype == torch.float64
    layer.tau = [4.0, 8.0]
    assert layer.tau.dtype == torch.float64
    layer.tau = 1.0
    layer.beta = 0.0
    assert layer.tau == 1.0 and layer.beta == 0.0
    torch.testing.assert_close(layer(x), torch.tensor([[1, 1]]))
    assert layer.state_dict() == {}


def test_vector_scales_survive_state_dict_round_trip():
    original = GroupSum(2, tau=[2.0, 4.0], beta=[-1.0, 1.0])
    restored = GroupSum(2, tau=[1.0, 1.0], beta=[0.0, 0.0])
    restored.load_state_dict(original.state_dict())
    x = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    torch.testing.assert_close(restored(x), original(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_reassigned_vector_stays_on_module_device():
    layer = GroupSum(2, tau=[2.0, 4.0]).cuda()
    layer.tau = [4.0, 8.0]
    assert layer.tau.device.type == "cuda"
    assert layer(torch.ones(1, 4, dtype=torch.bool, device="cuda")).device.type == "cuda"


@pytest.mark.parametrize("tau,beta", [
    ([2.0, 4.0], [-2.0, 1.0]),
    (2.0, [-2.0, 1.0]),
    ([2.0, 4.0], -2.0),
    (2.0, 1.0),
    (1.0, -2.0),
])
def test_circuit_and_c_match_groupsum(tau, beta):
    model = GroupSum(2, tau=tau, beta=beta)
    x = torch.tensor([[(i >> j) & 1 for j in range(6)] for i in range(64)], dtype=torch.bool)
    expected = model(x)
    circuit = Circuit.from_model(model, input_shape=(6,))
    assert len(circuit.sum_nodes) == 2
    torch.testing.assert_close(circuit(x).float(), expected.float())

    restored = Circuit.from_dict(circuit.to_dict())
    torch.testing.assert_close(restored(x).float(), expected.float())

    for pack_bits in (None, 8):
        circuit.compile(pack_bits=pack_bits)
        actual = torch.from_numpy(circuit(x.numpy(), use_compiled=True))
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-6, atol=1e-6)


def test_circuit_keeps_offset_added_after_groupsum():
    class ShiftedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = GroupSum(2, tau=[2.0, 4.0])
            self.register_buffer("shift", torch.tensor([1.0, -2.0]))

        def forward(self, x):
            return self.head(x) + self.shift

    model = ShiftedHead()
    x = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    circuit = Circuit.from_model(model, input_shape=(4,))
    torch.testing.assert_close(circuit(x), model(x))


def _simulate_verilog(code, n_inputs, n_outputs, tmp_path, signed=False):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog is not installed")
    circuit_path = tmp_path / "circuit.v"
    bench_path = tmp_path / "bench.v"
    executable = tmp_path / "sim.out"
    circuit_path.write_text(code)
    score_bits = 32
    fields = ", ".join(
        f"scores_flat[{j}*{score_bits} +: {score_bits}]" for j in range(n_outputs)
    )
    bench_path.write_text(f"""
module bench;
    reg [{n_inputs - 1}:0] inp;
    wire [{n_outputs * score_bits - 1}:0] scores_flat;
    circuit dut(.inp(inp), .scores_flat(scores_flat));
    integer i;
    initial begin
        for (i = 0; i < {1 << n_inputs}; i = i + 1) begin
            inp = i;
            #1;
            $display("{(' %0d' * n_outputs).strip()}", {fields});
        end
        $finish;
    end
endmodule
""")
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(executable), str(circuit_path), str(bench_path)],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run(["vvp", str(executable)], check=True, capture_output=True, text=True)
    rows = [[int(v) for v in line.split()] for line in result.stdout.splitlines()]
    if signed:
        rows = [[v - (1 << 32) if v >= 1 << 31 else v for v in row] for row in rows]
    return np.asarray(rows)


def test_verilog_default_remains_raw_and_fixed_point_matches_regression(tmp_path):
    model = GroupSum(2, tau=[3.0, 2.5], beta=[-1.5, 0.5])
    circuit = Circuit.from_model(model, input_shape=(6,))
    x = torch.tensor([[(i >> j) & 1 for j in range(6)] for i in range(64)], dtype=torch.bool)

    raw = _simulate_verilog(circuit.get_verilog_code(), 6, 2, tmp_path)
    np.testing.assert_array_equal(raw, x.reshape(64, 2, 3).sum(-1).numpy())

    fractional_bits = 16
    code = circuit.get_verilog_code(fixed_point_fractional_bits=fractional_bits)
    fixed = _simulate_verilog(code, 6, 2, tmp_path, signed=True)
    expected = model(x).numpy()
    np.testing.assert_allclose(fixed / (1 << fractional_bits), expected,
                               rtol=0, atol=1 / (1 << fractional_bits))
    written = tmp_path / "written.v"
    circuit.write_verilog_code(written, fixed_point_fractional_bits=fractional_bits)
    assert written.read_text() == code


@pytest.mark.parametrize("tau,beta,score_bits", [
    (1.0, 0.0, 8),
    (1.0, -1.0, 8),
    (2.0, 0.0, 32),
])
def test_default_verilog_keeps_scalar_score_width(tau, beta, score_bits):
    circuit = Circuit.from_model(GroupSum(1, tau=tau, beta=beta), input_shape=(4,))
    assert f"output reg  [{score_bits - 1}:0] scores_flat" in circuit.get_verilog_code()


def test_fixed_point_verilog_matches_logic_model(tmp_path):
    torch.manual_seed(1)
    model = torch.nn.Sequential(
        LogicDense(8, 12, parametrization="raw",
                   parametrization_kwargs={"weight_init": "random"}),
        GroupSum(3, tau=[2.5, 3.0, 4.0], beta=[-2.0, 0.5, 1.0]),
    )
    circuit = Circuit.from_model(model, input_shape=(8,))
    circuit.simplify()
    x = torch.tensor([[(i >> j) & 1 for j in range(8)] for i in range(256)], dtype=torch.bool)
    code = circuit.get_verilog_code(fixed_point_fractional_bits=16)
    actual = _simulate_verilog(code, 8, 3, tmp_path, signed=True) / (1 << 16)
    np.testing.assert_allclose(actual, model(x).numpy(), rtol=0, atol=1 / (1 << 16))

    if shutil.which("verilator"):
        subprocess.run(
            ["verilator", "--lint-only", "-Wno-fatal", str(tmp_path / "circuit.v")],
            check=True, capture_output=True, text=True,
        )


def test_fixed_point_rejects_unrepresentable_range():
    circuit = Circuit.from_model(GroupSum(1, tau=1e-8), input_shape=(2,))
    with pytest.raises(ValueError, match="range"):
        circuit.get_verilog_code(fixed_point_fractional_bits=16)
    with pytest.raises(ValueError, match="0 to 30"):
        circuit.get_verilog_code(fixed_point_fractional_bits=31)
