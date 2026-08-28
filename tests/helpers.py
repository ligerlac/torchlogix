"""Shared assertions for the test suite.

Imported, not collected - pytest only collects ``test_*.py``.
"""
import torch


def weighted_loss(module, x, weights):
    """Scalar loss with a fixed random weighting on the output.

    A plain ``.sum()`` makes gradients cancel symmetrically and can be exactly
    zero for whole parameters, which would make a finite-difference comparison
    pass without testing anything. Weighting the output avoids that.
    """
    return (module(x) * weights).sum()


def assert_finite_difference_matches_autograd(module, x, weights, n_probes=4,
                                             eps=1e-2, tol=5e-2):
    """Central finite differences must agree with autograd on real gradients.

    On eps: the models here are float32 and their losses run into the hundreds,
    while the gradients being probed are ~0.1. A central difference at eps=1e-3
    changes the loss by ~2e-4, which is about float32's last significant digit
    at that magnitude, so cancellation alone produced 25-30% error. eps=1e-2
    keeps the difference roughly 40x above the noise floor and brings the
    relative error under 1%. Do not shrink it "for accuracy" - it does the
    opposite here.

    The module must be in train mode: the logic layers detach in eval mode, so
    there is no gradient to check there. The forward also has to be
    deterministic, which it is for every parametrization once training=True -
    otherwise the two probe evaluations would differ for the wrong reason.

    Probes the largest-magnitude gradient entries rather than arbitrary ones,
    so the comparison is made where the gradient actually carries signal.
    """
    assert module.training, "finite differences need train mode; eval detaches"

    params = [p for p in module.parameters() if p.requires_grad]
    assert params, "module has no trainable parameters"

    module.zero_grad()
    weighted_loss(module, x, weights).backward()

    # Determinism: two identical forwards must give the same value, or the
    # probes below are measuring noise.
    with torch.no_grad():
        a = weighted_loss(module, x, weights).item()
        b = weighted_loss(module, x, weights).item()
    assert abs(a - b) < 1e-9, "forward is not deterministic; cannot use finite differences"

    checked = 0
    for param in params:
        if param.grad is None:
            continue
        flat_grad = param.grad.flatten()
        n = min(n_probes, flat_grad.numel())
        for flat_idx in flat_grad.abs().topk(n).indices.tolist():
            if flat_grad[flat_idx] == 0:
                continue    # nothing to compare against
            idx = torch.unravel_index(torch.tensor(flat_idx), param.shape)
            original = param.data[idx].clone()
            with torch.no_grad():
                param.data[idx] = original + eps
                loss_plus = weighted_loss(module, x, weights).item()
                param.data[idx] = original - eps
                loss_minus = weighted_loss(module, x, weights).item()
                param.data[idx] = original

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic = flat_grad[flat_idx].item()
            assert abs(numerical - analytic) <= tol * max(1.0, abs(analytic)), (
                f"gradient mismatch at {idx}: autograd {analytic:+.6f} vs "
                f"finite difference {numerical:+.6f}"
            )
            checked += 1

    assert checked > 0, "no non-zero gradient entries were available to check"


def assert_compiled_gradients_match_eager(module, x, weights, tol=1e-5):
    """torch.compile must not change the gradients it produces."""
    module.zero_grad()
    weighted_loss(module, x, weights).backward()
    eager = [p.grad.clone() for p in module.parameters() if p.grad is not None]

    module.zero_grad()
    weighted_loss(torch.compile(module), x, weights).backward()
    compiled = [p.grad.clone() for p in module.parameters() if p.grad is not None]

    assert len(eager) == len(compiled) and eager, "gradient sets differ in length"
    for i, (a, b) in enumerate(zip(eager, compiled)):
        assert torch.allclose(a, b, atol=tol), (
            f"parameter {i}: compiled gradient differs from eager "
            f"(max |delta| = {(a - b).abs().max():.3e})"
        )
