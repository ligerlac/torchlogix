# The torchlogix test suite

Run everything:

```bash
pytest
```

Skip the handful of tests that pay a one-off `torch.compile` cost:

```bash
pytest -m "not slow"
```

Check coverage (needs the `dev` extra):

```bash
pytest --cov=torchlogix --cov-branch --cov-report=term-missing
```

## Where things live

The suite is organised by **what a test asserts**, not by which class it
touches. That is the one rule worth remembering: it is why there is no
`test_conv2d.py` and `test_conv3d.py` pair.

| File | Holds |
|---|---|
| `models.py` | The shared models, as plain factory functions. Not a test file. |
| `helpers.py` | Shared assertions (gradient checks). Not a test file. |
| `conftest.py` | Just the OpenMP workaround; the models live in `models.py`. |
| `test_functional.py` | Primitives in `functional.py`, each against an independent reference. |
| `test_parametrization.py` | Weight parametrizations. |
| `test_connections.py` | Connection index generation, dense and convolutional. |
| `test_binarization.py` | Binarization layers. |
| `test_layers.py` | Layer properties that hold **by formula** — shapes, gradients, regularization. |
| `test_gate_semantics.py` | Layer **behaviour** pinned by hand-computed truth tables. |
| `test_models.py` | Whole-model properties, incl. eval-vs-export and exported-graph purity. |
| `test_circuit.py` | Circuit build, equivalence, compilation, disk round-trip, AIG. |
| `test_alkaid_plugin.py` | The optional alkaid integration (skipped if not installed). |

## Adding a test

**A new model.** Write a function in `models.py` that builds and returns it,
and append that function to `MODELS`:

```python
def my_model():
    model = nn.Sequential(LogicDense(64, 32), LogicDense(32, 16))
    model.input_shape = (64,)
    model.input_dtype = torch.float32
    return model
```

That is the whole job — it now inherits every model-level property test:
gradient correctness, state-dict round-trip, eval-vs-export agreement,
exported-graph purity, circuit equivalence, compilation, AIG conversion. Use a
class only if the model needs a custom `forward` (see `BranchModel`). If it is
not built from torchlogix layers, `TORCHLOGIX_MODELS` excludes it.

**A variant of an existing model.** Build it in the test rather than adding a
flag to the model. A `nn.Sequential` supports `del model[-1]`, so dropping the
`GroupSum` head is one line where it is needed.

**A new layer configuration to sweep.** Add one `pytest.param(...)` line to
`CONV_CONFIGS` in `test_connections.py`, or to `TRANSPOSE_FAMILIES` /
`GRAD_CHECK_KINDS` in `test_layers.py`.

**Anything else.** Write a plain function in the file that matches what you are
asserting:

```python
def test_the_thing_does_what_it_should():
    ...
    assert ...
```

Use `@pytest.mark.parametrize` freely. Please do not introduce test classes or
nested test helpers unless a plain function genuinely cannot express it.

## Two conventions that exist for a reason

**Prefer a table of valid configurations over a cartesian product.** The suite
used to generate ~5,900 index tests from nine parametrize axes and then throw
half of them away with `pytest.skip` inside a fixture, because the product kept
producing combinations that are invalid by construction. Listing the valid
configurations explicitly is shorter, has no skips, and is far easier to read.
A skip should mean "not applicable on this machine", not "we generated
something impossible".

**Seed anything whose result you compare numerically.** Layer construction
draws from the global RNG, so an unseeded layer's weights — and therefore its
gradient magnitudes — depend on how many tests happened to run before it. That
produces tests which pass alone and fail in a full run.

## Gradient tests

`helpers.assert_finite_difference_matches_autograd` compares autograd against
central finite differences. Two things about it are counter-intuitive:

- It needs **train mode**. The logic layers detach in eval mode, so there is no
  gradient there to check.
- Its step size is `1e-2`, and making it *smaller* makes it *worse*. These
  losses reach ~500 in float32 while the gradients are ~0.1, so a `1e-3` step
  moves the loss by about float32's last significant digit and cancellation
  alone gives 25–30% error.
