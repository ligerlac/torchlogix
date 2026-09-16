# The torchlogix test suite

Run everything:

```bash
pytest -s
```

Skip the handful of tests that pay a high one-off cost (e.g. compilation):

```bash
pytest -s -m "not slow"
```

Check coverage (needs the `dev` extra):

```bash
pytest --cov=torchlogix --cov-branch --cov-report=term-missing
```
