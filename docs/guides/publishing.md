# Publishing torchlogix to PyPI

This guide explains how to publish the torchlogix package to PyPI so it can be installed via `pip install torchlogix`.

## Prerequisites

1. **PyPI Account**: Create an account at [https://pypi.org](https://pypi.org)
2. **TestPyPI Account** (optional but recommended): Create an account at [https://test.pypi.org](https://test.pypi.org)
3. **API Token**: Generate an API token from your PyPI account settings

## Step 1: Install Build Tools

```bash
pip install --upgrade pip build twine
```

## Step 2: Clean Previous Builds

```bash
rm -rf build dist *.egg-info src/*.egg-info
```

## Step 3: Build the Package

```bash
python -m build
```

This will create two files in the `dist/` directory:
- `torchlogix-0.1.0.tar.gz` (source distribution)
- `torchlogix-0.1.0-py3-none-any.whl` (wheel distribution)

## Step 4: Validate the Build

```bash
python -m twine check dist/*
```

You should see:
```
Checking dist/torchlogix-0.1.0-py3-none-any.whl: PASSED
Checking dist/torchlogix-0.1.0.tar.gz: PASSED
```

## Step 5: Test on TestPyPI (Recommended)

Before publishing to the real PyPI, test on TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for your username (use `__token__`) and your TestPyPI API token.

Then test installing from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps torchlogix
```

## Step 6: Publish to PyPI

Once you've verified everything works on TestPyPI:

```bash
python -m twine upload dist/*
```

You'll be prompted for your username (use `__token__`) and your PyPI API token.

## Step 7: Verify Installation

After publishing, verify that the package can be installed:

```bash
pip install torchlogix
```

## Using API Tokens (Recommended)

Instead of entering credentials each time, you can create a `.pypirc` file in your home directory:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

**Important**: Never commit `.pypirc` to version control!

## Updating the Package

When you want to release a new version:

1. Update the version number in:
   - `setup.py` (line 13)
   - `pyproject.toml` (line 7)

2. Update `CHANGELOG.md` with the changes

3. Commit the changes:
   ```bash
   git add setup.py pyproject.toml CHANGELOG.md
   git commit -m "Bump version to X.Y.Z"
   git tag vX.Y.Z
   git push && git push --tags
   ```

4. Clean, rebuild, and republish:
   ```bash
   rm -rf build dist *.egg-info src/*.egg-info
   python -m build
   python -m twine check dist/*
   python -m twine upload dist/*
   ```

## Versioning

This package follows [Semantic Versioning](https://semver.org/):
- MAJOR version (X.0.0): Incompatible API changes
- MINOR version (0.X.0): Add functionality (backwards-compatible)
- PATCH version (0.0.X): Bug fixes (backwards-compatible)

## Troubleshooting

### "File already exists" error

If you get this error, it means you're trying to upload a version that already exists on PyPI. You must increment the version number.

### Package not found after publishing

It may take a few minutes for the package to be indexed and available. Try again after a few minutes.

### Dependencies not installing

Make sure all dependencies are listed in both `setup.py` and `pyproject.toml` with compatible version constraints.

## GitHub Actions (Optional)

You can automate the publishing process using GitHub Actions. See `.github/workflows/publish.yml` for an example workflow.
