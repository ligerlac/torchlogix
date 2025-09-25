# Contributing to TorchLogix

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/ligerlac/torchlogix.git
cd torchlogix
```

2. Install in development mode:
```bash
pip install -e .[dev]
```

3. Install documentation dependencies:
```bash
pip install -r docs/requirements.txt
```

## Building Documentation

From the `docs/` directory:

```bash
# Build HTML documentation
make html

# Open in browser (macOS)
open _build/html/index.html

# Clean build
make clean
make html
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=torchlogix

# Run specific test file
pytest tests/test_lgn.py
```

## Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Format your code:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## Submitting Changes

1. Create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Format code with black/isort
6. Submit a pull request