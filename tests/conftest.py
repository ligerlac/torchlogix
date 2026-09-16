import importlib.util
import os

# torch and scikit-learn (a required torchlogix dependency) each bundle their
# own separate copy of the OpenMP runtime (libomp.dylib on macOS). Loading
# both into one process trips OpenMP's duplicate-runtime safety check and
# aborts the interpreter. This is the standard, low-risk workaround for that
# specific benign double-load (not a general-purpose crash suppressant) -
# must be set before torch/sklearn are actually imported. setdefault() so an
# explicit value set by the caller isn't clobbered.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# The shared models live in tests/models.py and the tests parametrize over
# models.MODELS directly, so no model fixtures are defined here.


def pytest_terminal_summary(terminalreporter):
    """Say loudly when an optional integration suite did not run.

    A missing optional dependency otherwise shows up only inside the bare
    "N skipped" count, or with -rs, so an entire integration suite can quietly
    not run and nobody notices - including in CI.
    """
    skipped = terminalreporter.stats.get("skipped", [])
    if not any("test_alkaid_plugin" in report.nodeid for report in skipped):
        return
    if importlib.util.find_spec("alkaid") is not None:
        return      # skipped for some other reason; don't blame the install

    terminalreporter.write_sep("=", "alkaid integration tests DID NOT RUN",
                               yellow=True, bold=True)
    terminalreporter.write_line(
        "The optional 'alkaid' package is not installed, so every test in "
        "tests/test_alkaid_plugin.py was skipped."
    )
    terminalreporter.write_line("")
    terminalreporter.write_line("    pip install 'torchlogix[alkaid]'")
    terminalreporter.write_line("")
    terminalreporter.write_sep("=", yellow=True, bold=True)
