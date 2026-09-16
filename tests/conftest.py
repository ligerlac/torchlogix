import os

# torch and scikit-learn (a required torchlogix dependency) each bundle their
# own separate copy of the OpenMP runtime (libomp.dylib on macOS). Loading
# both into one process trips OpenMP's duplicate-runtime safety check and
# aborts the interpreter. This is the standard, low-risk workaround for that
# specific benign double-load (not a general-purpose crash suppressant) -
# must be set before torch/sklearn are actually imported. setdefault() so an
# explicit value set by the caller isn't clobbered.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# The shared models live in tests/models.py as plain factory functions, and
# the tests parametrize over models.MODELS directly. No model fixtures are
# defined here: every test needs a fresh instance it can mutate (set weights,
# switch to export mode), which a function call expresses more simply than a
# fixture.
