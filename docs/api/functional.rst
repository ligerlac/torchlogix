Functional Module
=================

.. currentmodule:: torchlogix.functional

The functional module contains the core mathematical operations and utility functions for logic gate computations.

Logic Operations
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bin_op
   bin_op_s
   bin_op_cnn
   bin_op_cnn_slow
   compute_all_logic_ops_vectorized

Utility Functions
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_unique_connections
   GradFactor

Constants
---------

.. autodata:: ID_TO_OP

   Dictionary mapping logic gate IDs (0-15) to their corresponding operations.

   Each logic gate represents one of the 16 possible binary Boolean operations:

   * 0: False (always 0)
   * 1: AND (a ∧ b)
   * 2: A and not B (a ∧ ¬b)
   * 3: A (identity for a)
   * 4: not A and B (¬a ∧ b)
   * 5: B (identity for b)
   * 6: XOR (a ⊕ b)
   * 7: OR (a ∨ b)
   * 8: NOR (¬(a ∨ b))
   * 9: XNOR (¬(a ⊕ b))
   * 10: NOT B (¬b)
   * 11: B implies A (b → a)
   * 12: NOT A (¬a)
   * 13: A implies B (a → b)
   * 14: NAND (¬(a ∧ b))
   * 15: True (always 1)

Function Details
----------------

.. autofunction:: bin_op

.. autofunction:: bin_op_s

.. autofunction:: bin_op_cnn

.. autofunction:: bin_op_cnn_slow

.. autofunction:: compute_all_logic_ops_vectorized

.. autofunction:: get_unique_connections

.. autoclass:: GradFactor
   :members:
   :undoc-members:
   :show-inheritance: