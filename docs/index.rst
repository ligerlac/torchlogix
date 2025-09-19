TorchLogix Documentation
========================

**TorchLogix** is a PyTorch library for differentiable logic gate neural networks. It extends the original DiffLogic work with enhanced features, improved usability, and comprehensive documentation.

.. note::
   TorchLogix builds upon the foundational work of `DiffLogic <https://github.com/Felix-Petersen/difflogic>`_ by Felix Petersen et al.
   This library provides extensions and improvements while maintaining compatibility with the original concepts.

Key Features
------------

* **Differentiable Logic Gates**: Implement neural networks using 16 different binary logical operations
* **Convolutional Support**: 2D and 3D logic convolutional layers with flexible receptive fields
* **Model Compilation**: Convert trained logic networks to optimized implementations
* **CUDA Acceleration**: Optional CUDA extensions for high-performance computing
* **Easy Integration**: Drop-in replacement for standard PyTorch layers

Quick Start
-----------

.. code-block:: python

   import torch
   from torchlogix.layers import LogicDense, LogicConv2d
   from torchlogix.models import CNN

   # Create a simple logic dense layer
   layer = LogicDense(in_dim=784, out_dim=128, tree_depth=3)

   # Create a logic convolutional layer
   conv = LogicConv2d(
       in_dim=(28, 28),
       num_kernels=16,
       tree_depth=3,
       receptive_field_size=5
   )

   # Use pre-built models
   model = CNN(class_count=10, tau=1.0)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/installation
   guides/quickstart
   guides/logic_gates
   guides/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/torchlogix
   api/layers
   api/models
   api/functional

.. toctree::
   :maxdepth: 1
   :caption: Development

   guides/contributing

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`