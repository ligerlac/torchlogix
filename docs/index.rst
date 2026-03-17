TorchLogix Documentation
========================

**TorchLogix** is a PyTorch-based library for training and inference of **logic neural networks**.
These solve machine learning tasks by learning combinations of boolean logic expressions.
As the choice of boolean expressions is conventionally non-differentiable, relaxations are applied to allow training with gradient-based methods.
The final model can be discretized again, resulting in a fully boolean expression with extremely efficient inference, e.g., beyond a
million images of MNIST per second on a single CPU core.

.. note::
   TorchLogix is based on the `difflogic` package ([https://github.com/Felix-Petersen/difflogic/](https://github.com/Felix-Petersen/difflogic/)),
   and extends it by new concepts such as compact parametrizations, higher-dimensional logic blocks, learnable connections and binarization as
   described in "WARP Logic Neural Networks" (Paper @ [ArXiv](https://arxiv.org/abs/2602.03527)).
   It also implements convolutions as described in "Convolutional Differentiable Logic Gate Networks (Paper @ [ArXiv](https://arxiv.org/pdf/2411.04732)).

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/installation
   guides/quickstart
   guides/concepts
   guides/hardware_deployment

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/torchlogix
   api/layers
   api/models
   api/functional

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`