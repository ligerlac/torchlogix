"""Setup configuration for the torchlogix package."""

import os

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# Check if we should build CUDA extensions
# Only build when explicitly requested via pip install .[cuda]
build_cuda_ext = False

# Check if 'cuda' is in the extras being installed
import sys
if any('cuda' in arg for arg in sys.argv if '[' in arg and ']' in arg):
    build_cuda_ext = True

if build_cuda_ext:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        ext_modules = [
            CUDAExtension(
                "torchlogix.torchlogix_cuda",
                [
                    "src/torchlogix/cuda/torchlogix.cpp",
                    "src/torchlogix/cuda/torchlogix_kernel.cu",
                ],
                extra_compile_args={"nvcc": ["-lineinfo"]},
            )
        ]
    except ImportError:
        print("Warning: CUDA extension requested but torch.utils.cpp_extension not available")
        ext_modules = []
else:
    ext_modules = []


setup(
    name="torchlogix",
    version="0.1.0",
    author="Lino Gerlach",
    author_email="lino.oscar.gerlach@cern.ch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ligerlac/torchlogix",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
    if ext_modules
    else {},  # Only if building extensions
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.6.0",
        "numpy>=1.26",
        "tqdm",
        "scikit-learn",
        "torchvision",
        "rich",
        "torch-geometric",
        "wheel",
    ],
    extras_require={
        "dev": [
            "flake8>=6.1.0",
            "black>=23.12.1", 
            "isort>=5.13.2",
            "pre-commit>=3.6.0",
            "pytest>=8.0.0",
            "autopep8>=2.0.4",
        ],
        "cuda": [
            # No additional dependencies needed, just triggers CUDA extension build
        ],
    },
)
