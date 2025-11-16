"""Setup configuration for the torchlogix package."""

import os

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="torchlogix",
    version="0.1.0",
    author="Lino Gerlach",
    author_email="lino.oscar.gerlach@cern.ch",
    description="Differentiable Logic Gate Networks in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ligerlac/torchlogix",
    project_urls={
        "Documentation": "https://ligerlac.github.io/torchlogix/",
        "Source": "https://github.com/ligerlac/torchlogix",
        "Bug Tracker": "https://github.com/ligerlac/torchlogix/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.6.0",
        "numpy>=1.19.0",
        "tqdm>=4.50.0",
        "scikit-learn>=0.24.0",
        "torchvision>=0.8.0",
        "rich>=10.0.0",
    ],
    extras_require={
        "dev": [
            "flake8>=6.1.0",
            "black>=23.12.1",
            "isort>=5.13.2",
            "pre-commit>=3.6.0",
            "pytest>=8.0.0",
            "autopep8>=2.0.4",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "cuda": [
            # No additional dependencies needed, just triggers CUDA extension build
        ],
    },
    keywords="deep-learning pytorch logic-gates neural-networks machine-learning",
    include_package_data=True,
)
