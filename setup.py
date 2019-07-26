#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "argh",
    "attr",
    "related",

    "concise",
    "git+https://github.com/kundajelab/DeepExplain.git",

    # ml
    "gin-config",
    "keras>=2.2.4",
    "scikit-learn",
    "tensorflow",

    # numerics
    "h5py",
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",

    # Plotting
    "matplotlib>=3.0.2",
    "plotnine",
    "seaborn",

    # genomics
    "pybigwig",
    "pybedtools",  # remove?
    "modisco==0.5.1.1",
    # "pyranges",

    "joblib",
    "cloudpickle",  # - remove?
    "kipoi",
    "kipoiseq",

    "papermill",
    "nbconvert",
    "vdom>=0.6",

    # utils
    "ipython",
    "tqdm",
    "pprint",

    # Remove
    "gin-train",
    "genomelake",
    "pysam",  # replace with pyfaidx
]

optional = [
    "comet_ml",
    "wandb",
    "fastparquet",
    "python-snappy",
    "ipywidgets",  # for motif simulation
]


optional2 = [
    "pygam",
    "pytorch",
]

test_requirements = [
    "pytest",
    "virtualenv",
]

setup(
    name="bpnet",
    version="0.0.1",
    description=("BPNet: toolkit to learn motif synthax from high-resolution functional genomics data"
                 " using convolutional neural networks"),
    author="Ziga Avsec",
    author_email="avsec@in.tum.de",
    url="https://github.com/kundajelab/bpnet",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": test_requirements,
    },
    license="MIT license",
    zip_safe=False,
    keywords=["deep learning",
              "computational biology",
              "bioinformatics",
              "genomics"],
    test_suite="tests",
    include_package_data=False,
    tests_require=test_requirements
)
