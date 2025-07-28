#!/usr/bin/env python3
"""setup.py – build / install script for PyBNF"""

from pathlib import Path
import re
from setuptools import setup, find_packages

# ----------------------------------------------------------------------
#  Metadata
# ----------------------------------------------------------------------
version = re.search(
    r'^__version__\s*=\s*"(.*)"',
    Path("pybnf/pybnf.py").read_text(),
    re.M,
).group(1)

long_desc = Path("README.md").read_text(encoding="utf-8")

# ----------------------------------------------------------------------
#  Setup
# ----------------------------------------------------------------------
setup(
    name="pybnf",
    version=version,
    description="Tool for parameterization of BNGL and SBML models",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author="Eshan Mitra, Ryan Suderman, Alex Ionkov, Bill Hlavacek",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    entry_points={"console_scripts": ["pybnf = pybnf.pybnf:main"]},

    # ------------------------------------------------------------------
    #  Runtime dependencies – what users really need to *run* PyBNF
    # ------------------------------------------------------------------
    install_requires=[
        "numpy>=1.22",
        "scipy",
        "pyparsing",
        "dask[distributed]>=2021.5.0",
        "tornado>=6.1",
        "paramiko",
        "msgpack==0.6.2",
        "libroadrunner>=1.5.2",
    ],

    # ------------------------------------------------------------------
    #  Extras – optional deps that *developers / CI* can pull in
    # ------------------------------------------------------------------
    extras_require={
        "dev": [
            "pytest",
            "nose",         # <-- keep here only while a few tests still use nose.tools
            "black",
            "ruff",
        ],
    },
)
