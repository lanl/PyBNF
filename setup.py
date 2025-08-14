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
        "numpy>=2.0",
        "scipy>=1.15",
        "pyparsing>=3.0",
        "dask[distributed]>=2022.5",
        "tornado>=6.1",
        "paramiko>=3.0",
        "libroadrunner>=2.8",
    ],

    # ------------------------------------------------------------------
    #  Extras – optional deps that *developers / CI* can pull in
    # ------------------------------------------------------------------
    extras_require={
        "dev": [
            "pytest",
            "black",
            "ruff",
        ],
    },
)
