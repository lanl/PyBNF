"""setup.py: setuptools control"""


import re
from setuptools import setup

version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('pybnf/pybnf.py').read(),
    re.M
    ).group(1)

with open('README.md', 'rb') as f:
    long_desc = f.read().decode('utf-8')

setup(name='pybnf',
      packages=['pybnf'],
      entry_points={'console_scripts': ['pybnf = pybnf.pybnf:main']},
      version=version,
      description='An application for parallel fitting of BioNetGen models using metaheuristics',
      long_description=long_desc,
      author='Alex Ionkov, Eshan Mitra, Ryan Suderman',
      package=['pybnf'],
      install_requires=['dask', 'distributed', 'numpy', 'nose', 'pyparsing'])
