from setuptools import setup

setup(name='pybnf',
      version='0.1',
      description='Fits BioNetGen models using metaheuristics',
      author='Alex Ionkov, Eshan Mitra, Ryan Suderman',
      package=['pybnf'],
      requires=['dask', 'distributed', 'numpy'])