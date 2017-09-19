"""pybnf.pybnf: defines the entry point for the PyBNF application"""


import logging


__version__ = "0.1"


def main():
    log_format = "%(levelname)s\t%(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)

    logging.info("PyBNF v%s" % __version__)
