import logging
from types import SimpleNamespace

import numpy as np

from pybnf import pybnf


def test_configured_random_seed_resolves_and_logs(caplog):
    """A configured seed is returned, written back to the config, and logged.

    Since the default_rng migration, _initialize_random_seed no longer seeds
    NumPy's legacy global RNG -- each algorithm builds its own
    np.random.Generator(default_rng) from config['random_seed'] -- so this only
    checks the resolve-and-log contract.
    """
    cfg = SimpleNamespace(config={'random_seed': 123})

    with caplog.at_level(logging.INFO, logger='pybnf.pybnf'):
        seed = pybnf._initialize_random_seed(cfg)

    assert seed == 123
    assert cfg.config['random_seed'] == 123
    assert 'Random seed: 123' in caplog.text


def test_missing_random_seed_drawn_from_entropy(caplog):
    """An unset seed is drawn from system entropy, recorded, and logged.

    The drawn seed must fall in the config's allowed [0, 2**32) range so it can be
    fed straight back to default_rng on a reproducing run.
    """
    cfg = SimpleNamespace(config={'random_seed': None})

    with caplog.at_level(logging.INFO, logger='pybnf.pybnf'):
        seed = pybnf._initialize_random_seed(cfg)

    assert isinstance(seed, int)
    assert 0 <= seed < 2**32
    assert cfg.config['random_seed'] == seed
    assert f'Random seed: {seed}' in caplog.text


def test_resolved_seed_drives_default_rng_deterministically():
    """The resolved seed is what reproduces a run: default_rng(seed) gives the same
    stream every time, which is the guarantee the algorithms rely on."""
    cfg = SimpleNamespace(config={'random_seed': 2024})
    seed = pybnf._initialize_random_seed(cfg)

    a = np.random.default_rng(seed).random(5)
    b = np.random.default_rng(seed).random(5)
    assert np.array_equal(a, b)
