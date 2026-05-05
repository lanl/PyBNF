import logging
from types import SimpleNamespace

import numpy as np

from pybnf import pybnf


def test_configured_random_seed_seeds_numpy_and_logs(caplog):
    cfg = SimpleNamespace(config={'random_seed': 123})

    with caplog.at_level(logging.INFO, logger='pybnf.pybnf'):
        seed = pybnf._initialize_random_seed(cfg)

    assert seed == 123
    assert cfg.config['random_seed'] == 123
    assert np.random.random() == np.random.RandomState(123).random_sample()
    assert 'Random seed: 123' in caplog.text


def test_missing_random_seed_generates_logs_and_uses_effective_seed(monkeypatch, caplog):
    cfg = SimpleNamespace(config={'random_seed': None})
    seed_calls = []

    def fake_seed(seed=None):
        seed_calls.append(seed)

    def fake_randint(low, high):
        assert low == 0
        assert high == 2**31
        return 456

    monkeypatch.setattr(pybnf.np.random, 'seed', fake_seed)
    monkeypatch.setattr(pybnf.np.random, 'randint', fake_randint)

    with caplog.at_level(logging.INFO, logger='pybnf.pybnf'):
        seed = pybnf._initialize_random_seed(cfg)

    assert seed == 456
    assert cfg.config['random_seed'] == 456
    assert seed_calls == [None, 456]
    assert 'Random seed: 456' in caplog.text
