import types
import pytest
from pybnf.cluster import Cluster
from pybnf.printing import PybnfError
import pybnf.cluster as clmod

class _Cfg:
    def __init__(self, **kw):
        self.config = kw

def test_read_node_names_unknown_type():
    cfg = _Cfg(cluster_type="weird")
    with pytest.raises(PybnfError):
        Cluster.read_node_names(cfg)

def test_read_node_names_torque_unimplemented():
    cfg = _Cfg(cluster_type="torque")
    with pytest.raises(PybnfError):
        Cluster.read_node_names(cfg)

def test_read_node_names_slurm_parses_nodes(monkeypatch):
    cfg = _Cfg(cluster_type="slurm")
    class _Res:
        stdout = b"nodeA\nnodeB"
    monkeypatch.setattr(clmod, "run", lambda *a, **k: _Res())
    sched, nodes = Cluster.read_node_names(cfg)
    assert sched == "nodeA" and nodes == "nodeA nodeB"

def test_setup_cluster_builds_expected_command(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(clmod, "cpu_count", lambda: 8)
    def fake_popen(cmd, **kwargs):
        seen["cmd"] = cmd
        class _P:  # minimal stub
            def terminate(self): pass
        return _P()
    monkeypatch.setattr(clmod, "Popen", fake_popen)
    monkeypatch.setattr(clmod, "time", types.SimpleNamespace(sleep=lambda *_: None))

    # Default path: uses total cpu_count() for --nprocs
    Cluster.setup_cluster("n1 n2", str(tmp_path))
    assert "dask-ssh n1 n2" in seen["cmd"]
    assert "--nthreads 1" in seen["cmd"] and "--nprocs 8" in seen["cmd"]

    # Manual parallel_count splits across nodes: ceil(4/2) = 2
    Cluster.setup_cluster("n1 n2", str(tmp_path), parallel_count=4)
    assert "--nprocs 2" in seen["cmd"]
