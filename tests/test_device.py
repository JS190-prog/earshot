"""EARSHOT_DEVICE: which torch device CLAP and AST load on (no model is loaded here)"""
import pytest

torch = pytest.importorskip('torch')

from earshot.models import clap


def test_cpu_is_cpu():
    assert clap.pick_device(torch, 'cpu').type == 'cpu'


def test_bad_value_is_an_error():
    for bad in ('gpu', 'cuda:x', 'mps'):
        with pytest.raises(ValueError):
            clap.pick_device(torch, bad)


def test_auto_follows_cuda(monkeypatch):
    monkeypatch.setattr(clap, '_cuda_runs', lambda t: (False, 'no gpu'))
    assert clap.pick_device(torch, 'auto').type == 'cpu'
    monkeypatch.setattr(clap, '_cuda_runs', lambda t: (True, ''))
    assert clap.pick_device(torch, 'auto').type == 'cuda'


def test_explicit_cuda_never_falls_back(monkeypatch):
    monkeypatch.setattr(clap, '_cuda_runs', lambda t: (False, 'no gpu'))
    with pytest.raises(RuntimeError, match='EARSHOT_DEVICE=cuda'):
        clap.pick_device(torch, 'cuda')
    monkeypatch.setattr(clap, '_cuda_runs', lambda t: (True, ''))
    assert str(clap.pick_device(torch, 'cuda:1')) == 'cuda:1'
