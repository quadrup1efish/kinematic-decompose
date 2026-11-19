import numpy as np
import pytest

from kinematic_decompose.analysis import stellar_assembly as assembly
from kinematic_decompose.PyTNG import stellar_assembly as legacy_assembly


def test_legacy_imports():
    assert legacy_assembly.compute_stellar_assembly is assembly.compute_stellar_assembly
    assert legacy_assembly.cal_in_ex_situ is assembly.cal_in_ex_situ


@pytest.mark.parametrize('use_header', [True, False])
def test_legacy_buckets(monkeypatch, use_header):
    tree = {'SnapNum': [99, 50, 40, 20], 'SubfindID': [7, 8, 9, 10]}
    rows = {
        (20, 10): ([1, 8], [.10, .12]),
        (40, 9): None,
        (50, 8): ([4, 2, 5, 3], [.35, .15, .20 + 1e-10, -.1]),
        (99, 7): ([1, 2, 3, 4, 5, 6], [.10, .15, .45, .35, .20, -.2]),
    }
    calls = []

    def load(base, snap, sid, part, fields):
        calls.append((snap, sid))
        row = rows[snap, sid]
        if row is None:
            return {'count': 0}
        ids, formation = row
        return {'ParticleIDs': np.array(ids, dtype=np.uint64),
                'GFM_StellarFormationTime': np.array(formation)}

    monkeypatch.setattr(assembly.il.snapshot, 'loadSubhalo', load)
    monkeypatch.setattr(assembly.il.sublink, 'loadTree', lambda *a, **kw: tree)
    scales = {20: .20, 50: .50}
    header_calls = []

    def load_header(base, snap):
        assert base == 'mock'
        header_calls.append(snap)
        return {'Time': scales[snap]}

    monkeypatch.setattr(assembly.il.groupcat, 'loadHeader', load_header)
    if use_header:
        inside, outside = assembly.compute_stellar_assembly('mock', 7)
    else:
        inside, outside = assembly.compute_stellar_assembly(
            'mock', 7, snapnum_to_scale_factor=scales.__getitem__)
    assert header_calls == ([20, 50] if use_header else [])
    expected_in = [[] for _ in range(100)]
    expected_ex = [[] for _ in range(100)]
    expected_in[20], expected_in[50], expected_in[99] = [1], [4, 3], [6]
    expected_ex[50] = [2, 5]
    assert (inside, outside) == (expected_in, expected_ex)
    assert calls == [(99, 7), (20, 10), (40, 9), (50, 8), (99, 7)]


def test_legacy_load_failure_propagates(monkeypatch):
    def load(base, snap, sid, *args, **kwargs):
        if snap == 20:
            raise OSError('snapshot unavailable')
        return {'ParticleIDs': np.array([1])}

    monkeypatch.setattr(assembly.il.snapshot, 'loadSubhalo', load)
    with pytest.raises(OSError, match='snapshot unavailable'):
        assembly.cal_in_ex_situ(
            {'SnapNum': [99, 20], 'SubfindID': [7, 10]}, 7,
            base_path='mock', snapnum_to_scale_factor=lambda snap: .20)