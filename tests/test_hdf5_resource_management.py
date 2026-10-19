"""Regression tests for deterministic HDF5 handle closure in legacy readers."""

import h5py
import numpy as np
import pytest

from kinematic_decompose.PyTNG.illustris_python import groupcat, lhalotree, snapshot


def _track_open_files(monkeypatch, module):
    """Keep references to opened files so tests can inspect explicit closure."""
    opened = []
    real_file = h5py.File

    def tracked_file(*args, **kwargs):
        handle = real_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(module.h5py, "File", tracked_file)
    return opened


def _write_groupcat_chunk(path, *, total, this_file, mass=None):
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["Nsubgroups_Total"] = total
        header.attrs["Nsubgroups_ThisFile"] = this_file
        header.attrs["NumFiles"] = 2
        subhalo = handle.create_group("Subhalo")
        if mass is not None:
            subhalo.create_dataset("SubhaloMass", data=np.asarray(mass, dtype=float))


def _write_groupcat_pair(directory, *, second_mass):
    directory.mkdir(parents=True)
    _write_groupcat_chunk(
        directory / "groups_099.0.hdf5", total=1, this_file=0, mass=[]
    )
    _write_groupcat_chunk(
        directory / "groups_099.1.hdf5", total=1,
        this_file=1 if second_mass is not None else 0,
        mass=second_mass,
    )


def test_load_subset_closes_files_when_particle_type_is_in_later_chunk(
    tmp_path, monkeypatch
):
    snapshot_dir = tmp_path / "snapdir_099"
    snapshot_dir.mkdir()
    total = np.array([0, 2, 0, 0, 0, 0], dtype=np.uint32)
    with h5py.File(snapshot_dir / "snap_099.0.hdf5", "w") as handle:
        header = handle.create_group("Header")
        header.attrs["NumPart_Total"] = total
        header.attrs["NumPart_ThisFile"] = np.zeros(6, dtype=np.uint32)
    with h5py.File(snapshot_dir / "snap_099.1.hdf5", "w") as handle:
        header = handle.create_group("Header")
        header.attrs["NumPart_Total"] = total
        header.attrs["NumPart_ThisFile"] = np.array([0, 2, 0, 0, 0, 0], dtype=np.uint32)
        handle.create_group("PartType1").create_dataset(
            "Coordinates", data=np.array([[1., 2., 3.], [4., 5., 6.]])
        )

    opened = _track_open_files(monkeypatch, snapshot)
    result = snapshot.loadSubset(str(tmp_path), 99, "dm", ["Coordinates"])

    np.testing.assert_array_equal(result, [[1., 2., 3.], [4., 5., 6.]])
    assert opened and all(not handle.id.valid for handle in opened)


def test_load_objects_closes_empty_and_nonempty_chunk_files(tmp_path, monkeypatch):
    _write_groupcat_pair(tmp_path / "groups_099", second_mass=[3.0])
    opened = _track_open_files(monkeypatch, groupcat)

    result = groupcat.loadObjects(
        str(tmp_path), 99, "Subhalo", "subgroups", ["SubhaloMass"]
    )

    np.testing.assert_array_equal(result, [3.0])
    assert opened and all(not handle.id.valid for handle in opened)


def test_load_objects_closes_chunk_file_when_field_read_raises(tmp_path, monkeypatch):
    group_dir = tmp_path / "groups_099"
    group_dir.mkdir()
    _write_groupcat_chunk(group_dir / "groups_099.0.hdf5", total=2, this_file=1, mass=[3.0])
    _write_groupcat_chunk(group_dir / "groups_099.1.hdf5", total=2, this_file=1, mass=None)
    opened = _track_open_files(monkeypatch, groupcat)

    with pytest.raises(Exception, match="does not have requested field"):
        groupcat.loadObjects(
            str(tmp_path), 99, "Subhalo", "subgroups", ["SubhaloMass"]
        )

    assert opened and all(not handle.id.valid for handle in opened)


def _write_tree(path):
    with h5py.File(path, "w") as handle:
        tree = handle.create_group("Tree0")
        tree.create_dataset("FirstProgenitor", data=np.array([-1], dtype=np.int64))
        tree.create_dataset("NextProgenitor", data=np.array([-1], dtype=np.int64))
        tree.create_dataset("Mass", data=np.array([2.5]))


def test_load_tree_closes_file_on_success_and_field_validation_error(
    tmp_path, monkeypatch
):
    tree_path = tmp_path / "tree.hdf5"
    _write_tree(tree_path)
    monkeypatch.setattr(lhalotree, "treeOffsets", lambda *args: (0, 0, 0))
    monkeypatch.setattr(lhalotree, "treePath", lambda *args: str(tree_path))

    def copy_single_node(conn, index, data_in, data_out, count, only_mpb):
        if not isinstance(data_in, dict):
            data_out[0] = data_in[0]
        return 1

    monkeypatch.setattr(lhalotree, "singleNodeFlat", copy_single_node)
    opened = _track_open_files(monkeypatch, lhalotree)

    result = lhalotree.loadTree("unused", 99, 0, fields=["Mass"])
    np.testing.assert_array_equal(result, [2.5])
    assert opened and all(not handle.id.valid for handle in opened)

    opened.clear()
    with pytest.raises(Exception, match="Requested field Missing"):
        lhalotree.loadTree("unused", 99, 0, fields=["Missing"])
    assert opened and all(not handle.id.valid for handle in opened)
