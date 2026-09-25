from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _potential

G = _potential.gravitational_constant()


def setUnits(*, mass=0.0, length=0.0, velocity=0.0, time=0.0):
    """Set external units with Agama's ``setUnits`` argument semantics."""
    global G
    G = _potential.setUnits(mass, length, velocity, time)


def getUnits():
    """Return the active external unit scales, like Agama's ``getUnits``."""
    return _potential.getUnits()


class Potential:
    """Agama-compatible subset: Multipole, additive Composite and INI I/O."""

    def __init__(self, *components, **kwargs):
        if len(components) == 1 and isinstance(components[0], (str, Path)) and not kwargs:
            self._native = _potential.load(str(components[0]))
        elif components and not kwargs and all(isinstance(item, Potential) for item in components):
            self._native = _potential.composite([item._native for item in components])
        else:
            kind = kwargs.pop("type", "Multipole")
            if kind.lower() != "multipole":
                raise ValueError("The in-tree backend currently supports type='Multipole' only")
            particles = kwargs.pop("particles", None)
            if particles is None or len(particles) != 2:
                raise ValueError("particles=(positions, masses) is required")
            positions, masses = particles
            softening = kwargs.pop("softening", None)
            if softening is not None and np.ndim(softening) == 0:
                softening = float(np.asarray(softening))
            symmetry = kwargs.pop("symmetry", "a")
            lmax = int(kwargs.pop("lmax", 4))
            mmax = int(kwargs.pop("mmax", lmax))
            grid_size = int(kwargs.pop("gridSizeR", kwargs.pop("gridsizeR", 40)))
            rmin = float(kwargs.pop("rmin", 0.0))
            rmax = float(kwargs.pop("rmax", 0.0))
            if kwargs:
                raise TypeError(f"Unsupported Potential arguments: {', '.join(sorted(kwargs))}")
            self._native = _potential.multipole(
                np.asarray(positions, dtype=float), np.asarray(masses, dtype=float),
                softening, symmetry, lmax, mmax, grid_size, rmin, rmax,
            )

    def potential(self, xyz):
        return self._native.potential(np.asarray(xyz, dtype=float))

    def force(self, xyz):
        return self._native.force(np.asarray(xyz, dtype=float))

    def export(self, filename):
        self._native.export(str(filename))


__all__ = ["Potential", "setUnits", "getUnits", "G"]
