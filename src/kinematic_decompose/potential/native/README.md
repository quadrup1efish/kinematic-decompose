# Native potential backend (Agama-derived)

This directory contains the in-tree, narrowly scoped native backend for the potential operations used by `kinematic-decompose`.

## Compatibility target

The backend is intended to preserve the Agama contract used by this project:

- `Multipole` construction from particles, including the native cubic-spline softened source;
- `potential(xyz)`, `force(xyz)` and `density(xyz)` in Cartesian coordinates;
- density queries reuse the existing Agama numerical core, return mass per volume in the active units, and accept a single `(3,)` point or an `(N,3)` array;
- additive `Composite` potentials;
- Agama-compatible Multipole INI export/load for supported Multipole and Composite objects.
- Agama-compatible `setUnits`, `getUnits`, and `G` behavior for supported potential operations.

This is not a replacement for the complete Agama Python package. It will not provide actions, distribution functions, orbit integration, or unrelated potential families. Compatibility is established by differential tests against the pinned Agama reference, not by matching names alone.

Density queries do not clip negative densities or replace NaN/infinite values. Exact-origin extrapolation may return non-finite values. This backend fixes the upstream non-axisymmetric Multipole axis-force conversion and near-axis density cancellation. Cartesian axis derivatives are obtained from radial directional derivatives; the axis Laplacian projects onto the m=0 azimuthal harmonic. Within R/|z| <= epsilon^(1/3), density uses this axis limit to avoid cancellation roundoff. The polar endpoint second derivatives of the angular splines enforce C2 harmonic regularity; this changes non-axisymmetric interpolation in the cells touching the poles, while axisymmetric splines and other cells remain unchanged. Regression tests cover positive/negative axes, triaxial/non-symmetric expansions, softened/unsoftened sources and exterior forces without expected-failure markers. Cross-backend tests can use `AGAMA_REFERENCE_PYTHON` and `AGAMA_REFERENCE_PATH` to select a matching Agama interpreter/source build without loading both extensions in one process; an unpatched reference is not an oracle at the affected axes.

## Source provenance

The vendored files in `agama/src` are based on Agama upstream revision `f302756`, with the local native softened-Multipole implementation included. Preserve source notices and the accompanying `LICENSE`. The upstream numerical core calls GSL; the build fetches a pinned GSL source release and links it statically, so no system GSL or `pkg-config` is required.

## Reproducible build dependencies

The project pins the Python build backend dependencies in `pyproject.toml` (`setuptools==82.0.1`, `wheel==0.41.2`, `pybind11==2.13.6`). Runtime Python dependencies are resolved by the committed `uv.lock`; use `uv sync --locked` so the lockfile is enforced.

GSL is built automatically from the official GSL 2.8 source archive and statically linked into the extension. The setup verifies the archive against the pinned SHA-256 before extracting or compiling it. Users do not need GSL headers/libraries installed, and the resulting extension does not need a GSL runtime library. The source archive and compiled static libraries are cached under `${XDG_CACHE_HOME:-~/.cache}/kinematic-decompose/gsl/2.8`, so later builds reuse them. Both the source distribution and wheel include the exact GSL source archive, its `COPYING` license, and the native C++ sources. The source distribution also carries `setup.py` and the build metadata, and can rebuild the wheel offline with respect to GSL. When redistributing a wheel, publish its matching source distribution alongside it (or provide an equally accessible corresponding-source bundle).

```sh
uv sync --locked
uv build
```

Building directly from a source checkout needs internet access on the first build, or a local archive supplied through `KINEMATIC_DECOMPOSE_GSL_ARCHIVE`; a C/C++ compiler and `make` are also required. Builds from the published source distribution already contain the verified source archive and do not need network access. To update GSL, change the version, archive URL, checksum, and distribution patterns together, then rebuild and rerun native compatibility tests. Automatic source builds currently support macOS and Linux.

The C++ extension still requires a C++11-capable compiler; `make` is used only to build the pinned static GSL dependency. `pkg-config`, Conda, Homebrew GSL, and system GSL libraries are not required. This Agama-derived backend uses GSL and is distributed under GPL-3.0-or-later; retain the accompanying Agama-derived source notice.

## Porting stages

1. Build the in-tree Python extension from the selected Agama source closure.
2. Expose the agreed Python API and unit settings.
3. Differential-test numerical evaluation and Agama INI round trips; keep upstream Agama as the optional compatibility-test reference.
4. Pipeline construction, energy evaluation, and serialized-potential loading now use the in-tree backend.
