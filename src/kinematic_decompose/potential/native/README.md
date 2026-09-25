# Native potential backend (Agama-derived)

This directory contains the in-tree, narrowly scoped native backend for the potential operations used by `kinematic-decompose`.

## Compatibility target

The backend is intended to preserve the Agama contract used by this project:

- `Multipole` construction from particles, including the native cubic-spline softened source;
- `potential(xyz)` and `force(xyz)` in Agama's Cartesian array conventions;
- additive `Composite` potentials;
- Agama-compatible Multipole INI export/load for supported Multipole and Composite objects.
- Agama-compatible `setUnits`, `getUnits`, and `G` behavior for supported potential operations.

This is not a replacement for the complete Agama Python package. It will not provide actions, distribution functions, orbit integration, or unrelated potential families. Compatibility is established by differential tests against the pinned Agama reference, not by matching names alone.

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
