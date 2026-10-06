#!/usr/bin/env python3
"""Render the softened/unsoftened potential comparison for an A&A-sized figure.

Run from the repository root:
    python tests/example_potential_profile_comparison.py

The compact input profile is stored under tests/fixtures; the PNG is written to
images/potential_profile_comparison.png at 180 mm width and 300 dpi.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(__file__).resolve().parent / "fixtures" / "potential_profile_symmetry_n.npz"
DEFAULT_OUTPUT = ROOT / "images" / "potential_profile_comparison.png"
FIGURE_WIDTH_IN = 180.0 / 25.4  # A&A maximum full-page figure width.
DPI = 300


def set_publication_style() -> None:
    """Set compact print-size type and crisp axes for the final 180-mm width."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12.0,
        "axes.labelsize": 14.0,
        "axes.titlesize": 14.0,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "legend.fontsize": 12.0,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.grid": False,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
    })


def create_figure(input_path: Path, output_path: Path):
    with np.load(input_path) as data:
        radii = np.asarray(data["radii_kpc"], dtype=np.float64)
        phi_direct = np.asarray(data["phi_direct_cubic_spline_reference_kms2"], dtype=np.float64)
        phi_soft = np.asarray(data["phi_multipole_softened_lmax6_grid40_kms2"], dtype=np.float64)
        phi_unsoft = np.asarray(data["phi_multipole_unsoftened_lmax6_grid40_kms2"], dtype=np.float64)

    assert len(radii) == 321 and np.isfinite(radii).all()
    assert np.all(np.diff(radii) > 0) and np.isclose(radii[0], 0.01) and np.isclose(radii[-1], 5.0)
    for profile in (phi_direct, phi_soft, phi_unsoft):
        assert profile.shape == radii.shape and np.isfinite(profile).all() and np.all(profile < 0)

    # Normalize each model by its own minimum over the sampled radial interval.
    direct_norm = phi_direct / abs(float(np.min(phi_direct)))
    soft_norm = phi_soft / abs(float(np.min(phi_soft)))
    unsoft_norm = phi_unsoft / abs(float(np.min(phi_unsoft)))
    soft_relative = (soft_norm - direct_norm) / np.abs(direct_norm) * 100.0
    unsoft_relative = (unsoft_norm - direct_norm) / np.abs(direct_norm) * 100.0

    set_publication_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN))
    grid = fig.add_gridspec(2, 1, height_ratios=(4.0, 1.0), hspace=0.0)
    ax_profile = fig.add_subplot(grid[0, 0])
    ax_error = fig.add_subplot(grid[1, 0], sharex=ax_profile)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.14, hspace=0.0)

    ax_profile.semilogx(radii, unsoft_norm, color="#222222", lw=2.0,
                        label="unsoftened multipole")
    ax_profile.semilogx(radii, soft_norm, color="navy", lw=2.0, ls="-.",
                        label="softened multipole")
    ax_profile.semilogx(radii, direct_norm, color="red", lw=2.0, ls="--",
                        dashes=(8, 5), label="direct spline")
    ax_profile.axhline(-1.0, color="#888888", lw=0.65, ls=":")
    ax_profile.set_ylabel(r"$\Phi(R,0)/|\Phi_{\min}|$")
    ax_profile.set_xlim(0.01, 5.0)
    ax_profile.legend(loc="upper left", handlelength=2.4, borderaxespad=0.45)
    ax_profile.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax_error.semilogx(radii, soft_relative, color="navy", lw=1.6, ls="-",
                      marker="o", markevery=32, ms=4.0, label="softened")
    ax_error.semilogx(radii, unsoft_relative, color="#222222", lw=1.6, ls="-",
                      marker="s", markevery=32, ms=4.0, label="unsoftened")
    ax_error.axhline(0.0, color="#888888", lw=0.65)
    ax_error.set_xlabel(r"$R\ ({\rm kpc})$")
    ax_error.set_ylabel("Relative error (%)")
    ax_error.set_xlim(0.01, 5.0)
    ax_error.set_ylim(-2.0, 13.5)
    ax_error.legend(loc="upper left", handlelength=2.4, borderaxespad=0.45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI)
    # Keep the requested PNG in images/ and also emit vector PDF for submission.
    fig.savefig(output_path.with_suffix(".pdf"))
    return fig, {
        "n_radii": int(len(radii)),
        "radial_range_kpc": [float(radii[0]), float(radii[-1])],
        "softened_p95_abs_relative_percent": float(np.percentile(np.abs(soft_relative), 95)),
        "softened_max_abs_relative_percent": float(np.max(np.abs(soft_relative))),
        "unsoftened_p95_abs_relative_percent": float(np.percentile(np.abs(unsoft_relative), 95)),
        "symmetry": "n for all components; lmax=mmax=6",
        "figure_width_mm": 180.0,
        "dpi": DPI,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png; a matching vector PDF is also written")
    fig, summary = create_figure(args.input, args.output)
    plt.close(fig)
    print(f"Saved {args.output}")
    print(summary)


if __name__ == "__main__":
    main()
