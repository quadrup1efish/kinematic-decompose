"""Smoke test on REAL TNG50-1 galaxy 307486 (z=0, snap 99).

Purpose
-------
The MIN_MAHAL post-hoc filter was removed from ``_find_residual_component``
(commit 3fc3225): candidates previously rejected for being too close to an
existing component (Mahalanobis < 0.75) are now admitted directly. This
smoke test runs the full ``kinematic_decomposition_pipeline`` on a real
galaxy and locks two contracts:

  * the pipeline still runs end-to-end without error on real data;
  * the number of discovered components does NOT inflate (the removed filter
    was a magic-number safeguard; if its absence caused over-splitting, the
    component count would blow past the historical ~7 for this galaxy).

The assertion is deliberately loose (<= 12 < MAX_N_COMPONENTS=15): it guards
against catastrophic regression (runaway over-splitting), not against small
legitimate variations across refactors.
"""
import os
import numpy as np
import pytest

from kinematic_decompose.config import TEST_IMAGE_PATH, TEST_DATA_PATH
from kinematic_decompose.pipeline import kinematic_decomposition_pipeline

RUN = "TNG50-1"
SNAP = 99
SUBID = 307486
MAX_EXPECTED_COMPONENTS = 12   # historical fit for this galaxy is 7; 12 << 15 cap


@pytest.mark.slow
def test_real_galaxy_pipeline_component_count():
    """Real galaxy smoke test: pipeline runs and the component count does not
    inflate after the MIN_MAHAL removal (no runaway over-splitting)."""
    model, galaxy, eoemin_cut, jzojc_cut = kinematic_decomposition_pipeline(
        run=RUN, snapNum=SNAP, subID=SUBID,
        gravity_potential_path=TEST_DATA_PATH,
        image_path=TEST_IMAGE_PATH,
        structure_properties_output_path=None,
        mixture_model_output_path=None,
    )
    assert model is not None, "pipeline returned no mixture model"
    n_comp = model.n_components
    assert 1 <= n_comp <= MAX_EXPECTED_COMPONENTS, (
        f"component count {n_comp} exceeds {MAX_EXPECTED_COMPONENTS} "
        f"(MIN_MAHAL removal may have caused over-splitting)"
    )
    # the fit must be anchored on real kinematic data
    assert galaxy.s['eoemin'].size > 0
    assert np.isfinite(model.means_).all()
    # weights must be a valid simplex
    assert np.isclose(model.weights_.sum(), 1.0)
    assert (model.weights_ >= 0).all()
