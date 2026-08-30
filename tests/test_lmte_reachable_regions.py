import math

import numpy as np

from crowd_sim.social.lmte import LightweightMotionTrendEstimator


def test_lmte_tracks_motion_and_returns_finite_ellipses():
    estimator = LightweightMotionTrendEstimator(
        K=3,
        tau_list=[0.5, 1.0],
        default_dt=0.1,
    )
    estimator.update(
        "human-1",
        position=[0.0, 0.0],
        velocity=[1.0, 0.25],
        timestamp=0.0,
    )
    estimate = estimator.update(
        "human-1",
        position=[0.1, 0.025],
        velocity=[1.0, 0.25],
        timestamp=0.1,
    )

    np.testing.assert_allclose(estimate["v_hat"], [1.0, 0.25])
    assert estimate["valid"] is True

    regions = estimator.reachable_regions("human-1")
    assert list(regions) == [0.5, 1.0]
    for tau, region in regions.items():
        assert region["type"] == "ellipse"
        assert region["tau"] == tau
        assert region["valid"] is True
        assert region["a"] > 0.0
        assert region["b"] > 0.0
        assert all(math.isfinite(value) for value in region["center"])
