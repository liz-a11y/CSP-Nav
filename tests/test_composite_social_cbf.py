import numpy as np
import pytest

from crowd_sim.social.composite_social_cbf import (
    LMTEAwareCompositeCBFSafetyShield,
)


def test_qp_reduces_an_action_that_violates_robot_human_cbf():
    config = {
        "features": {
            "enable_cbf_filter": True,
            "use_cbf_qp": True,
        },
        "social_cbf": {
            "alpha": 0.5,
            "dt": 1.0,
            "control_limit": 1.0,
            "robot_human_margin": 0.1,
            "robot_margin": 0.1,
            "obstacle_margin": 0.1,
            "wall_margin": 0.1,
            "lmte_axis_margin": 0.0,
            "lmte_uncertainty_margin": 0.25,
            "csp_threshold": 100.0,
            "csp_weight": 1.0,
            "min_h_clip": 20.0,
        },
    }
    robot = {
        "robot_id": 0,
        "position": [0.0, 0.0],
        "velocity": [0.0, 0.0],
        "radius": 0.2,
        "active": True,
    }
    human = {
        "human_id": "human-0",
        "position": [1.0, 0.0],
        "velocity": [0.0, 0.0],
        "radius": 0.3,
    }
    lmte = {
        "human-0": {
            "uncertainty": 0.0,
            "v_hat": [0.0, 0.0],
        }
    }
    nominal = np.asarray([[0.8, 0.0]], dtype=np.float64)

    safe, info = LMTEAwareCompositeCBFSafetyShield(config).filter(
        nominal,
        [robot],
        [human],
        lmte,
        fallback_action=np.zeros_like(nominal),
    )

    assert safe[0, 0] == pytest.approx(0.2, abs=2.0e-3)
    assert safe[0, 1] == pytest.approx(0.0, abs=1.0e-6)
    assert info["cbf_used"] == 1
    assert info["cbf_intervened"] == 1
    assert info["qp_success"] == 1
    assert info["qp_infeasible"] == 0
    assert info["fallback_used"] == 0
    assert info["min_h_source"] == "robot_human"
