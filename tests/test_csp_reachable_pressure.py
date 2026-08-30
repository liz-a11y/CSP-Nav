import importlib
import math

import numpy as np
import pytest


TAU_WEIGHTS = {0.5: 0.4, 1.0: 0.3, 1.5: 0.2, 2.0: 0.1}


def _metrics():
    try:
        return importlib.import_module("crowd_sim.social.csp_metrics")
    except ModuleNotFoundError as exc:
        pytest.fail(f"CSP metrics module is not implemented yet: {exc}")


def _robot(
    *,
    px=0.0,
    py=0.0,
    vx=0.0,
    vy=0.0,
    radius=0.2,
    active=True,
    robot_id="robot-0",
):
    return {
        "robot_id": robot_id,
        "px": px,
        "py": py,
        "vx": vx,
        "vy": vy,
        "radius": radius,
        "active": active,
    }


def _human(*, px=0.0, py=0.0, vx=0.0, vy=0.0, human_id="human-0"):
    return {
        "human_id": human_id,
        "px": px,
        "py": py,
        "vx": vx,
        "vy": vy,
        "radius": 0.3,
        "theta": 0.0,
    }


def _region(tau, center, *, a=0.25, b=0.2, theta=0.0):
    return {
        "type": "ellipse",
        "shape_type": "ellipse",
        "tau": tau,
        "center": list(center),
        "a": a,
        "b": b,
        "theta": theta,
        "valid": True,
    }


def _regions_for_robot_trajectory(robot, *, high_taus=TAU_WEIGHTS):
    high_taus = set(high_taus)
    regions = {}
    for tau in TAU_WEIGHTS:
        if tau in high_taus:
            center = [
                robot["px"] + tau * robot["vx"],
                robot["py"] + tau * robot["vy"],
            ]
        else:
            center = [100.0 + tau, 100.0]
        regions[tau] = _region(tau, center)
    return regions


def _lmte_output(regions):
    return {
        "speed": 1.0,
        "heading": 0.0,
        "reachable_regions": regions,
    }


def _assert_public_numbers_finite(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            if not str(key).startswith("_"):
                _assert_public_numbers_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        for index, item in enumerate(value):
            _assert_public_numbers_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        assert math.isfinite(float(value)), f"{path} is not finite: {value!r}"


def _assert_compact_main_metrics_match(detailed, compact):
    main_keys = {
        "CSP_scene_mean",
        "CSP_scene_max",
        "CSP_scene_CVaR",
        "mean_P_ps",
        "mean_P_enc",
        "mean_P_blk",
        "mean_P_reach",
        "max_P_reach",
    }
    for key in main_keys:
        assert abs(compact[key] - detailed[key]) <= 1.0e-10, key


def test_default_parameters_include_reachable_pressure_configuration():
    metrics = _metrics()

    assert metrics.DEFAULT_PARAMS["csp"]["lambda_reach"] == pytest.approx(0.75)
    assert metrics.DEFAULT_PARAMS["csp"]["top_k"] == 2
    assert metrics.DEFAULT_PARAMS["reachable"] == {
        "reach_threshold": 1.5,
        "reach_gain": 1.0,
        "reach_axis_margin": 0.10,
        "tau_weights": TAU_WEIGHTS,
    }


def test_robot_predicted_trajectory_inside_all_ellipses_has_high_pressure():
    metrics = _metrics()
    robot = _robot(vx=1.0, vy=0.25)
    result = metrics.compute_reachable_pressure(
        [robot],
        _human(),
        lmte_output=_lmte_output(_regions_for_robot_trajectory(robot)),
    )

    assert result["P_reach"] > 0.5
    assert len(result["horizon_terms"]) == 4
    assert result["max_reach_pressure"] == pytest.approx(1.0)
    assert result["human_id"] == "human-0"


def test_robot_predicted_trajectory_inside_capsule_mode_has_high_pressure():
    metrics = _metrics()
    robot = _robot(vx=1.0)
    regions = {
        tau: {
            "type": "capsule",
            "shape_type": "capsule",
            "tau": tau,
            "center": [0.5 * tau, 0.0],
            "centerline": [[0.0, 0.0], [tau, 0.0]],
            "radius": 0.35,
            "a": 0.35 + tau,
            "b": 0.35,
            "theta": 0.0,
            "valid": True,
        }
        for tau in TAU_WEIGHTS
    }

    result = metrics.compute_reachable_pressure(
        [robot],
        _human(),
        lmte_output=_lmte_output(regions),
    )

    assert result["P_reach"] > 0.5


def test_robot_far_from_all_reachable_ellipses_has_negligible_pressure():
    metrics = _metrics()
    robot = _robot(px=-100.0, py=-100.0)
    regions = {
        tau: _region(tau, [100.0 + tau, 100.0])
        for tau in TAU_WEIGHTS
    }

    result = metrics.compute_reachable_pressure(
        [robot],
        _human(),
        lmte_output=_lmte_output(regions),
    )

    assert result["P_reach"] < 0.01
    assert len(result["horizon_terms"]) == 4


def test_singular_reachable_region_falls_back_to_its_tau():
    metrics = _metrics()
    robot = _robot(vx=1.0)
    region = _region(1.0, [1.0, 0.0])

    result = metrics.compute_reachable_pressure(
        [robot],
        _human(),
        lmte_output={
            "speed": 1.0,
            "heading": 0.0,
            "reachable_region": region,
        },
    )

    assert result["P_reach"] > 0.0
    assert len(result["horizon_terms"]) == 4


def test_scene_csp_equals_lambda_reach_times_reachable_pressure_when_other_lambdas_zero():
    metrics = _metrics()
    robot = _robot(vx=0.5)
    human = _human()
    lmte_output = _lmte_output(_regions_for_robot_trajectory(robot))
    params = {
        "csp": {
            "lambda_ps": 0.0,
            "lambda_enc": 0.0,
            "lambda_blk": 0.0,
            "lambda_reach": 0.75,
            "top_k": 2,
        }
    }

    per_human = metrics.compute_csp_for_human(
        [robot],
        human,
        lmte_output=lmte_output,
        params=params,
    )
    scene = metrics.compute_scene_csp(
        [robot],
        [human],
        lmte_outputs={"human-0": lmte_output},
        params=params,
    )

    expected = 0.75 * per_human["P_reach"]
    assert per_human["CSP_j"] == pytest.approx(expected)
    assert scene["CSP_scene_mean"] == pytest.approx(expected)
    assert scene["CSP_scene_CVaR"] == pytest.approx(expected)


def test_inactive_robots_contribute_zero_to_all_four_pressure_terms():
    metrics = _metrics()
    inactive_robots = [
        _robot(px=-0.1, robot_id="robot-left", active=False),
        _robot(px=0.1, robot_id="robot-right", active=False),
    ]
    lmte_output = _lmte_output(
        _regions_for_robot_trajectory(inactive_robots[0])
    )

    result = metrics.compute_csp_for_human(
        inactive_robots,
        _human(),
        lmte_output=lmte_output,
    )

    assert result["P_ps"] == 0.0
    assert result["P_enc"] == 0.0
    assert result["P_blk"] == 0.0
    assert result["P_reach"] == 0.0
    assert result["CSP_j"] == 0.0


@pytest.mark.parametrize("high_tau, expected_weight", TAU_WEIGHTS.items())
def test_each_reachable_horizon_contributes_its_configured_weight(
    high_tau,
    expected_weight,
):
    metrics = _metrics()
    robot = _robot(vx=1.0)
    regions = _regions_for_robot_trajectory(robot, high_taus=[high_tau])

    result = metrics.compute_reachable_pressure(
        [robot],
        _human(),
        lmte_output=_lmte_output(regions),
    )

    assert result["P_reach"] == pytest.approx(expected_weight)
    assert len(result["horizon_terms"]) == 4


def test_missing_active_field_is_treated_as_active():
    metrics = _metrics()
    robot = _robot(vx=1.0)
    robot.pop("active")

    result = metrics.compute_reachable_pressure(
        [robot],
        _human(),
        lmte_output=_lmte_output(_regions_for_robot_trajectory(robot)),
    )

    assert result["P_reach"] > 0.5


def test_empty_human_scene_reports_zero_reachable_pressure_metrics():
    metrics = _metrics()

    result = metrics.compute_scene_csp([_robot()], [])

    assert result["mean_P_reach"] == 0.0
    assert result["max_P_reach"] == 0.0


def test_nonfinite_inputs_do_not_propagate_nan_to_scene_metrics():
    metrics = _metrics()
    robot = _robot(
        px=np.nan,
        py=np.inf,
        vx=-np.inf,
        radius=np.nan,
        active=False,
    )
    human = _human(px=np.nan, py=-np.inf, vx=np.inf)
    regions = {
        0.5: _region(0.5, [np.nan, np.inf], a=np.nan, b=np.inf, theta=np.nan),
        1.0: _region(1.0, [0.0, 0.0]),
        1.5: _region(1.5, [np.inf, -np.inf]),
        2.0: _region(2.0, [0.0, 0.0], a=-np.inf, b=np.nan),
    }

    scene = metrics.compute_scene_csp(
        [robot],
        [human],
        lmte_outputs={"human-0": _lmte_output(regions)},
    )

    numeric_scene_metrics = [
        value
        for value in scene.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    assert numeric_scene_metrics
    assert all(math.isfinite(float(value)) for value in numeric_scene_metrics)
    _assert_public_numbers_finite(
        scene["CSP_per_human"],
        path="scene.CSP_per_human",
    )
    compact = metrics.compute_scene_csp(
        [robot],
        [human],
        lmte_outputs={"human-0": _lmte_output(regions)},
        include_details=False,
    )
    _assert_public_numbers_finite(compact, path="compact")


def test_extreme_finite_active_states_do_not_overflow_scene_metrics():
    metrics = _metrics()
    robot = _robot(
        px=1.0e308,
        py=-1.0e308,
        vx=1.0e308,
        vy=-1.0e308,
        radius=1.0e308,
        active=True,
    )
    human = _human(
        px=0.0,
        py=0.0,
        vx=1.0e308,
        vy=1.0e308,
    )
    regions = {
        tau: _region(
            tau,
            [1.0e308, -1.0e308],
            a=1.0e308,
            b=1.0e308,
            theta=math.pi / 4.0,
        )
        for tau in TAU_WEIGHTS
    }

    with np.errstate(over="raise", invalid="raise"):
        scene = metrics.compute_scene_csp(
            [robot],
            [human],
            lmte_outputs={"human-0": _lmte_output(regions)},
        )
        compact = metrics.compute_scene_csp(
            [robot],
            [human],
            lmte_outputs={"human-0": _lmte_output(regions)},
            include_details=False,
        )

    _assert_public_numbers_finite(scene, path="scene")
    _assert_public_numbers_finite(compact, path="compact")


def test_compact_scene_matches_detailed_training_scalars():
    metrics = _metrics()
    robots = [
        _robot(px=-0.4, py=0.0, vx=0.6, robot_id="robot-0"),
        _robot(px=0.5, py=0.2, vx=-0.2, robot_id="robot-1"),
        _robot(px=1.2, py=-0.3, vx=0.1, robot_id="robot-2"),
    ]
    humans = [
        _human(px=0.0, py=0.0, human_id="human-0"),
        _human(px=1.0, py=0.5, vx=0.2, human_id="human-1"),
    ]
    lmte_outputs = {
        human["human_id"]: _lmte_output(
            {
                tau: _region(
                    tau,
                    [
                        human["px"] + tau * human["vx"],
                        human["py"] + tau * human["vy"],
                    ],
                    a=0.5 + 0.1 * tau,
                    b=0.35,
                )
                for tau in TAU_WEIGHTS
            }
        )
        for human in humans
    }

    detailed = metrics.compute_scene_csp(
        robots,
        humans,
        lmte_outputs=lmte_outputs,
    )
    compact = metrics.compute_scene_csp(
        robots,
        humans,
        lmte_outputs=lmte_outputs,
        include_details=False,
    )

    expected_keys = {
        "CSP_scene_mean",
        "CSP_scene_max",
        "CSP_scene_CVaR",
        "mean_P_ps",
        "mean_P_enc",
        "mean_P_blk",
        "mean_P_reach",
        "max_P_reach",
        "human_count",
    }
    assert set(compact) == expected_keys
    assert "CSP_per_human" not in compact
    assert compact["human_count"] == len(humans)
    _assert_compact_main_metrics_match(detailed, compact)
    _assert_public_numbers_finite(compact, path="compact")


def test_compact_scene_supports_center_state_key_like_detailed_path():
    metrics = _metrics()
    robot = {
        "robot_id": "robot-center",
        "center": [0.9, 0.0],
        "vx": 0.0,
        "vy": 0.0,
        "radius": 0.2,
        "active": True,
    }
    human = {
        "human_id": "human-center",
        "center": [0.0, 0.0],
        "vx": 0.0,
        "vy": 0.0,
        "radius": 0.3,
        "theta": 0.0,
    }
    params = {
        "csp": {
            "lambda_ps": 1.0,
            "lambda_enc": 0.0,
            "lambda_blk": 0.0,
            "lambda_reach": 0.0,
        }
    }

    detailed = metrics.compute_scene_csp([robot], [human], params=params)
    compact = metrics.compute_scene_csp(
        [robot],
        [human],
        params=params,
        include_details=False,
    )

    _assert_compact_main_metrics_match(detailed, compact)


def test_compact_extreme_finite_csp_weights_match_detailed_overflow_semantics():
    metrics = _metrics()
    robot = _robot()
    human = _human()
    lmte_output = _lmte_output(_regions_for_robot_trajectory(robot))
    params = {
        "csp": {
            "lambda_ps": 1.0e308,
            "lambda_enc": 1.0e308,
            "lambda_blk": 1.0e308,
            "lambda_reach": 1.0e308,
        }
    }

    detailed = metrics.compute_scene_csp(
        [robot],
        [human],
        lmte_outputs={"human-0": lmte_output},
        params=params,
    )
    compact = metrics.compute_scene_csp(
        [robot],
        [human],
        lmte_outputs={"human-0": lmte_output},
        params=params,
        include_details=False,
    )

    _assert_public_numbers_finite(detailed, path="detailed")
    _assert_public_numbers_finite(compact, path="compact")
    _assert_compact_main_metrics_match(detailed, compact)


def test_compact_scene_normal_inputs_do_not_use_numpy_vector_helper(monkeypatch):
    metrics = _metrics()
    robots = [
        _robot(px=-0.4, py=0.1, vx=0.3, robot_id="robot-0"),
        _robot(px=0.6, py=-0.2, vy=0.2, robot_id="robot-1"),
    ]
    human = _human(px=0.2, py=0.3, vx=0.1)
    lmte_output = _lmte_output(
        {
            tau: _region(
                tau,
                [human["px"] + tau * human["vx"], human["py"]],
                a=0.5,
                b=0.35,
            )
            for tau in TAU_WEIGHTS
        }
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("compact path used the NumPy vector helper")

    monkeypatch.setattr(metrics, "_bounded_vec2", fail_if_called)

    scene = metrics.compute_scene_csp(
        robots,
        [human],
        lmte_outputs={"human-0": lmte_output},
        include_details=False,
    )

    assert scene["human_count"] == 1
    _assert_public_numbers_finite(scene, path="scene")


def test_compact_paths_do_not_call_detailed_construction_helpers(monkeypatch):
    metrics = _metrics()
    robot = _robot(vx=0.5)
    human = _human()
    lmte_output = _lmte_output(_regions_for_robot_trajectory(robot))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("compact path called a detailed construction helper")

    for helper_name in (
        "compute_personal_space_pressure",
        "compute_enclosure_pressure",
        "compute_blocking_pressure",
        "compute_reachable_pressure",
    ):
        monkeypatch.setattr(metrics, helper_name, fail_if_called)

    per_human = metrics.compute_csp_for_human(
        [robot],
        human,
        lmte_output=lmte_output,
        include_details=False,
    )
    scene = metrics.compute_scene_csp(
        [robot],
        [human],
        lmte_outputs={"human-0": lmte_output},
        include_details=False,
    )

    assert set(per_human) == {"P_ps", "P_enc", "P_blk", "P_reach", "CSP_j"}
    assert "CSP_per_human" not in scene
    _assert_public_numbers_finite(per_human, path="per_human")
    _assert_public_numbers_finite(scene, path="scene")


def test_reachable_regions_are_iterated_once_per_human_not_once_per_robot():
    metrics = _metrics()

    class CountingRegions(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.items_calls = 0

        def items(self):
            self.items_calls += 1
            return super().items()

    robot_a = _robot(vx=0.5, robot_id="robot-a")
    robot_b = _robot(vx=0.25, robot_id="robot-b")
    regions = CountingRegions(_regions_for_robot_trajectory(robot_a))

    result = metrics.compute_reachable_pressure(
        [robot_a, robot_b],
        _human(),
        lmte_output=_lmte_output(regions),
    )

    assert len(result["horizon_terms"]) == 4
    assert regions.items_calls == 1


TASK3_REWARD_INFO_KEYS = {
    "r_csp",
    "CSP_scene_mean",
    "CSP_scene_max",
    "CSP_scene_CVaR",
    "mean_P_ps",
    "mean_P_enc",
    "mean_P_blk",
    "mean_P_reach",
    "max_P_reach",
}

TASK3_FILTER_INFO_KEYS = {
    "csp_action_filter_used",
    "candidate_count",
    "selected_index",
    "selected_CSP_scene_CVaR",
    "selected_score",
    "filtered_by_csp_action_filter",
    "feasible_candidate_count",
}


def _reward_module():
    try:
        return importlib.import_module("crowd_sim.social.csp_reward")
    except ModuleNotFoundError as exc:
        pytest.fail(f"CSP reward module is not implemented yet: {exc}")


def _action_filter_module():
    try:
        return importlib.import_module("crowd_sim.social.csp_action_filter")
    except ModuleNotFoundError as exc:
        pytest.fail(f"CSP action filter module is not implemented yet: {exc}")


def _compact_scene(cvar=0.0):
    return {
        "CSP_scene_mean": float(cvar),
        "CSP_scene_max": float(cvar),
        "CSP_scene_CVaR": float(cvar),
        "mean_P_ps": 0.1,
        "mean_P_enc": 0.2,
        "mean_P_blk": 0.3,
        "mean_P_reach": 0.4,
        "max_P_reach": 0.5,
        "human_count": 1,
    }


def _filter_config(*, enabled=True, max_action_norm=1.0):
    return {
        "features": {"enable_csp_action_filter": enabled},
        "csp_action_filter": {
            "horizon_steps": 1,
            "dt": 1.0,
            "csp_threshold": 0.5,
            "task_cost_weight": 0.01,
            "csp_cost_weight": 1.0,
            "threshold_violation_weight": 10.0,
            "deviation_weight": 0.0,
            "max_action_norm": max_action_norm,
        },
    }


class _FixedMetrics:
    def __init__(self, cvars):
        self.cvars = list(cvars)
        self.calls = []

    def compute_scene_csp(
        self,
        robot_states,
        human_states,
        lmte_outputs=None,
        include_details=True,
    ):
        assert include_details is False
        call_index = len(self.calls)
        self.calls.append(
            {
                "robot_states": robot_states,
                "human_states": human_states,
            }
        )
        return _compact_scene(cvar=self.cvars[call_index])


class _RecordingMetrics:
    def __init__(self):
        self.robot_positions = []

    def compute_scene_csp(
        self,
        robot_states,
        human_states,
        lmte_outputs=None,
        include_details=True,
    ):
        assert include_details is False
        self.robot_positions.append(
            [(float(robot["px"]), float(robot["py"])) for robot in robot_states]
        )
        return _compact_scene(cvar=0.0)


class _StateRecordingMetrics:
    def __init__(self):
        self.calls = []

    def compute_scene_csp(
        self,
        robot_states,
        human_states,
        lmte_outputs=None,
        include_details=True,
    ):
        assert include_details is False
        self.calls.append(
            {
                "robot_states": robot_states,
                "human_states": human_states,
            }
        )
        return _compact_scene(cvar=0.0)


def test_csp_reward_only_changes_robots_active_at_step_start_without_mutating_input():
    reward_module = _reward_module()
    wrapper = reward_module.CSPRewardWrapper(
        {"features": {"enable_csp_reward": True}}
    )
    wrapper.csp_metrics.compute_scene_csp = (
        lambda *args, **kwargs: _compact_scene(cvar=1.0)
    )
    env_reward = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    original = env_reward.copy()

    shaped, info = wrapper.compute(
        env_reward,
        np.asarray([True, False, True]),
        [_robot(robot_id=f"robot-{index}") for index in range(3)],
        [_human()],
    )

    np.testing.assert_allclose(shaped, [0.98, 2.0, 2.98])
    np.testing.assert_array_equal(env_reward, original)
    assert shaped.shape == env_reward.shape
    assert shaped.dtype == env_reward.dtype
    assert info["r_csp"] == pytest.approx(-0.02)
    assert set(info) == TASK3_REWARD_INFO_KEYS


def test_csp_reward_integer_input_returns_float32_without_truncating_penalty():
    reward_module = _reward_module()
    wrapper = reward_module.CSPRewardWrapper({})
    wrapper.csp_metrics.compute_scene_csp = (
        lambda *args, **kwargs: _compact_scene(cvar=1.0)
    )
    env_reward = np.asarray([1, 2, 3], dtype=np.int32)
    original = env_reward.copy()

    shaped, _ = wrapper.compute(
        env_reward,
        np.asarray([True, False, True]),
        [_robot(robot_id=f"robot-{index}") for index in range(3)],
        [_human()],
    )

    assert shaped.dtype == np.float32
    assert shaped.shape == env_reward.shape
    np.testing.assert_allclose(shaped, [0.98, 2.0, 2.98], atol=1.0e-6)
    np.testing.assert_array_equal(env_reward, original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_csp_reward_floating_input_preserves_dtype_and_shape(dtype):
    reward_module = _reward_module()
    wrapper = reward_module.CSPRewardWrapper({})
    wrapper.csp_metrics.compute_scene_csp = (
        lambda *args, **kwargs: _compact_scene(cvar=0.5)
    )
    env_reward = np.asarray([[1.0], [2.0]], dtype=dtype)

    shaped, _ = wrapper.compute(
        env_reward,
        np.asarray([True, True]),
        [_robot(robot_id="robot-0"), _robot(robot_id="robot-1")],
        [_human()],
    )

    assert shaped.dtype == env_reward.dtype
    assert shaped.shape == env_reward.shape
    np.testing.assert_allclose(
        shaped,
        np.asarray([[0.99], [1.99]], dtype=dtype),
        atol=2.0e-3,
    )


def test_csp_reward_uses_compact_scene_metrics(monkeypatch):
    reward_module = _reward_module()
    wrapper = reward_module.CSPRewardWrapper({})
    calls = []

    def compact_only(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["include_details"] is False
        return _compact_scene(cvar=0.25)

    monkeypatch.setattr(wrapper.csp_metrics, "compute_scene_csp", compact_only)

    wrapper.compute(
        np.asarray([1.0], dtype=np.float64),
        np.asarray([True]),
        [_robot()],
        [_human()],
    )

    assert calls == [{"include_details": False}]


def test_disabled_csp_reward_adds_zero_and_keeps_compact_scalar_info(monkeypatch):
    reward_module = _reward_module()
    wrapper = reward_module.CSPRewardWrapper(
        {"features": {"enable_csp_reward": False}}
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled reward should not compute CSP")

    monkeypatch.setattr(wrapper.csp_metrics, "compute_scene_csp", fail_if_called)
    env_reward = np.asarray([1.0, 2.0], dtype=np.float32)

    shaped, info = wrapper.compute(
        env_reward,
        np.asarray([True, True]),
        [_robot(robot_id="robot-0"), _robot(robot_id="robot-1")],
        [_human()],
    )

    np.testing.assert_array_equal(shaped, env_reward)
    assert info["r_csp"] == 0.0
    assert set(info) == TASK3_REWARD_INFO_KEYS
    assert all(np.isscalar(value) for value in info.values())


def test_csp_reporting_computes_metrics_without_changing_reward(monkeypatch):
    reward_module = _reward_module()
    wrapper = reward_module.CSPRewardWrapper(
        {
            "features": {
                "enable_csp_reward": False,
                "enable_csp_reporting": True,
            }
        }
    )
    monkeypatch.setattr(
        wrapper.csp_metrics,
        "compute_scene_csp",
        lambda *args, **kwargs: _compact_scene(cvar=0.75),
    )
    env_reward = np.asarray([1.0, 2.0], dtype=np.float32)

    shaped, info = wrapper.compute(
        env_reward,
        np.asarray([True, True]),
        [_robot(robot_id="robot-0"), _robot(robot_id="robot-1")],
        [_human()],
    )

    np.testing.assert_array_equal(shaped, env_reward)
    assert info["r_csp"] == 0.0
    assert info["CSP_scene_CVaR"] == pytest.approx(0.75)


def test_csp_action_filter_selects_zero_velocity_low_pressure_candidate(monkeypatch):
    filter_module = _action_filter_module()
    action_filter = filter_module.CSPActionFilter(_filter_config())
    calls = []

    def pressure_from_rollout(robot_states, human_states, lmte_outputs=None, **kwargs):
        calls.append(kwargs)
        pressure = 2.0 if robot_states[0]["px"] > 0.5 else 0.0
        return _compact_scene(cvar=pressure)

    monkeypatch.setattr(
        action_filter.metrics,
        "compute_scene_csp",
        pressure_from_rollout,
    )
    candidates = np.asarray([[[1.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32)

    selected, info = action_filter.select_action(
        candidates,
        [_robot()],
        [_human(px=5.0)],
        nominal_action=candidates[0],
    )

    np.testing.assert_array_equal(selected, candidates[1])
    assert info["selected_index"] == 1
    assert info["filtered_by_csp_action_filter"] == 1
    assert info["feasible_candidate_count"] == 1
    assert calls


@pytest.mark.parametrize(
    "enabled,candidates",
    [
        (
            False,
            np.asarray(
                [[[3.0, 4.0]], [[0.0, 0.0]]],
                dtype=np.float64,
            ),
        ),
        (
            True,
            np.asarray([[[3.0, 4.0]]], dtype=np.float64),
        ),
    ],
)
def test_csp_action_filter_disabled_or_single_candidate_returns_index_zero(
    enabled,
    candidates,
):
    filter_module = _action_filter_module()
    action_filter = filter_module.CSPActionFilter(
        _filter_config(enabled=enabled, max_action_norm=1.0)
    )

    selected, info = action_filter.select_action(
        candidates,
        [_robot()],
        [_human()],
        nominal_action=np.asarray([[3.0, 4.0]]),
    )

    assert info["selected_index"] == 0
    assert info["csp_action_filter_used"] == 0
    assert np.linalg.norm(selected[0]) == pytest.approx(1.0)
    assert np.all(np.isfinite(selected))


@pytest.mark.parametrize(
    "enabled,candidates",
    [
        (
            False,
            np.asarray(
                [[[0.5, 0.0]], [[0.0, 0.0]]],
                dtype=np.float64,
            ),
        ),
        (
            True,
            np.asarray([[[0.5, 0.0]]], dtype=np.float64),
        ),
    ],
)
def test_csp_action_filter_early_return_does_not_validate_invalid_nominal(
    enabled,
    candidates,
):
    filter_module = _action_filter_module()
    action_filter = filter_module.CSPActionFilter(
        _filter_config(enabled=enabled)
    )

    selected, info = action_filter.select_action(
        candidates,
        [_robot()],
        [_human()],
        nominal_action=np.asarray([1.0, 2.0, 3.0]),
    )

    np.testing.assert_array_equal(selected, candidates[0].astype(np.float32))
    assert info["selected_index"] == 0
    assert info["csp_action_filter_used"] == 0


def test_csp_action_filter_rejects_candidate_robot_count_mismatch():
    filter_module = _action_filter_module()
    action_filter = filter_module.CSPActionFilter(_filter_config())
    candidates = np.asarray(
        [
            [[0.5, 0.0]],
            [[0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="robot"):
        action_filter.select_action(
            candidates,
            [
                _robot(robot_id="robot-0"),
                _robot(robot_id="robot-1"),
            ],
            [_human()],
        )


def test_csp_action_filter_hot_path_does_not_call_deepcopy(monkeypatch):
    filter_module = _action_filter_module()
    action_filter = filter_module.CSPActionFilter(_filter_config())
    recording_metrics = _StateRecordingMetrics()
    action_filter.metrics = recording_metrics

    def fail_if_called(*args, **kwargs):
        raise AssertionError("deepcopy used in CSP action-filter hot path")

    monkeypatch.setattr(filter_module, "deepcopy", fail_if_called, raising=False)
    nested = {"large": [object(), object()]}
    robots = [
        {
            **_robot(),
            "goal": [10.0, 0.0],
            "unrelated": nested,
        }
    ]
    humans = [
        {
            **_human(),
            "unrelated": nested,
        }
    ]

    selected, info = action_filter.select_action(
        np.asarray(
            [
                [[0.5, 0.0]],
                [[0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        robots,
        humans,
        nominal_action=np.asarray([[0.5, 0.0]]),
    )

    assert info["candidate_count"] == 2
    assert selected.shape == (1, 2)
    assert recording_metrics.calls
    for call in recording_metrics.calls:
        assert "unrelated" not in call["robot_states"][0]
        assert "unrelated" not in call["human_states"][0]


@pytest.mark.parametrize("horizon_steps", [0, 9, 1.5, True, False])
def test_csp_action_filter_rejects_invalid_horizon_steps(horizon_steps):
    filter_module = _action_filter_module()
    config = _filter_config()
    config["csp_action_filter"]["horizon_steps"] = horizon_steps

    with pytest.raises(ValueError, match="horizon_steps"):
        filter_module.CSPActionFilter(config)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_csp_action_filter_rejects_invalid_dt(dt):
    filter_module = _action_filter_module()
    config = _filter_config()
    config["csp_action_filter"]["dt"] = dt

    with pytest.raises(ValueError, match="dt"):
        filter_module.CSPActionFilter(config)


@pytest.mark.parametrize(
    "field",
    [
        "csp_threshold",
        "task_cost_weight",
        "csp_cost_weight",
        "threshold_violation_weight",
        "deviation_weight",
        "max_action_norm",
    ],
)
@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf])
def test_csp_action_filter_rejects_nonfinite_or_negative_cost_config(
    field,
    value,
):
    filter_module = _action_filter_module()
    config = _filter_config()
    config["csp_action_filter"][field] = value

    with pytest.raises(ValueError, match=field):
        filter_module.CSPActionFilter(config)


def test_csp_action_filter_zero_max_norm_clips_each_robot_action_to_zero():
    filter_module = _action_filter_module()
    action_filter = filter_module.CSPActionFilter(
        _filter_config(enabled=False, max_action_norm=0.0)
    )
    candidates = np.asarray(
        [[[3.0, 4.0], [-6.0, 8.0]]],
        dtype=np.float64,
    )

    selected, info = action_filter.select_action(
        candidates,
        [
            _robot(robot_id="robot-0"),
            _robot(robot_id="robot-1"),
        ],
        [_human()],
        nominal_action=candidates[0],
    )

    np.testing.assert_array_equal(selected, np.zeros((2, 2), dtype=np.float32))
    assert info["selected_index"] == 0


def test_feasible_selected_score_matches_task_csp_and_deviation_formula():
    filter_module = _action_filter_module()
    config = _filter_config(max_action_norm=10.0)
    config["csp_action_filter"].update(
        {
            "csp_threshold": 1.0,
            "task_cost_weight": 2.0,
            "csp_cost_weight": 3.0,
            "threshold_violation_weight": 100.0,
            "deviation_weight": 4.0,
        }
    )
    action_filter = filter_module.CSPActionFilter(config)
    action_filter.metrics = _FixedMetrics([0.2, 0.4])
    candidates = np.asarray(
        [
            [[1.0, 0.0]],
            [[0.5, 0.0]],
        ],
        dtype=np.float64,
    )

    _, info = action_filter.select_action(
        candidates,
        [{**_robot(), "gx": 10.0, "gy": 0.0}],
        [],
        nominal_action=np.asarray([[0.0, 0.0]]),
    )

    task_cost = -0.5
    cvar = 0.4
    deviation = 0.5
    expected = 2.0 * task_cost + 3.0 * cvar + 4.0 * deviation
    assert info["selected_index"] == 1
    assert info["feasible_candidate_count"] == 2
    assert info["selected_score"] == pytest.approx(expected, abs=1.0e-12)


def test_all_infeasible_selected_score_adds_threshold_violation_penalty():
    filter_module = _action_filter_module()
    config = _filter_config(max_action_norm=10.0)
    config["csp_action_filter"].update(
        {
            "csp_threshold": 0.5,
            "task_cost_weight": 1.0,
            "csp_cost_weight": 2.0,
            "threshold_violation_weight": 10.0,
            "deviation_weight": 0.0,
        }
    )
    action_filter = filter_module.CSPActionFilter(config)
    action_filter.metrics = _FixedMetrics([1.0, 0.8])
    candidates = np.asarray(
        [
            [[1.0, 0.0]],
            [[0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    _, info = action_filter.select_action(
        candidates,
        [{**_robot(), "gx": 10.0, "gy": 0.0}],
        [],
        nominal_action=candidates[0],
    )

    task_cost = 0.0
    cvar = 0.8
    deviation = 1.0
    expected = (
        1.0 * task_cost
        + 2.0 * cvar
        + 0.0 * deviation
        + 10.0 * max(0.0, cvar - 0.5)
    )
    assert info["feasible_candidate_count"] == 0
    assert info["selected_index"] == 1
    assert info["selected_score"] == pytest.approx(expected, abs=1.0e-12)


def test_action_deviation_is_mean_per_robot_l2_distance_from_nominal():
    filter_module = _action_filter_module()
    config = _filter_config(max_action_norm=10.0)
    config["csp_action_filter"].update(
        {
            "csp_threshold": 1.0,
            "task_cost_weight": 0.0,
            "csp_cost_weight": 20.0,
            "threshold_violation_weight": 0.0,
            "deviation_weight": 2.0,
        }
    )
    action_filter = filter_module.CSPActionFilter(config)
    action_filter.metrics = _FixedMetrics([0.9, 0.0])
    candidates = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[3.0, 0.0], [0.0, 4.0]],
        ],
        dtype=np.float64,
    )

    _, info = action_filter.select_action(
        candidates,
        [
            _robot(robot_id="robot-0"),
            _robot(robot_id="robot-1"),
        ],
        [],
        nominal_action=candidates[0],
    )

    expected_deviation = (3.0 + 4.0) / 2.0
    assert info["selected_index"] == 1
    assert info["selected_score"] == pytest.approx(
        2.0 * expected_deviation,
        abs=1.0e-12,
    )


def test_horizon_steps_two_advances_robot_twice_before_metrics_call():
    filter_module = _action_filter_module()
    config = _filter_config(max_action_norm=10.0)
    config["csp_action_filter"].update(
        {
            "horizon_steps": 2,
            "dt": 1.0,
            "task_cost_weight": 0.0,
            "csp_cost_weight": 0.0,
            "threshold_violation_weight": 0.0,
            "deviation_weight": 0.0,
        }
    )
    action_filter = filter_module.CSPActionFilter(config)
    recording_metrics = _RecordingMetrics()
    action_filter.metrics = recording_metrics

    action_filter.select_action(
        np.asarray(
            [
                [[1.0, 0.0]],
                [[0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        [_robot(px=0.0, py=0.0)],
        [],
        nominal_action=np.asarray([[0.0, 0.0]]),
    )

    assert recording_metrics.robot_positions == [
        [(2.0, 0.0)],
        [(0.0, 0.0)],
    ]


def test_inactive_robot_does_not_move_and_has_zero_rollout_velocity():
    filter_module = _action_filter_module()
    config = _filter_config(max_action_norm=10.0)
    config["csp_action_filter"]["horizon_steps"] = 2
    action_filter = filter_module.CSPActionFilter(config)
    recording_metrics = _StateRecordingMetrics()
    action_filter.metrics = recording_metrics
    robots = [
        _robot(
            px=0.0,
            py=0.0,
            vx=0.0,
            vy=0.0,
            active=True,
            robot_id="active",
        ),
        _robot(
            px=5.0,
            py=6.0,
            vx=3.0,
            vy=4.0,
            active=False,
            robot_id="inactive",
        ),
    ]

    action_filter.select_action(
        np.asarray(
            [
                [[1.0, 0.0], [9.0, 0.0]],
                [[0.0, 0.0], [-9.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        robots,
        [],
        nominal_action=np.zeros((2, 2), dtype=np.float64),
    )

    assert len(recording_metrics.calls) == 2
    for call in recording_metrics.calls:
        inactive = call["robot_states"][1]
        assert inactive["px"] == pytest.approx(5.0)
        assert inactive["py"] == pytest.approx(6.0)
        assert inactive["vx"] == pytest.approx(0.0)
        assert inactive["vy"] == pytest.approx(0.0)
        assert inactive["velocity"] == [0.0, 0.0]


def test_csp_action_filter_uses_compact_scene_metrics(monkeypatch):
    filter_module = _action_filter_module()
    action_filter = filter_module.CSPActionFilter(_filter_config())
    calls = []

    def compact_only(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["include_details"] is False
        return _compact_scene(cvar=0.0)

    monkeypatch.setattr(action_filter.metrics, "compute_scene_csp", compact_only)
    candidates = np.asarray([[[0.1, 0.0]], [[0.0, 0.0]]], dtype=np.float32)

    action_filter.select_action(
        candidates,
        [_robot()],
        [_human()],
        nominal_action=candidates[0],
    )

    assert len(calls) == len(candidates)
    assert all(call == {"include_details": False} for call in calls)


def test_csp_action_filter_sanitizes_nonfinite_candidates_and_scores(monkeypatch):
    filter_module = _action_filter_module()
    action_filter = filter_module.CSPActionFilter(_filter_config())

    def nonfinite_scene(*args, **kwargs):
        result = _compact_scene(cvar=0.0)
        result["CSP_scene_CVaR"] = np.nan
        return result

    monkeypatch.setattr(
        action_filter.metrics,
        "compute_scene_csp",
        nonfinite_scene,
    )
    candidates = np.asarray(
        [
            [[np.nan, np.inf]],
            [[-np.inf, 0.0]],
        ],
        dtype=np.float64,
    )

    first_selected, first_info = action_filter.select_action(
        candidates,
        [_robot(px=np.nan)],
        [_human(px=np.inf)],
        nominal_action=np.asarray([[np.nan, np.inf]]),
    )
    second_selected, second_info = action_filter.select_action(
        candidates,
        [_robot(px=np.nan)],
        [_human(px=np.inf)],
        nominal_action=np.asarray([[np.nan, np.inf]]),
    )

    assert first_info["selected_index"] == second_info["selected_index"]
    np.testing.assert_array_equal(first_selected, second_selected)
    assert np.all(np.isfinite(first_selected))
    assert math.isfinite(float(first_info["selected_CSP_scene_CVaR"]))
    assert math.isfinite(float(first_info["selected_score"]))


def test_csp_action_filter_info_contains_only_requested_compact_scalars(monkeypatch):
    filter_module = _action_filter_module()
    action_filter = filter_module.CSPActionFilter(_filter_config())
    monkeypatch.setattr(
        action_filter.metrics,
        "compute_scene_csp",
        lambda *args, **kwargs: _compact_scene(cvar=0.0),
    )

    _, info = action_filter.select_action(
        np.asarray([[[0.1, 0.0]], [[0.0, 0.0]]], dtype=np.float32),
        [_robot()],
        [_human()],
        nominal_action=np.asarray([[0.1, 0.0]], dtype=np.float32),
    )

    assert set(info) == TASK3_FILTER_INFO_KEYS
    assert all(np.isscalar(value) for value in info.values())


def test_inactive_robot_does_not_affect_action_filter_task_progress(monkeypatch):
    filter_module = _action_filter_module()
    config = _filter_config()
    config["csp_action_filter"]["csp_cost_weight"] = 0.0
    config["csp_action_filter"]["task_cost_weight"] = 1.0
    action_filter = filter_module.CSPActionFilter(config)
    monkeypatch.setattr(
        action_filter.metrics,
        "compute_scene_csp",
        lambda *args, **kwargs: _compact_scene(cvar=0.0),
    )
    robots = [
        {
            **_robot(robot_id="active", active=True),
            "gx": 0.0,
            "gy": 0.0,
        },
        {
            **_robot(robot_id="inactive", active=False),
            "gx": 10.0,
            "gy": 0.0,
        },
    ]
    candidates = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    _, info = action_filter.select_action(
        candidates,
        robots,
        [_human(px=100.0, py=100.0)],
        nominal_action=candidates[0],
    )

    assert info["selected_index"] == 0
