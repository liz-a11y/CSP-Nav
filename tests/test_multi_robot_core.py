import numpy as np
import pytest

from crowd_sim.multi_robot_core import (
    AgentStatus,
    build_agent_masks,
    circle_clearance,
    count_crossing_pairs,
    mix_active_rewards,
    segment_distance,
)


def test_mix_active_rewards_excludes_previously_inactive_agents():
    rewards = np.array([10.0, -20.0, 999.0], dtype=np.float32)
    active_at_step_start = np.array([True, True, False])

    mixed, team = mix_active_rewards(
        rewards,
        active_at_step_start,
        individual_coef=0.8,
        team_coef=0.2,
    )

    assert team == pytest.approx(-5.0)
    np.testing.assert_allclose(mixed, [7.0, -17.0, 0.0])


def test_mix_active_rewards_rejects_invalid_coefficients():
    with pytest.raises(ValueError, match="sum to 1"):
        mix_active_rewards(
            np.zeros(3),
            np.ones(3, dtype=bool),
            individual_coef=0.7,
            team_coef=0.2,
        )


def test_agent_masks_separate_agent_done_from_environment_done():
    previous = np.array([AgentStatus.ACTIVE] * 3, dtype=object)
    current = np.array(
        [AgentStatus.REACHED, AgentStatus.ACTIVE, AgentStatus.COLLIDED],
        dtype=object,
    )

    masks = build_agent_masks(previous, current, timeout=False)

    np.testing.assert_array_equal(masks.agent_dones, [True, False, True])
    np.testing.assert_array_equal(masks.active_masks[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(masks.rnn_masks[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(masks.bad_masks[:, 0], [1.0, 1.0, 1.0])
    assert masks.env_done is False


def test_timeout_ends_environment_and_marks_active_agents_as_bad_transitions():
    status = np.array(
        [AgentStatus.REACHED, AgentStatus.ACTIVE, AgentStatus.ACTIVE],
        dtype=object,
    )

    masks = build_agent_masks(status, status, timeout=True)

    assert masks.env_done is True
    np.testing.assert_array_equal(masks.agent_dones, [False, True, True])
    np.testing.assert_array_equal(masks.bad_masks[:, 0], [1.0, 0.0, 0.0])


def test_timeout_does_not_override_reach_or_collision_from_same_step():
    previous = np.array([AgentStatus.ACTIVE] * 3, dtype=object)
    current = np.array(
        [AgentStatus.REACHED, AgentStatus.ACTIVE, AgentStatus.COLLIDED],
        dtype=object,
    )

    masks = build_agent_masks(previous, current, timeout=True)

    np.testing.assert_array_equal(masks.agent_dones, [True, True, True])
    np.testing.assert_array_equal(masks.bad_masks[:, 0], [1.0, 0.0, 1.0])


def test_circle_clearance_is_surface_to_surface_distance():
    assert circle_clearance((0.0, 0.0), 0.2, (1.0, 0.0), 0.3) == pytest.approx(0.5)


def test_segment_distance_is_zero_for_crossing_routes():
    assert segment_distance((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)) == pytest.approx(0.0)


def test_segment_distance_handles_parallel_routes():
    assert segment_distance((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0)) == pytest.approx(1.0)


def test_count_crossing_pairs_counts_each_pair_once():
    starts = np.array([[0.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    goals = np.array([[2.0, 2.0], [2.0, 0.0], [3.0, 2.0]])

    assert count_crossing_pairs(starts, goals, threshold=0.1) == 1
