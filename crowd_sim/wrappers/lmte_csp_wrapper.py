from __future__ import annotations

from copy import deepcopy

import gym
import numpy as np

from crowd_sim.social.csp_action_filter import CSPActionFilter
from crowd_sim.social.csp_reward import CSPRewardWrapper
from crowd_sim.social.composite_social_cbf import CompositeSocialCBFFilter
from crowd_sim.social.lmte import LightweightMotionTrendEstimator
from crowd_sim.social.mappo_social_adapter import MAPPOSocialAdapter


SOCIAL_CONFIG_SECTIONS = (
    "features",
    "reward",
    "lmte",
    "csp",
    "personal_space",
    "enclosure",
    "blocking",
    "reachable",
    "csp_action_filter",
    "social_cbf",
    "lcsp",
)


def social_config_to_dict(config):
    runtime = {}
    for name in SOCIAL_CONFIG_SECTIONS:
        section = (
            config.get(name, {})
            if isinstance(config, dict)
            else getattr(config, name, {})
        )
        if section is None:
            runtime[name] = {}
        elif isinstance(section, dict):
            runtime[name] = deepcopy(section)
        else:
            runtime[name] = {
                key: deepcopy(value)
                for key, value in vars(section).items()
                if not key.startswith("_")
            }
    return runtime


def _disabled_filter_info():
    return {
        "csp_action_filter_used": 0,
        "selected_index": 0,
        "filtered_by_csp_action_filter": 0,
    }


def _disabled_cbf_info():
    return {
        "cbf_used": 0,
        "cbf_intervened": 0,
        "min_h_before": 0.0,
        "min_h_after": 0.0,
        "cbf_condition": 0.0,
        "qp_success": 0,
        "qp_infeasible": 0,
        "fallback_used": 0,
        "action_deviation": 0.0,
        "min_h_source": "none",
    }


class LMTECSPWrapper(gym.Wrapper):
    """Thin stateful orchestration layer for LMTE and CSP."""

    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        runtime = social_config_to_dict(config)
        self.features = runtime["features"]
        self.adapter = MAPPOSocialAdapter(env)
        self.lmte = LightweightMotionTrendEstimator(**runtime["lmte"])
        self.reward_wrapper = CSPRewardWrapper(runtime)
        self.action_filter = CSPActionFilter(runtime)
        self.social_cbf_filter = CompositeSocialCBFFilter(runtime)
        self.cbf_filter = self.social_cbf_filter
        self.filter_interval = max(
            1, int(runtime["csp_action_filter"].get("interval", 4))
        )
        self.cbf_interval = max(1, int(runtime["social_cbf"].get("cbf_interval", 1)))
        self.step_index = 0
        self.lmte_outputs = {}
        self.cached_robot_states = []
        self.cached_human_states = []

    def _global_time(self):
        return float(self.adapter._env_attr("global_time"))

    def _update_lmte(self, human_states):
        if not bool(self.features.get("enable_lmte", True)):
            self.lmte.reset()
            self.lmte_outputs = {}
            return

        active_ids = {human["human_id"] for human in human_states}
        for stale_id in set(self.lmte.batch_estimate()) - active_ids:
            self.lmte.reset(stale_id)

        outputs = {}
        timestamp = self._global_time()
        for human in human_states:
            human_id = human["human_id"]
            trend = dict(
                self.lmte.update(
                    human_id,
                    human["position"],
                    velocity=human["velocity"],
                    timestamp=timestamp,
                    valid=True,
                )
            )
            regions = self.lmte.reachable_regions_from_estimate(
                human_id, trend
            )
            trend["reachable_regions"] = regions
            trend["reachable_region"] = regions[1.0]
            outputs[human_id] = trend
        self.lmte_outputs = outputs

    def _clearance_context(self):
        context = {}
        try:
            context["arena_size"] = float(self.adapter._env_attr("arena_size"))
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            sim_config = getattr(self.adapter._env_attr("config"), "sim", None)
            context["borders"] = bool(getattr(sim_config, "borders", False))
            context["human_pos_noise_range"] = float(
                getattr(sim_config, "human_pos_noise_range", 0.0)
            )
        except (AttributeError, TypeError, ValueError):
            context.setdefault("borders", False)
            context.setdefault("human_pos_noise_range", 0.0)
        for name in ("cur_obstacles", "obstacles"):
            try:
                obstacles = self.adapter._env_attr(name)
            except AttributeError:
                continue
            context["obstacles"] = np.asarray(obstacles, dtype=np.float64)
            break
        return context

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.lmte.reset()
        self.step_index = 0
        self.cached_robot_states = self.adapter.robot_states()
        self.cached_human_states = self.adapter.human_states()
        self._update_lmte(self.cached_human_states)
        return result

    def step(self, actions):
        robot_count = len(self.cached_robot_states)
        raw_actions = self.adapter._validated_actions(
            actions, robot_count
        ).copy()
        active_at_start = np.asarray(
            [state["active"] for state in self.cached_robot_states],
            dtype=bool,
        )
        executed_actions = raw_actions.copy()
        filter_info = _disabled_filter_info()

        if (
            self.action_filter.enabled
            and self.step_index % self.filter_interval == 0
        ):
            braking_actions = self.adapter.braking_actions()
            if not np.array_equal(raw_actions, braking_actions):
                candidate_actions = np.stack(
                    [raw_actions, braking_actions]
                )
                candidate_velocities = np.stack(
                    [
                        self.adapter.preview_world_velocities(candidate)
                        for candidate in candidate_actions
                    ]
                )
                _, filter_info = self.action_filter.select_action(
                    candidate_velocities,
                    self.cached_robot_states,
                    self.cached_human_states,
                    self.lmte_outputs,
                    nominal_action=candidate_velocities[0],
                )
                selected_index = int(filter_info.get("selected_index", 0))
                if selected_index not in (0, 1):
                    raise ValueError("filter selected an invalid candidate")
                executed_actions = candidate_actions[selected_index].copy()

        csp_actions = executed_actions.copy()
        cbf_info = _disabled_cbf_info()
        if (
            self.cbf_filter.enabled
            and self.step_index % self.cbf_interval == 0
        ):
            braking_actions = self.adapter.braking_actions()
            nominal_world_velocities = self.adapter.preview_world_velocities(
                csp_actions
            )
            fallback_world_velocities = self.adapter.preview_world_velocities(
                braking_actions
            )
            safe_world_velocities, cbf_info = self.cbf_filter.filter(
                nominal_world_velocities,
                self.cached_robot_states,
                self.cached_human_states,
                self.lmte_outputs,
                clearance_context=self._clearance_context(),
                fallback_action=fallback_world_velocities,
            )
            executed_actions = self.adapter.world_velocities_to_controls(
                safe_world_velocities
            )

        observation, base_rewards, done, base_info = self.env.step(
            executed_actions
        )

        next_robot_states = self.adapter.robot_states()
        next_human_states = self.adapter.human_states()
        previous_outputs = self.lmte_outputs
        self._update_lmte(next_human_states)
        try:
            rewards, reward_info = self.reward_wrapper.compute(
                base_rewards,
                active_at_start,
                next_robot_states,
                next_human_states,
                self.lmte_outputs,
            )
        except Exception:
            self.lmte_outputs = previous_outputs
            raise

        self.cached_robot_states = next_robot_states
        self.cached_human_states = next_human_states
        self.step_index += 1

        uncertainties = [
            float(output.get("uncertainty", 0.0))
            for output in self.lmte_outputs.values()
        ]
        info = dict(base_info or {})
        info.update(reward_info)
        info.update(cbf_info)
        info.update(
            {
                "lmte_mean_uncertainty": (
                    float(np.mean(uncertainties))
                    if uncertainties
                    else 0.0
                ),
                "csp_action_filter_used": int(
                    filter_info.get("csp_action_filter_used", 0)
                ),
                "csp_action_filtered": int(
                    filter_info.get(
                        "filtered_by_csp_action_filter", 0
                    )
                ),
                "raw_policy_action": raw_actions.copy(),
                "csp_action": csp_actions.copy(),
                "safe_action": executed_actions.copy(),
                "executed_action": executed_actions.copy(),
            }
        )
        return observation, rewards, done, info