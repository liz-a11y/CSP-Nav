import argparse
import csv
import importlib
import os
import shutil
import time
from collections import deque

import numpy as np
import torch

from crowd_sim import *  # noqa: F401,F403 - imports Gym registrations
from training.algo.mappo import MAPPO
from training.networks.mappo_policy import MAPPOPolicy
from training.networks.mappo_storage import MAPPORolloutStorage
from training.networks.multi_agent_envs import make_multi_agent_vec_envs


DEFAULT_CONFIG_MODULE = "crowd_nav.configs.config_mappo"


def load_config_module(module_name):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Private training configuration is not included. Place config.py "
            "and config_mappo.py under crowd_nav/configs, or provide an "
            "importable module with --config-module."
        ) from exc
    if not hasattr(module, "Config"):
        raise AttributeError(
            "Configuration module {!r} does not define Config".format(
                module_name
            )
        )
    return module


class EpisodeRewardTracker:
    def __init__(self, num_envs, window_size=100):
        self.current_rewards = np.zeros(num_envs, dtype=np.float64)
        self.completed_rewards = deque(maxlen=window_size)

    def update(self, rewards, env_dones):
        if torch.is_tensor(rewards):
            rewards = rewards.detach().cpu().numpy()
        rewards = np.asarray(rewards, dtype=np.float64)
        env_dones = np.asarray(env_dones, dtype=bool)
        per_environment = rewards.reshape(rewards.shape[0], -1).mean(axis=1)
        self.current_rewards += per_environment
        for env_id, done in enumerate(env_dones):
            if done:
                self.completed_rewards.append(float(self.current_rewards[env_id]))
                self.current_rewards[env_id] = 0.0

    def summary(self):
        values = np.asarray(self.completed_rewards, dtype=np.float64)
        if values.size == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }


def extract_agent_masks(infos, env_dones, agent_num, device):
    rnn_masks = []
    active_masks = []
    bad_masks = []
    for info, env_done in zip(infos, env_dones):
        rnn_masks.append(
            np.asarray(
                info.get("rnn_masks", np.ones((agent_num, 1))),
                dtype=np.float32,
            )
        )
        next_active = np.asarray(
            info.get("active_masks", np.ones((agent_num, 1))),
            dtype=np.float32,
        )
        if env_done:
            next_active = np.ones((agent_num, 1), dtype=np.float32)
        active_masks.append(next_active)
        bad_masks.append(
            np.asarray(
                info.get("bad_masks", np.ones((agent_num, 1))),
                dtype=np.float32,
            )
        )
    return (
        torch.as_tensor(np.stack(rnn_masks), device=device),
        torch.as_tensor(np.stack(active_masks), device=device),
        torch.as_tensor(np.stack(bad_masks), device=device),
    )


def count_training_steps(update_index, num_processes, num_steps, agent_num):
    physical_steps = (update_index + 1) * num_processes * num_steps
    return physical_steps, physical_steps * agent_num


def resolve_update_range(start_update, requested_updates, default_updates):
    if requested_updates is None:
        return start_update, default_updates
    if requested_updates < 0:
        raise ValueError("requested updates must be non-negative")
    return start_update, start_update + requested_updates


def should_save_checkpoint(update, end_update, save_interval):
    return update % save_interval == 0 or update == end_update - 1


def aggregate_rollout_metrics(infos):
    individual_rewards = []
    mixed_rewards = []
    team_rewards = []
    active_masks = []
    scalar_series = {
        "r_csp": [],
        "CSP_scene_CVaR": [],
        "mean_P_reach": [],
        "lmte_mean_uncertainty": [],
        "csp_action_filter_used": [],
        "csp_action_filtered": [],
        "cbf_used": [],
        "cbf_intervened": [],
        "min_h_before": [],
        "min_h_after": [],
        "cbf_condition": [],
        "qp_success": [],
        "qp_infeasible": [],
        "fallback_used": [],
        "action_deviation": [],
    }
    distance_series = {
        "minimum_human_distance": [],
        "minimum_robot_distance": [],
        "minimum_obstacle_distance": [],
    }
    counts = {
        "success_count": 0,
        "collision_human_count": 0,
        "collision_robot_count": 0,
        "collision_obstacle_count": 0,
        "collision_wall_count": 0,
        "timeout_count": 0,
    }

    def append_finite(target, values):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = array[np.isfinite(array)]
        target.extend(finite.tolist())

    for info in infos:
        individual_rewards.extend(
            np.asarray(info.get("individual_rewards", []), dtype=np.float64).reshape(-1)
        )
        mixed_rewards.extend(
            np.asarray(info.get("mixed_rewards", []), dtype=np.float64).reshape(-1)
        )
        if "team_reward" in info:
            team_rewards.append(float(info["team_reward"]))
        active_masks.extend(
            np.asarray(info.get("active_masks", []), dtype=np.float64).reshape(-1)
        )
        for key, values in scalar_series.items():
            if key in info:
                append_finite(values, [info[key]])
        for key, values in distance_series.items():
            if key in info:
                append_finite(values, info[key])

        reasons = list(info.get("terminal_reason", []))
        collision_with = list(info.get("collision_with", ["none"] * len(reasons)))
        agent_dones = info.get("agent_dones")
        if agent_dones is None:
            terminal_mask = np.ones(len(reasons), dtype=bool)
        else:
            terminal_mask = np.asarray(agent_dones, dtype=bool).reshape(-1)
        for agent_id, reason in enumerate(reasons):
            if agent_id >= len(terminal_mask) or not terminal_mask[agent_id]:
                continue
            if reason == "reached":
                counts["success_count"] += 1
            elif reason == "timeout":
                counts["timeout_count"] += 1
            elif reason == "collided":
                target = (
                    collision_with[agent_id]
                    if agent_id < len(collision_with)
                    else "none"
                )
                key = "collision_{}_count".format(target)
                if key in counts:
                    counts[key] += 1

    def mean_or_zero(values):
        return float(np.mean(values)) if values else 0.0

    metrics = {
        "individual_reward_mean": mean_or_zero(individual_rewards),
        "mixed_reward_mean": mean_or_zero(mixed_rewards),
        "team_reward_mean": mean_or_zero(team_rewards),
        "active_agent_fraction": mean_or_zero(active_masks),
        "r_csp_mean": mean_or_zero(scalar_series["r_csp"]),
        "CSP_scene_CVaR_mean": mean_or_zero(scalar_series["CSP_scene_CVaR"]),
        "mean_P_reach_mean": mean_or_zero(scalar_series["mean_P_reach"]),
        "lmte_mean_uncertainty_mean": mean_or_zero(
            scalar_series["lmte_mean_uncertainty"]
        ),
        "csp_filter_use_rate": mean_or_zero(
            scalar_series["csp_action_filter_used"]
        ),
        "csp_filter_intervention_rate": mean_or_zero(
            scalar_series["csp_action_filtered"]
        ),
        "cbf_use_rate": mean_or_zero(scalar_series["cbf_used"]),
        "cbf_intervention_rate": mean_or_zero(
            scalar_series["cbf_intervened"]
        ),
        "cbf_min_h_before_mean": mean_or_zero(
            scalar_series["min_h_before"]
        ),
        "cbf_min_h_after_mean": mean_or_zero(
            scalar_series["min_h_after"]
        ),
        "cbf_condition_mean": mean_or_zero(
            scalar_series["cbf_condition"]
        ),
        "cbf_qp_success_rate": mean_or_zero(scalar_series["qp_success"]),
        "cbf_qp_infeasible_rate": mean_or_zero(
            scalar_series["qp_infeasible"]
        ),
        "cbf_fallback_rate": mean_or_zero(scalar_series["fallback_used"]),
        "cbf_action_deviation_mean": mean_or_zero(
            scalar_series["action_deviation"]
        ),
        "minimum_human_distance_mean": mean_or_zero(
            distance_series["minimum_human_distance"]
        ),
        "minimum_robot_distance_mean": mean_or_zero(
            distance_series["minimum_robot_distance"]
        ),
        "minimum_obstacle_distance_mean": mean_or_zero(
            distance_series["minimum_obstacle_distance"]
        ),
    }
    metrics.update(counts)
    return metrics


def build_progress_row(
    update,
    physical_steps,
    agent_steps,
    fps,
    reward_summary,
    optimizer_metrics,
    rollout_metrics,
):
    row = {
        "misc/nupdates": update,
        "misc/total_timesteps": physical_steps,
        "fps": fps,
        "eprewmean": reward_summary["mean"],
        "loss/policy_entropy": optimizer_metrics["entropy"],
        "loss/policy_loss": optimizer_metrics["actor_loss"],
        "loss/value_loss": optimizer_metrics["critic_loss"],
        "misc/agent_steps": agent_steps,
        "episode/reward_median": reward_summary["median"],
        "episode/reward_min": reward_summary["min"],
        "episode/reward_max": reward_summary["max"],
        "loss/approx_kl": optimizer_metrics["approx_kl"],
        "loss/clip_fraction": optimizer_metrics["clip_fraction"],
        "loss/explained_variance": optimizer_metrics["explained_variance"],
    }
    row.update(
        {
            "rollout/{}".format(key): value
            for key, value in rollout_metrics.items()
        }
    )
    return row


def append_progress_row(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_training_configs(output_dir, config_module):
    config_dir = os.path.join(output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    module_path = os.path.abspath(config_module.__file__)
    source_dir = os.path.dirname(module_path)
    base_path = os.path.join(source_dir, "config.py")
    if os.path.isfile(base_path):
        shutil.copy2(base_path, os.path.join(config_dir, "config.py"))
    shutil.copy2(module_path, os.path.join(config_dir, "config_mappo.py"))


def build_checkpoint(
    actor_state,
    critic_state,
    actor_optimizer,
    critic_optimizer,
    update,
    elapsed_seconds,
    config_snapshot,
):
    return {
        "actor": actor_state,
        "critic": critic_state,
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "update": update,
        "elapsed_seconds": elapsed_seconds,
        "config": config_snapshot,
    }


def config_snapshot(config):
    return {
        "env_name": config.env.env_name,
        "seed": config.env.seed,
        "robot_num": config.mappo.robot_num,
        "num_processes": config.training.num_processes,
        "num_steps": config.ppo.num_steps,
        "num_mini_batch": config.ppo.num_mini_batch,
        "individual_reward_coef": config.mappo.individual_reward_coef,
        "team_reward_coef": config.mappo.team_reward_coef,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-module",
        default=DEFAULT_CONFIG_MODULE,
        help="Import path for the private module that defines Config.",
    )
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--num-mini-batch", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def train(args=None):
    args = parse_args() if args is None else args
    config_module = load_config_module(
        getattr(args, "config_module", DEFAULT_CONFIG_MODULE)
    )
    config = config_module.Config()
    if args.smoke:
        config.training.num_processes = 1
        config.ppo.num_steps = 8
        config.ppo.num_mini_batch = 1
        args.updates = 2 if args.updates is None else args.updates
    if args.num_processes is not None:
        config.training.num_processes = args.num_processes
    if args.num_steps is not None:
        config.ppo.num_steps = args.num_steps
    if args.num_mini_batch is not None:
        config.ppo.num_mini_batch = args.num_mini_batch
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    if args.cpu:
        config.training.cuda = False
    if config.training.num_processes % config.ppo.num_mini_batch != 0:
        raise ValueError("num_processes must be divisible by num_mini_batch")

    torch.manual_seed(config.env.seed)
    np.random.seed(config.env.seed)
    device = torch.device(
        "cuda"
        if config.training.cuda and torch.cuda.is_available()
        else "cpu"
    )
    envs = make_multi_agent_vec_envs(
        config.env.env_name,
        config.env.seed,
        config.training.num_processes,
        device,
        config,
        phase="train",
    )
    policy = MAPPOPolicy(
        envs.observation_space,
        envs.action_space,
        config,
    ).to(device)
    rollouts = MAPPORolloutStorage(
        num_steps=config.ppo.num_steps,
        num_envs=config.training.num_processes,
        agent_num=config.mappo.robot_num,
        observation_space=envs.observation_space,
        action_shape=1,
        actor_rnn_size=policy.actor_rnn_size,
        critic_rnn_size=policy.critic_rnn_size,
    ).to(device)
    trainer = MAPPO(
        policy=policy,
        clip_param=config.ppo.clip_param,
        ppo_epoch=config.ppo.epoch,
        num_mini_batch=config.ppo.num_mini_batch,
        value_loss_coef=config.ppo.value_loss_coef,
        entropy_coef=config.ppo.entropy_coef,
        actor_lr=config.mappo.actor_lr,
        critic_lr=config.mappo.critic_lr,
        eps=config.training.eps,
        max_grad_norm=config.training.max_grad_norm,
    )
    start_update = 0
    elapsed_before_resume = 0.0
    resume_path = getattr(args, "resume", None)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        policy.load_actor_state_dict(checkpoint["actor"])
        policy.critic.load_state_dict(checkpoint["critic"])
        trainer.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        start_update = int(checkpoint["update"]) + 1
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))

    observation = envs.reset()
    for key in rollouts.obs:
        rollouts.obs[key][0].copy_(observation[key])
    default_updates = (
        int(config.training.num_env_steps)
        // config.ppo.num_steps
        // config.training.num_processes
    )
    start_update, end_update = resolve_update_range(
        start_update,
        args.updates,
        default_updates,
    )
    os.makedirs(config.training.output_dir, exist_ok=True)
    save_training_configs(config.training.output_dir, config_module)
    checkpoint_dir = os.path.join(config.training.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    progress_path = os.path.join(config.training.output_dir, "progress.csv")
    episode_rewards = EpisodeRewardTracker(config.training.num_processes)
    start_time = time.time()

    try:
        for update in range(start_update, end_update):
            rollout_infos = []
            for step in range(config.ppo.num_steps):
                step_observation = {
                    key: rollouts.obs[key][step] for key in rollouts.obs
                }
                with torch.no_grad():
                    (
                        values,
                        actions,
                        action_log_probs,
                        actor_states,
                        critic_states,
                    ) = policy.get_actions(
                        step_observation,
                        rollouts.actor_rnn_states[step],
                        rollouts.critic_rnn_states[step],
                        rollouts.rnn_masks[step],
                    )
                next_observation, rewards, env_dones, infos = envs.step(actions)
                rnn_masks, active_masks, bad_masks = extract_agent_masks(
                    infos,
                    env_dones,
                    config.mappo.robot_num,
                    device,
                )
                rollout_infos.extend(infos)
                episode_rewards.update(rewards, env_dones)
                rollouts.insert(
                    observation=next_observation,
                    actor_rnn_states=actor_states,
                    critic_rnn_states=critic_states,
                    actions=actions,
                    action_log_probs=action_log_probs,
                    value_preds=values,
                    rewards=rewards,
                    rnn_masks=rnn_masks,
                    active_masks=active_masks,
                    bad_masks=bad_masks,
                )

            with torch.no_grad():
                next_values, _ = policy.get_values(
                    {key: rollouts.obs[key][-1] for key in rollouts.obs},
                    rollouts.critic_rnn_states[-1],
                    rollouts.rnn_masks[-1],
                )
            rollouts.compute_returns(
                next_values,
                gamma=config.reward.gamma,
                gae_lambda=config.ppo.gae_lambda,
                use_gae=config.ppo.use_gae,
                use_proper_time_limits=config.training.use_proper_time_limits,
            )
            metrics = trainer.update(rollouts)
            rollouts.after_update()

            physical_steps, agent_steps = count_training_steps(
                update,
                config.training.num_processes,
                config.ppo.num_steps,
                config.mappo.robot_num,
            )
            reward_summary = episode_rewards.summary()
            rollout_metrics = aggregate_rollout_metrics(rollout_infos)
            elapsed_seconds = (
                elapsed_before_resume + time.time() - start_time
            )
            fps = int(physical_steps / max(elapsed_seconds, 1e-8))
            if (
                (update % config.training.log_interval == 0 or args.smoke)
                and reward_summary["count"] > 1
            ):
                print(
                    "Updates {}, num timesteps {}, FPS {}\n"
                    " Last {} training episodes: mean/median reward "
                    "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    " Rollout: success {}, collisions H/R/O/W {}/{}/{}/{}, "
                    "timeout {}, active {:.2f}, r_csp {:.3f}, csp {:.3f}, "
                    "filter {:.2f}/{:.2f}, obs_dist {:.3f}\n".format(
                        update,
                        physical_steps,
                        fps,
                        reward_summary["count"],
                        reward_summary["mean"],
                        reward_summary["median"],
                        reward_summary["min"],
                        reward_summary["max"],
                        rollout_metrics["success_count"],
                        rollout_metrics["collision_human_count"],
                        rollout_metrics["collision_robot_count"],
                        rollout_metrics["collision_obstacle_count"],
                        rollout_metrics["collision_wall_count"],
                        rollout_metrics["timeout_count"],
                        rollout_metrics["active_agent_fraction"],
                        rollout_metrics["r_csp_mean"],
                        rollout_metrics["CSP_scene_CVaR_mean"],
                        rollout_metrics["csp_filter_use_rate"],
                        rollout_metrics["csp_filter_intervention_rate"],
                        rollout_metrics["minimum_obstacle_distance_mean"],
                    )
                )
                append_progress_row(
                    progress_path,
                    build_progress_row(
                        update,
                        physical_steps,
                        agent_steps,
                        fps,
                        reward_summary,
                        metrics,
                        rollout_metrics,
                    ),
                )
            if should_save_checkpoint(
                update,
                end_update,
                config.training.save_interval,
            ):
                checkpoint = build_checkpoint(
                    actor_state=policy.actor_state_dict(),
                    critic_state=policy.critic.state_dict(),
                    actor_optimizer=trainer.actor_optimizer,
                    critic_optimizer=trainer.critic_optimizer,
                    update=update,
                    elapsed_seconds=elapsed_seconds,
                    config_snapshot=config_snapshot(config),
                )
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, "{:05d}.pt".format(update)),
                )
    finally:
        envs.close()
    return policy


if __name__ == "__main__":
    train()
