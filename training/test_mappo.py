import argparse
import importlib.util
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from crowd_sim import *  # noqa: F401,F403 - imports Gym registrations
from training.networks.mappo_policy import MAPPOPolicy
from training.networks.multi_agent_envs import make_multi_agent_vec_envs


DEFAULT_MODEL_DIR = None
DEFAULT_TEST_MODEL = None
LOGGER_NAME = "test_mappo"


def classify_episode_status(terminal_reason):
    """Return one team-level status label for a completed episode."""
    reasons = list(terminal_reason)
    has_collision = any(reason == "collided" for reason in reasons)
    has_timeout = any(reason == "timeout" for reason in reasons)

    if reasons and all(reason == "reached" for reason in reasons):
        return "success"
    if has_collision and has_timeout:
        return "failure_collision_timeout"
    if has_collision:
        return "failure_collision"
    if has_timeout:
        return "timeout"
    return "failure"


def summarize_episode(terminal_reason, collision_with):
    successes = [int(reason == "reached") for reason in terminal_reason]
    collided = [reason == "collided" for reason in terminal_reason]
    timed_out = [reason == "timeout" for reason in terminal_reason]
    return {
        "successes": successes,
        "team_success": int(all(successes)),
        "collision_rate": int(any(collided)),
        "timeout_rate": int(any(timed_out)),
        "collision_human": sum(
            reason == "collided" and target == "human"
            for reason, target in zip(terminal_reason, collision_with)
        ),
        "collision_robot": sum(
            reason == "collided" and target == "robot"
            for reason, target in zip(terminal_reason, collision_with)
        ),
        "collision_obstacle": sum(
            reason == "collided" and target == "obstacle"
            for reason, target in zip(terminal_reason, collision_with)
        ),
        "collision_wall": sum(
            reason == "collided" and target == "wall"
            for reason, target in zip(terminal_reason, collision_with)
        ),
        "timeouts": sum(timed_out),
    }


def accumulate_social_step(metrics, info):
    metrics["social_steps"] += 1
    metrics["CSP_scene_CVaR_sum"] += float(
        info.get("CSP_scene_CVaR", 0.0)
    )
    metrics["P_reach_sum"] += float(info.get("mean_P_reach", 0.0))
    metrics["r_csp_sum"] += float(info.get("r_csp", 0.0))
    metrics["csp_action_filtered_count"] += int(
        info.get("csp_action_filtered", 0)
    )


def infer_model_dir_from_checkpoint(checkpoint_path):
    if checkpoint_path is None:
        return None
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(checkpoint_dir) == "checkpoints":
        return os.path.dirname(checkpoint_dir)
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a 3-robot MAPPO checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Full checkpoint path. If omitted, provide --model_dir and "
            "--test_model."
        ),
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help=(
            "Model directory. Required unless it can be inferred from a "
            "checkpoint located under a checkpoints folder."
        ),
    )
    parser.add_argument(
        "--test_model",
        type=str,
        default=None,
        help="Checkpoint filename inside model_dir/checkpoints.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true",
        help=(
            "Render the test environment. Keep disabled for reliable "
            "benchmark numbers."
        ),
    )
    parser.add_argument(
        "--test_case",
        "--test-case",
        dest="test_case",
        type=int,
        default=-1,
        help=(
            "Fixed test case id; -1 evaluates different default test cases."
        ),
    )
    args = parser.parse_args(argv)
    if args.model_dir is None:
        args.model_dir = infer_model_dir_from_checkpoint(args.checkpoint)
    if args.model_dir is None:
        parser.error(
            "provide --model_dir, or a --checkpoint located under a "
            "model_dir/checkpoints directory"
        )
    if args.checkpoint is None and args.test_model is None:
        parser.error("provide --checkpoint or --test_model")
    return args


def resolve_checkpoint_path(args):
    checkpoint_path = getattr(args, "checkpoint", None)
    if checkpoint_path:
        return checkpoint_path
    return os.path.join(
        getattr(args, "model_dir", DEFAULT_MODEL_DIR),
        "checkpoints",
        getattr(args, "test_model", None) or DEFAULT_TEST_MODEL,
    )


def _load_config_file(path):
    base_path = os.path.join(os.path.dirname(path), "config.py")
    if os.path.basename(path) != "config.py" and os.path.isfile(base_path):
        base_spec = importlib.util.spec_from_file_location(
            "crowd_nav.configs.config", base_path
        )
        if base_spec is None or base_spec.loader is None:
            raise ImportError("cannot import config file {}".format(base_path))
        base_module = importlib.util.module_from_spec(base_spec)
        sys.modules["crowd_nav.configs.config"] = base_module
        base_spec.loader.exec_module(base_module)
    module_name = (
        "test_mappo_saved_config_"
        + str(abs(hash(os.path.abspath(path))))
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot import config file {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "Config")


def load_config_class(model_dir):
    config_dir = os.path.join(model_dir, "configs")
    candidates = (
        os.path.join(config_dir, "config_mappo.py"),
        os.path.join(config_dir, "config.py"),
    )
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            return _load_config_file(candidate), candidate
        except Exception:
            continue
    raise FileNotFoundError(
        "No private configuration snapshot was found under {!r}".format(
            config_dir
        )
    )


def configure_logging(model_dir, checkpoint_path, test_model, visualize):
    log_dir = os.path.join(model_dir, "test")
    os.makedirs(log_dir, exist_ok=True)
    if visualize:
        log_name = "test_visual.log"
    else:
        model_name = test_model or os.path.basename(checkpoint_path)
        log_name = "test_{}.log".format(model_name)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger, log_path


def render_first_env(envs):
    venv = getattr(envs, "venv", None)
    env_list = getattr(venv, "envs", None)
    if env_list:
        env_list[0].render()


def evaluate(args=None):
    args = parse_args() if args is None else args
    if getattr(args, "model_dir", None) is None:
        args.model_dir = infer_model_dir_from_checkpoint(
            getattr(args, "checkpoint", None)
        )
    if args.model_dir is None:
        raise ValueError(
            "model_dir is required when checkpoint is not under checkpoints/"
        )
    checkpoint_path = resolve_checkpoint_path(args)
    config_class, config_source = load_config_class(args.model_dir)
    config = config_class()
    config.training.num_processes = 1
    config.sim.render = bool(getattr(args, "visualize", False))
    if getattr(args, "cpu", False):
        config.training.cuda = False
    device = torch.device(
        "cuda"
        if config.training.cuda and torch.cuda.is_available()
        else "cpu"
    )
    logger, log_path = configure_logging(
        args.model_dir,
        checkpoint_path,
        getattr(args, "test_model", None),
        bool(getattr(args, "visualize", False)),
    )
    logger.info("checkpoint: %s", checkpoint_path)
    logger.info("config source: %s", config_source)
    logger.info("log file: %s", log_path)
    logger.info("device: %s", device)
    envs = make_multi_agent_vec_envs(
        config.env.env_name,
        config.env.seed,
        1,
        device,
        config,
        phase="test",
        test_case=int(getattr(args, "test_case", -1)),
    )
    policy = MAPPOPolicy(
        envs.observation_space,
        envs.action_space,
        config,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_actor_state_dict(checkpoint["actor"])
    policy.critic.load_state_dict(checkpoint["critic"])
    policy.eval()

    actor_states = torch.zeros(
        1,
        config.mappo.robot_num,
        1,
        policy.actor_rnn_size,
        device=device,
    )
    critic_states = torch.zeros(
        1,
        config.mappo.robot_num,
        1,
        policy.critic_rnn_size,
        device=device,
    )
    masks = torch.zeros(1, config.mappo.robot_num, 1, device=device)
    observation = envs.reset()
    metrics = defaultdict(float)
    success_per_robot = np.zeros(config.mappo.robot_num, dtype=np.float64)
    base_env = envs.venv.envs[0]
    arena_scale = max(float(base_env.arena_size), 1.0)

    try:
        for episode_index in range(1, args.episodes + 1):
            done = False
            episode_rewards = np.zeros(config.mappo.robot_num, dtype=np.float64)
            path_lengths = np.zeros(config.mappo.robot_num, dtype=np.float64)
            last_positions = (
                observation["global_robot_states"][0, :, :2].cpu().numpy()
                * arena_scale
            )
            final_info = {
                "terminal_reason": ["timeout"] * config.mappo.robot_num,
                "collision_with": ["none"] * config.mappo.robot_num,
            }
            while not done:
                with torch.no_grad():
                    (
                        _,
                        actions,
                        _,
                        actor_states,
                        critic_states,
                    ) = policy.get_actions(
                        observation,
                        actor_states,
                        critic_states,
                        masks,
                        deterministic=True,
                    )
                observation, rewards, env_dones, infos = envs.step(actions)
                done = bool(env_dones[0])
                final_info = infos[0]
                accumulate_social_step(metrics, final_info)
                episode_rewards += rewards[0, :, 0].cpu().numpy()
                if done and "robot_positions" in final_info:
                    positions = np.asarray(
                        final_info["robot_positions"],
                        dtype=np.float32,
                    )
                else:
                    positions = (
                        observation["global_robot_states"][0, :, :2].cpu().numpy()
                        * arena_scale
                    )
                path_lengths += np.linalg.norm(positions - last_positions, axis=1)
                last_positions = positions
                masks = torch.as_tensor(
                    np.asarray(final_info["rnn_masks"])[None],
                    dtype=torch.float32,
                    device=device,
                )
                if getattr(args, "visualize", False):
                    render_first_env(envs)

            summary = summarize_episode(
                final_info["terminal_reason"],
                final_info["collision_with"],
            )
            episode_status = classify_episode_status(
                final_info["terminal_reason"]
            )
            logger.info(
                "episode %d/%d | status=%s | team_success=%d | "
                "terminal_reason=%s | collision_with=%s | "
                "reward=%.3f | path_length=%.3f",
                episode_index,
                args.episodes,
                episode_status,
                summary["team_success"],
                final_info["terminal_reason"],
                final_info["collision_with"],
                episode_rewards.mean(),
                path_lengths.mean(),
            )
            success_per_robot += summary.pop("successes")
            for key, value in summary.items():
                metrics[key] += value
            metrics["reward"] += episode_rewards.mean()
            metrics["path_length"] += path_lengths.mean()
            human_distances = np.asarray(
                final_info.get("minimum_human_distance", []),
                dtype=np.float32,
            )
            robot_distances = np.asarray(
                final_info.get("minimum_robot_distance", []),
                dtype=np.float32,
            )
            obstacle_distances = np.asarray(
                final_info.get("minimum_obstacle_distance", []),
                dtype=np.float32,
            )
            finite_human = human_distances[np.isfinite(human_distances)]
            finite_robot = robot_distances[np.isfinite(robot_distances)]
            finite_obstacle = obstacle_distances[np.isfinite(obstacle_distances)]
            if finite_human.size:
                metrics["minimum_human_distance"] += finite_human.min()
            if finite_robot.size:
                metrics["minimum_robot_distance"] += finite_robot.min()
            if finite_obstacle.size:
                metrics["minimum_obstacle_distance"] += finite_obstacle.min()
            actor_states.zero_()
            critic_states.zero_()
            masks.zero_()
    finally:
        envs.close()

    denominator = float(args.episodes)
    social_steps = max(float(metrics.pop("social_steps", 0.0)), 1.0)
    mean_csp = metrics.pop("CSP_scene_CVaR_sum", 0.0) / social_steps
    mean_reach = metrics.pop("P_reach_sum", 0.0) / social_steps
    mean_r_csp = metrics.pop("r_csp_sum", 0.0) / social_steps
    filter_rate = (
        metrics.pop("csp_action_filtered_count", 0.0) / social_steps
    )
    result = {
        "success_rate_per_robot": (success_per_robot / denominator).tolist(),
        **{key: value / denominator for key, value in metrics.items()},
        "mean_CSP_scene_CVaR": mean_csp,
        "mean_P_reach": mean_reach,
        "mean_r_csp": mean_r_csp,
        "csp_filter_intervention_rate": filter_rate,
    }
    for key, value in result.items():
        logger.info("%s: %s", key, value)
    return result


if __name__ == "__main__":
    evaluate()
