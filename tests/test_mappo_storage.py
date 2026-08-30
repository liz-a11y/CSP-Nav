import gym
import numpy as np
import torch

from training.networks.mappo_storage import MAPPORolloutStorage


def make_spaces():
    return gym.spaces.Dict(
        {
            "robot_node": gym.spaces.Box(
                low=-1,
                high=1,
                shape=(3, 1, 5),
                dtype=np.float32,
            ),
            "global_robot_states": gym.spaces.Box(
                low=-1,
                high=1,
                shape=(3, 10),
                dtype=np.float32,
            ),
        }
    )


def make_storage(num_steps=2, num_envs=2):
    return MAPPORolloutStorage(
        num_steps=num_steps,
        num_envs=num_envs,
        agent_num=3,
        observation_space=make_spaces(),
        action_shape=1,
        actor_rnn_size=4,
        critic_rnn_size=5,
    )


def test_storage_preserves_time_env_agent_axes_and_after_update():
    storage = make_storage()
    assert storage.obs["robot_node"].shape == (3, 2, 3, 1, 5)
    assert storage.actions.shape == (2, 2, 3, 1)
    assert storage.actor_rnn_states.shape == (3, 2, 3, 1, 4)
    assert storage.critic_rnn_states.shape == (3, 2, 3, 1, 5)

    storage.obs["robot_node"][-1].fill_(7.0)
    storage.rnn_masks[-1].fill_(0.0)
    storage.after_update()

    assert torch.all(storage.obs["robot_node"][0] == 7.0)
    assert torch.all(storage.rnn_masks[0] == 0.0)


def test_insert_writes_one_joint_environment_step():
    storage = make_storage()
    observation = {
        "robot_node": torch.ones(2, 3, 1, 5),
        "global_robot_states": torch.ones(2, 3, 10),
    }
    storage.insert(
        observation=observation,
        actor_rnn_states=torch.ones(2, 3, 1, 4),
        critic_rnn_states=torch.ones(2, 3, 1, 5),
        actions=torch.ones(2, 3, 1, dtype=torch.long),
        action_log_probs=torch.ones(2, 3, 1),
        value_preds=torch.ones(2, 3, 1),
        rewards=torch.ones(2, 3, 1),
        rnn_masks=torch.ones(2, 3, 1),
        active_masks=torch.ones(2, 3, 1),
        bad_masks=torch.ones(2, 3, 1),
    )

    assert storage.step == 1
    assert torch.all(storage.obs["robot_node"][1] == 1.0)
    assert torch.all(storage.actions[0] == 1)


def test_gae_stops_at_individual_agent_terminal():
    storage = make_storage(num_steps=2, num_envs=1)
    storage.rewards[:, 0, :, 0] = torch.tensor(
        [[1.0, 1.0, 1.0], [100.0, 2.0, 2.0]]
    )
    storage.value_preds.zero_()
    storage.rnn_masks.fill_(1.0)
    storage.rnn_masks[1, 0, 0, 0] = 0.0

    storage.compute_returns(
        next_value=torch.zeros(1, 3, 1),
        gamma=1.0,
        gae_lambda=1.0,
        use_gae=True,
        use_proper_time_limits=False,
    )

    assert storage.returns[0, 0, 0, 0] == 1.0
    assert storage.returns[0, 0, 1, 0] == 3.0


def test_masked_advantage_normalization_excludes_inactive_values():
    advantages = torch.tensor([[[[1.0], [3.0], [1000.0]]]])
    active_masks = torch.tensor([[[[1.0], [1.0], [0.0]]]])

    normalized = MAPPORolloutStorage.normalize_advantages(
        advantages,
        active_masks,
    )

    torch.testing.assert_close(normalized[0, 0, :2, 0], torch.tensor([-1.0, 1.0]))
    assert normalized[0, 0, 2, 0] == 0.0


def test_recurrent_generator_keeps_all_agents_from_selected_environments():
    storage = make_storage(num_steps=2, num_envs=4)
    for env_id in range(4):
        storage.obs["robot_node"][:-1, env_id].fill_(float(env_id))
    advantages = torch.zeros(2, 4, 3, 1)

    batches = list(storage.recurrent_generator(advantages, num_mini_batch=2))

    assert len(batches) == 2
    for batch in batches:
        robot_nodes = batch["obs"]["robot_node"]
        assert robot_nodes.shape[2] == 3
        for mini_env_id in range(robot_nodes.shape[1]):
            values = robot_nodes[:, mini_env_id, :, 0, 0]
            assert torch.unique(values).numel() == 1
