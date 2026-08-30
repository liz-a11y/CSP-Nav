import copy
import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from training.networks.selfAttn_srnn_merge_lidar import (
    selfAttn_merge_SRNN_lidar,
)


LOCAL_HEIGHT_KEYS = (
    "robot_node",
    "spatial_edges",
    "detected_human_num",
    "point_clouds",
)


def _layout(observations):
    robot_node = observations["robot_node"]
    if robot_node.dim() == 4:
        return False, 1, robot_node.size(0), robot_node.size(1)
    if robot_node.dim() == 5:
        return True, robot_node.size(0), robot_node.size(1), robot_node.size(2)
    raise ValueError("robot_node must have [N,A,...] or [T,N,A,...] dimensions")


def _single_agent_spaces(observation_space):
    spaces = observation_space.spaces
    return {
        key: SimpleNamespace(shape=spaces[key].shape[1:])
        for key in LOCAL_HEIGHT_KEYS
    }


class OriginalHEIGHTAdapter(nn.Module):
    """Adapts MAPPO's explicit agent axis to the original HEIGHT backbone."""

    def __init__(self, observation_space, config):
        super().__init__()
        backbone_config = copy.deepcopy(config)
        backbone_config.env.env_name = "CrowdSim3DTbObs-v0"
        backbone_config.training.num_processes = 1
        backbone_config.ppo.num_mini_batch = 1
        self.backbone = selfAttn_merge_SRNN_lidar(
            _single_agent_spaces(observation_space),
            backbone_config,
        )
        self.rnn_size = backbone_config.SRNN.human_node_rnn_size
        self.output_size = backbone_config.SRNN.human_node_output_size
        self.lidar_input_size = int(360.0 / backbone_config.lidar.angular_res)

        for parameter in self.backbone.critic.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.critic_linear.parameters():
            parameter.requires_grad = False

    def _flatten_observations(
        self,
        observations,
        sequence,
        time_steps,
        envs,
        agents,
    ):
        flattened = {}
        for key in LOCAL_HEIGHT_KEYS:
            value = observations[key]
            trailing = value.shape[3:] if sequence else value.shape[2:]
            flattened[key] = value.reshape(
                time_steps * envs * agents,
                *trailing,
            )

        lidar = flattened["point_clouds"]
        expected_lidar_shape = (
            time_steps * envs * agents,
            self.backbone.lidar_channel_num,
            self.lidar_input_size,
        )
        if tuple(lidar.shape) != expected_lidar_shape:
            raise ValueError(
                "point_clouds must have shape {}, got {}".format(
                    expected_lidar_shape,
                    tuple(lidar.shape),
                )
            )

        max_humans = flattened["spatial_edges"].size(-2)
        flattened["detected_human_num"] = flattened[
            "detected_human_num"
        ].clamp(min=1, max=max_humans)
        return flattened

    def forward(self, observations, rnn_states, masks):
        sequence, time_steps, envs, agents = _layout(observations)
        flattened = self._flatten_observations(
            observations,
            sequence,
            time_steps,
            envs,
            agents,
        )
        flattened_states = rnn_states.reshape(
            envs * agents,
            1,
            self.rnn_size,
        )
        flattened_masks = masks.reshape(
            time_steps * envs * agents,
            1,
        )

        self.backbone.seq_length = time_steps
        self.backbone.nenv = envs * agents
        self.backbone.nminibatch = 1
        self.backbone.config.training.cuda = observations["robot_node"].is_cuda
        _, features, next_states = self.backbone(
            flattened,
            {"rnn": flattened_states},
            flattened_masks,
            infer=not sequence,
        )

        if sequence:
            features = features.reshape(
                time_steps,
                envs,
                agents,
                self.output_size,
            )
        else:
            features = features.reshape(envs, agents, self.output_size)
        next_states = next_states["rnn"].reshape(
            envs,
            agents,
            1,
            self.rnn_size,
        )
        return features, next_states


class MaskedPeerAttention(nn.Module):
    def __init__(self, query_size, edge_size, output_size):
        super().__init__()
        self.query = nn.Linear(query_size, output_size)
        self.key = nn.Linear(edge_size, output_size)
        self.value = nn.Linear(edge_size, output_size)
        self.output_size = output_size

    def forward(self, query_features, edges, counts):
        leading_shape = query_features.shape[:-1]
        flat_query = query_features.reshape(-1, query_features.size(-1))
        flat_edges = edges.reshape(-1, edges.size(-2), edges.size(-1))
        flat_counts = counts.reshape(-1).long().clamp(
            min=0,
            max=flat_edges.size(1),
        )

        query = self.query(flat_query).unsqueeze(1)
        keys = self.key(flat_edges)
        values = self.value(flat_edges)
        scores = (query * keys).sum(-1) / math.sqrt(self.output_size)
        indices = torch.arange(
            flat_edges.size(1),
            device=flat_edges.device,
        ).unsqueeze(0)
        valid = indices < flat_counts.unsqueeze(1)
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
        pooled = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        pooled = pooled * (flat_counts > 0).to(pooled.dtype).unsqueeze(1)
        return pooled.reshape(*leading_shape, self.output_size)


class SharedHEIGHTActor(nn.Module):
    def __init__(self, observation_space, config):
        super().__init__()
        self.agent_num = config.mappo.robot_num
        self.height = OriginalHEIGHTAdapter(observation_space, config)
        self.rnn_size = self.height.rnn_size
        self.output_size = self.height.output_size
        self.peer_attention = MaskedPeerAttention(
            query_size=self.output_size,
            edge_size=observation_space.spaces["robot_robot_edges"].shape[-1],
            output_size=self.output_size,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.output_size * 2, self.output_size),
            nn.Tanh(),
        )

    def forward(self, observations, rnn_states, masks):
        _, _, _, agents = _layout(observations)
        if agents != self.agent_num:
            raise ValueError("actor received an unexpected agent count")
        local_features, next_states = self.height(
            observations,
            rnn_states,
            masks,
        )
        peer_features = self.peer_attention(
            local_features,
            observations["robot_robot_edges"],
            observations["detected_robot_num"],
        )
        return (
            self.fusion(torch.cat([local_features, peer_features], dim=-1)),
            next_states,
        )


class CentralizedHEIGHTCritic(nn.Module):
    def __init__(self, observation_space, config):
        super().__init__()
        self.agent_num = config.mappo.robot_num
        self.height = OriginalHEIGHTAdapter(observation_space, config)
        self.rnn_size = self.height.rnn_size
        self.local_size = self.height.output_size
        self.agent_attention = nn.MultiheadAttention(
            self.local_size,
            config.mappo.critic_attention_heads,
            batch_first=True,
        )
        global_size = observation_space.spaces["global_robot_states"].shape[-1]
        self.central_encoder = nn.Sequential(
            nn.Linear(
                self.local_size * 2 + global_size * self.agent_num,
                config.mappo.critic_hidden_size,
            ),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(config.mappo.critic_hidden_size, 1)

    def forward(self, observations, rnn_states, masks):
        sequence, time_steps, envs, agents = _layout(observations)
        if agents != self.agent_num:
            raise ValueError("critic received an unexpected agent count")
        local_features, next_states = self.height(
            observations,
            rnn_states,
            masks,
        )
        flat_local = local_features.reshape(
            time_steps * envs,
            agents,
            self.local_size,
        )

        active = observations.get("agent_active_mask")
        key_padding_mask = None
        if active is not None:
            flat_active = active.reshape(time_steps * envs, agents) > 0
            key_padding_mask = ~flat_active
            all_inactive = key_padding_mask.all(dim=1)
            key_padding_mask[all_inactive] = False
        context, _ = self.agent_attention(
            flat_local,
            flat_local,
            flat_local,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        global_states = observations["global_robot_states"]
        flat_global = global_states.reshape(
            time_steps * envs,
            agents * global_states.size(-1),
        )
        flat_global = flat_global.unsqueeze(1).expand(-1, agents, -1)
        encoded = self.central_encoder(
            torch.cat([flat_local, context, flat_global], dim=-1)
        )
        values = self.value_head(encoded)
        if sequence:
            values = values.reshape(time_steps, envs, agents, 1)
        else:
            values = values.reshape(envs, agents, 1)
        return values, next_states
