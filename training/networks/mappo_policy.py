import os

import torch
import torch.nn as nn

from training.networks.mappo_height import (
    CentralizedHEIGHTCritic,
    SharedHEIGHTActor,
)


class MAPPOPolicy(nn.Module):
    def __init__(self, observation_space, action_space, config):
        super().__init__()
        if action_space.__class__.__name__ != "MultiDiscrete":
            raise ValueError("MAPPOPolicy currently expects a MultiDiscrete action space")
        action_sizes = list(action_space.nvec)
        if len(set(int(size) for size in action_sizes)) != 1:
            raise ValueError("all shared-policy robots must use the same action count")

        self.actor = SharedHEIGHTActor(observation_space, config)
        self.critic = CentralizedHEIGHTCritic(observation_space, config)
        self.action_head = nn.Linear(self.actor.output_size, int(action_sizes[0]))
        self.actor_rnn_size = self.actor.rnn_size
        self.critic_rnn_size = self.critic.rnn_size

    def _distribution(self, actor_features):
        return torch.distributions.Categorical(logits=self.action_head(actor_features))

    def get_actions(
        self,
        observations,
        actor_rnn_states,
        critic_rnn_states,
        masks,
        deterministic=False,
    ):
        actor_features, next_actor_states = self.actor(
            observations,
            actor_rnn_states,
            masks,
        )
        values, next_critic_states = self.critic(
            observations,
            critic_rnn_states,
            masks,
        )
        distribution = self._distribution(actor_features)
        actions = (
            distribution.probs.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        action_log_probs = distribution.log_prob(actions).unsqueeze(-1)
        return (
            values,
            actions.unsqueeze(-1),
            action_log_probs,
            next_actor_states,
            next_critic_states,
        )

    def get_values(self, observations, critic_rnn_states, masks):
        return self.critic(observations, critic_rnn_states, masks)

    def evaluate_actions(
        self,
        observations,
        actor_rnn_states,
        critic_rnn_states,
        actions,
        masks,
    ):
        actor_features, _ = self.actor(
            observations,
            actor_rnn_states,
            masks,
        )
        values, _ = self.critic(
            observations,
            critic_rnn_states,
            masks,
        )
        distribution = self._distribution(actor_features)
        action_indices = actions.squeeze(-1)
        action_log_probs = distribution.log_prob(action_indices).unsqueeze(-1)
        entropy = distribution.entropy().unsqueeze(-1)
        return values, action_log_probs, entropy

    def actor_state_dict(self):
        state = {
            "backbone." + key: value
            for key, value in self.actor.state_dict().items()
        }
        state.update(
            {
                "action_head." + key: value
                for key, value in self.action_head.state_dict().items()
            }
        )
        return state

    def load_actor_state_dict(self, state, strict=True):
        backbone = {
            key[len("backbone.") :]: value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
        action_head = {
            key[len("action_head.") :]: value
            for key, value in state.items()
            if key.startswith("action_head.")
        }
        self.actor.load_state_dict(backbone, strict=strict)
        self.action_head.load_state_dict(action_head, strict=strict)

    def load_single_robot_backbone(self, checkpoint, map_location="cpu"):
        if isinstance(checkpoint, (str, os.PathLike)):
            source = torch.load(checkpoint, map_location=map_location)
        else:
            source = checkpoint
        if "state_dict" in source and isinstance(source["state_dict"], dict):
            source = source["state_dict"]

        normalized_source = {}
        for key, value in source.items():
            while key.startswith("module."):
                key = key[len("module.") :]
            normalized_source[key] = value

        backbone_source = {
            key[len("base.") :]: value
            for key, value in normalized_source.items()
            if key.startswith("base.")
        }
        actor_state = self.actor.height.backbone.state_dict()
        critic_state = self.critic.height.backbone.state_dict()
        actor_loaded = {}
        critic_loaded = {}
        mismatched = {}
        for key, value in backbone_source.items():
            if key in actor_state and actor_state[key].shape == value.shape:
                actor_loaded[key] = value
            elif key in actor_state:
                mismatched["actor." + key] = {
                    "checkpoint": tuple(value.shape),
                    "model": tuple(actor_state[key].shape),
                }
            if key in critic_state and critic_state[key].shape == value.shape:
                critic_loaded[key] = value
            elif key in critic_state:
                mismatched["critic." + key] = {
                    "checkpoint": tuple(value.shape),
                    "model": tuple(critic_state[key].shape),
                }

        actor_state.update(actor_loaded)
        critic_state.update(critic_loaded)
        self.actor.height.backbone.load_state_dict(actor_state)
        self.critic.height.backbone.load_state_dict(critic_state)

        action_head_loaded = []
        action_mapping = {
            "weight": "dist.linear.weight",
            "bias": "dist.linear.bias",
        }
        action_state = self.action_head.state_dict()
        for destination, source_key in action_mapping.items():
            if source_key not in normalized_source:
                continue
            value = normalized_source[source_key]
            if action_state[destination].shape == value.shape:
                action_state[destination] = value
                action_head_loaded.append(destination)
            else:
                mismatched["action_head." + destination] = {
                    "checkpoint": tuple(value.shape),
                    "model": tuple(action_state[destination].shape),
                }
        self.action_head.load_state_dict(action_state)

        return {
            "actor_loaded": sorted(actor_loaded),
            "critic_loaded": sorted(critic_loaded),
            "action_head_loaded": sorted(action_head_loaded),
            "mismatched": mismatched,
            "actor_missing": sorted(set(actor_state) - set(actor_loaded)),
            "critic_missing": sorted(set(critic_state) - set(critic_loaded)),
        }
