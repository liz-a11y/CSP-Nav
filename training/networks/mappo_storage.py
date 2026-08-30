import torch


class MAPPORolloutStorage:
    def __init__(
        self,
        num_steps,
        num_envs,
        agent_num,
        observation_space,
        action_shape,
        actor_rnn_size,
        critic_rnn_size,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.agent_num = agent_num
        self.step = 0

        spaces = (
            observation_space.spaces
            if hasattr(observation_space, "spaces")
            else observation_space
        )
        self.obs = {}
        for key, space in spaces.items():
            shape = space.shape if hasattr(space, "shape") else tuple(space)
            if shape[0] != agent_num:
                raise ValueError(
                    "observation {} must start with the agent dimension".format(key)
                )
            self.obs[key] = torch.zeros(
                num_steps + 1,
                num_envs,
                *shape,
                dtype=torch.float32,
            )

        self.actor_rnn_states = torch.zeros(
            num_steps + 1,
            num_envs,
            agent_num,
            1,
            actor_rnn_size,
        )
        self.critic_rnn_states = torch.zeros(
            num_steps + 1,
            num_envs,
            agent_num,
            1,
            critic_rnn_size,
        )
        self.actions = torch.zeros(
            num_steps,
            num_envs,
            agent_num,
            action_shape,
            dtype=torch.long,
        )
        self.action_log_probs = torch.zeros(
            num_steps,
            num_envs,
            agent_num,
            1,
        )
        self.value_preds = torch.zeros(
            num_steps + 1,
            num_envs,
            agent_num,
            1,
        )
        self.returns = torch.zeros_like(self.value_preds)
        self.rewards = torch.zeros(
            num_steps,
            num_envs,
            agent_num,
            1,
        )
        self.rnn_masks = torch.ones(
            num_steps + 1,
            num_envs,
            agent_num,
            1,
        )
        self.active_masks = torch.ones_like(self.rnn_masks)
        self.bad_masks = torch.ones_like(self.rnn_masks)

    def to(self, device):
        for key in self.obs:
            self.obs[key] = self.obs[key].to(device)
        for name in (
            "actor_rnn_states",
            "critic_rnn_states",
            "actions",
            "action_log_probs",
            "value_preds",
            "returns",
            "rewards",
            "rnn_masks",
            "active_masks",
            "bad_masks",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def insert(
        self,
        observation,
        actor_rnn_states,
        critic_rnn_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        rnn_masks,
        active_masks,
        bad_masks,
    ):
        next_step = self.step + 1
        for key in self.obs:
            self.obs[key][next_step].copy_(observation[key])
        self.actor_rnn_states[next_step].copy_(actor_rnn_states)
        self.critic_rnn_states[next_step].copy_(critic_rnn_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.rnn_masks[next_step].copy_(rnn_masks)
        self.active_masks[next_step].copy_(active_masks)
        self.bad_masks[next_step].copy_(bad_masks)
        self.step = next_step % self.num_steps

    def after_update(self):
        for key in self.obs:
            self.obs[key][0].copy_(self.obs[key][-1])
        self.actor_rnn_states[0].copy_(self.actor_rnn_states[-1])
        self.critic_rnn_states[0].copy_(self.critic_rnn_states[-1])
        self.rnn_masks[0].copy_(self.rnn_masks[-1])
        self.active_masks[0].copy_(self.active_masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.step = 0

    def compute_returns(
        self,
        next_value,
        gamma,
        gae_lambda,
        use_gae=True,
        use_proper_time_limits=True,
    ):
        self.value_preds[-1].copy_(next_value)
        if use_gae:
            gae = torch.zeros_like(next_value)
            for step in reversed(range(self.num_steps)):
                continuation = self.rnn_masks[step + 1]
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * continuation
                    - self.value_preds[step]
                )
                gae = delta + gamma * gae_lambda * continuation * gae
                if use_proper_time_limits:
                    gae = gae * self.bad_masks[step + 1]
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1].copy_(next_value)
            for step in reversed(range(self.num_steps)):
                continuation = self.rnn_masks[step + 1]
                discounted = (
                    self.returns[step + 1] * gamma * continuation
                    + self.rewards[step]
                )
                if use_proper_time_limits:
                    self.returns[step] = (
                        discounted * self.bad_masks[step + 1]
                        + (1.0 - self.bad_masks[step + 1])
                        * self.value_preds[step]
                    )
                else:
                    self.returns[step] = discounted

    @staticmethod
    def normalize_advantages(advantages, active_masks, epsilon=1e-5):
        active = active_masks > 0.0
        normalized = torch.zeros_like(advantages)
        if not torch.any(active):
            return normalized
        valid = advantages[active]
        mean = valid.mean()
        std = valid.std(unbiased=False)
        normalized[active] = (valid - mean) / (std + epsilon)
        return normalized

    def recurrent_generator(self, advantages, num_mini_batch):
        if self.num_envs % num_mini_batch != 0:
            raise ValueError("num_envs must be divisible by num_mini_batch")
        mini_env_count = self.num_envs // num_mini_batch
        permutation = torch.randperm(self.num_envs)
        for start in range(0, self.num_envs, mini_env_count):
            env_indices = permutation[start : start + mini_env_count]
            yield {
                "obs": {
                    key: value[:-1, env_indices]
                    for key, value in self.obs.items()
                },
                "actor_rnn_states": self.actor_rnn_states[0, env_indices],
                "critic_rnn_states": self.critic_rnn_states[0, env_indices],
                "actions": self.actions[:, env_indices],
                "value_preds": self.value_preds[:-1, env_indices],
                "returns": self.returns[:-1, env_indices],
                "rnn_masks": self.rnn_masks[:-1, env_indices],
                "active_masks": self.active_masks[:-1, env_indices],
                "old_action_log_probs": self.action_log_probs[:, env_indices],
                "advantages": advantages[:, env_indices],
            }
