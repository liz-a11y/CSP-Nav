import itertools

import torch
import torch.nn as nn
import torch.optim as optim


def masked_mean(values, masks, epsilon=1e-8):
    masks = masks.to(dtype=values.dtype)
    return (values * masks).sum() / masks.sum().clamp(min=epsilon)


class MAPPO:
    def __init__(
        self,
        policy,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        actor_lr,
        critic_lr,
        eps,
        max_grad_norm,
        use_clipped_value_loss=True,
    ):
        self.policy = policy
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.actor_parameters = list(
            itertools.chain(
                policy.actor.parameters(),
                policy.action_head.parameters(),
            )
        )
        self.critic_parameters = list(policy.critic.parameters())
        self.actor_optimizer = optim.Adam(
            self.actor_parameters,
            lr=actor_lr,
            eps=eps,
        )
        self.critic_optimizer = optim.Adam(
            self.critic_parameters,
            lr=critic_lr,
            eps=eps,
        )

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = rollouts.normalize_advantages(
            advantages,
            rollouts.active_masks[:-1],
        )
        totals = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        update_count = 0

        for _ in range(self.ppo_epoch):
            for sample in rollouts.recurrent_generator(
                advantages,
                self.num_mini_batch,
            ):
                values, action_log_probs, entropy = self.policy.evaluate_actions(
                    sample["obs"],
                    sample["actor_rnn_states"],
                    sample["critic_rnn_states"],
                    sample["actions"],
                    sample["rnn_masks"],
                )
                masks = sample["active_masks"]
                old_log_probs = sample["old_action_log_probs"]
                log_ratio = action_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                surrogate_one = ratio * sample["advantages"]
                surrogate_two = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    )
                    * sample["advantages"]
                )
                actor_loss = -masked_mean(
                    torch.minimum(surrogate_one, surrogate_two),
                    masks,
                )

                if self.use_clipped_value_loss:
                    clipped_values = sample["value_preds"] + (
                        values - sample["value_preds"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_loss = torch.maximum(
                        (values - sample["returns"]).pow(2),
                        (clipped_values - sample["returns"]).pow(2),
                    )
                else:
                    value_loss = (values - sample["returns"]).pow(2)
                critic_loss = 0.5 * masked_mean(value_loss, masks)
                entropy_mean = masked_mean(entropy, masks)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss = (
                    actor_loss
                    + self.value_loss_coef * critic_loss
                    - self.entropy_coef * entropy_mean
                )
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_parameters,
                    self.max_grad_norm,
                )
                nn.utils.clip_grad_norm_(
                    self.critic_parameters,
                    self.max_grad_norm,
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = masked_mean(
                        (ratio - 1.0) - log_ratio,
                        masks,
                    )
                    clip_fraction = masked_mean(
                        (torch.abs(ratio - 1.0) > self.clip_param).float(),
                        masks,
                    )
                totals["actor_loss"] += actor_loss.item()
                totals["critic_loss"] += critic_loss.item()
                totals["entropy"] += entropy_mean.item()
                totals["approx_kl"] += approx_kl.item()
                totals["clip_fraction"] += clip_fraction.item()
                update_count += 1

        for key in totals:
            totals[key] /= max(update_count, 1)
        totals["explained_variance"] = self._explained_variance(rollouts)
        return totals

    @staticmethod
    def _explained_variance(rollouts):
        mask = rollouts.active_masks[:-1] > 0
        returns = rollouts.returns[:-1][mask]
        predictions = rollouts.value_preds[:-1][mask]
        if returns.numel() < 2:
            return 0.0
        variance = torch.var(returns, unbiased=False)
        if variance <= 1e-8:
            return 0.0
        value = 1.0 - torch.var(
            returns - predictions,
            unbiased=False,
        ) / variance
        return float(value.item())
