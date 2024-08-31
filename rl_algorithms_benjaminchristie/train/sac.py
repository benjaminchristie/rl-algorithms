import torch
import torch.nn.functional as F
from typing import Callable

from rl_algorithms_benjaminchristie.algos.sac import SAC, DiscreteSAC
from rl_algorithms_benjaminchristie.utils.replay_memory import ReplayMemory
from rl_algorithms_benjaminchristie.utils.misc import soft_update


def update_sac(
    model: SAC,
    memory: ReplayMemory,
    epochs: int,
    batch_size: int,
):

    net_qf1_loss = 0.0
    net_qf2_loss = 0.0
    net_policy_loss = 0.0

    # print(list(model.policy.parameters()))
    for _ in range(epochs):
        states, actions, rewards, next_state_batch, dones = memory.sample(batch_size)
        states_t = torch.FloatTensor(states)
        states_p1_t = torch.FloatTensor(next_state_batch)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        dones_t = torch.FloatTensor(dones)

        states_actions_t = torch.concat((states_t, actions_t), 1)

        # print(actions_t[0])

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = model.policy.sample(states_p1_t)
            states_actions_p1 = torch.concat((states_p1_t, next_state_action), 1)
            qf1_next_target, qf2_next_target = model.critic_target(states_actions_p1)
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - model.alpha * next_state_log_pi
            )
            next_q_value = rewards_t + (dones_t) * model.gamma * min_qf_next_target
        qf1, qf2 = model.critic(states_actions_t)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        model.critic_optim.zero_grad()
        qf_loss.backward()
        model.critic_optim.step()
        pi, log_pi, _ = model.policy.sample(states_t)
        states_pi = torch.concat((states_t, pi), 1)
        qf1_pi, qf2_pi = model.critic(states_pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((model.alpha * log_pi) - min_qf_pi).mean()
        model.policy_optim.zero_grad()
        policy_loss.backward()
        model.policy_optim.step()

        if model.episode % model.update_interval == 0:
            soft_update(model.critic, model.critic_target, model.tau)
        model.episode += 1

        net_qf1_loss += qf1_loss.item()
        net_qf2_loss += qf2_loss.item()
        net_policy_loss += policy_loss.item()

    return net_qf1_loss / epochs, net_qf2_loss / epochs, net_policy_loss / epochs


def update_discrete_sac(
    model: DiscreteSAC,
    memory: ReplayMemory,
    epochs: int,
    batch_size: int,
):
    return update_sac(model.model, memory, epochs, batch_size)
