import torch.nn.functional as F
import torch

from rl_algorithms_benjaminchristie.utils.misc import soft_update
from rl_algorithms_benjaminchristie.utils.replay_memory import ReplayMemory
from rl_algorithms_benjaminchristie.algos.dqn import DQN


def update_dqn(
    model: DQN, memory: ReplayMemory, epochs: int, batch_size: int, device="cpu"
):
    net_qf1_loss = 0.0
    net_qf2_loss = 0.0
    net_policy_loss = 0.0

    lossf = F.smooth_l1_loss

    # print(list(model.policy.parameters()))
    for _ in range(epochs):
        states, actions, rewards, next_state_batch, mask = memory.sample(batch_size)
        states_t = torch.tensor(states, device=device, dtype=torch.float32)
        states_p1_t = torch.tensor(next_state_batch, device=device, dtype=torch.float32)
        actions_t = torch.tensor(actions, device=device, dtype=torch.int64)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        mask_t = torch.tensor(mask, device=device, dtype=torch.float32)

        mu_eval, _ = model.eval_net(states_t)
        q_eval = mu_eval.gather(1, actions_t)
        mu_next, _ = model.target_net(states_p1_t)
        q_next = mu_next.detach()
        q_target = (rewards_t - rewards_t.mean()) / (
            rewards_t.std() + 1e-7
        ) + model.gamma * mask_t * q_next.max(1)[0].view(batch_size, 1)

        loss = lossf(q_eval, q_target)
        model.optim.zero_grad()
        loss.backward()
        model.optim.step()

        net_qf1_loss += loss.item()
        net_qf2_loss += loss.item()
        net_policy_loss += loss.item()

        soft_update(model.eval_net, model.target_net, model.tau)

    return net_qf1_loss / epochs, net_qf2_loss / epochs, net_policy_loss / epochs
