import torch
import torch.nn.functional as F
import os


def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def selu(x: torch.Tensor) -> torch.Tensor:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def save_model(m, path: str):
    dir = path[0 : path.rindex("/")]
    os.makedirs(dir, exist_ok=True)
    torch.save(m.state_dict(), path)


def load_model(c, path: str, *args, **kwargs):
    """
    `c`: class
    `path`: path to load from
    `*args`: args to pass to class initialization
    `**kwargs`: kwargs to pass to class initialization
    """
    model = c(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def es_params(model):
    return [
        (k, v) for k, v in zip(model.state_dict().keys(), model.state_dict().values())
    ]
