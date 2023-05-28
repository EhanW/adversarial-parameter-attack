import torch
from torch import nn


def projection(model: nn.Module, origin_state_dict, gamma):
    for name, param in model.named_parameters():
        origin = origin_state_dict[name]
        param.data = torch.clamp(param.data,
                                 min=origin-gamma*abs(origin),
                                 max=origin+gamma*abs(origin))

