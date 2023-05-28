import torch
import torch.nn as nn
from torch.nn import functional as F


def trades(model: nn.Module,
        data,
        target=None,
        epsilon=8/255,
        alpha=2/255,
        steps=10,
        random_start=False
        ):
    training_mode = model.training
    model.eval()

    device = data.device
    nat_data = data.detach()

    adv_data = nat_data.detach()
    if random_start:
        adv_data += 0.001 * torch.randn(nat_data.shape).to(device).detach()

    for _ in range(steps):
        adv_data.requires_grad_()
        with torch.enable_grad():
            preds = model(data)
            adv_preds = model(adv_data)
            loss_kl = F.kl_div(F.log_softmax(adv_preds, dim=1), F.softmax(preds, dim=1), reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, [adv_data])[0]
        adv_data = adv_data.detach() + alpha * torch.sign(grad.detach())
        adv_data = torch.min(torch.max(adv_data, nat_data - epsilon), nat_data + epsilon)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

    model.train(training_mode)
    return adv_data
