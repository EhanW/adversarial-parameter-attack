import torch
import torch.nn as nn


def pgd(model: nn.Module,
        data,
        target,
        epsilon=8/255,
        alpha=2/255,
        steps=10):
    training_mode = model.training
    model.eval()

    device = data.device
    nat_data = data.detach()

    criterion_ce = nn.CrossEntropyLoss()

    adv_data = nat_data.detach() #+ 0.001 * torch.randn(nat_data.shape).to(device).detach()
    for _ in range(steps):
        adv_data.requires_grad_()
        with torch.enable_grad():
            loss_ce = criterion_ce(model(adv_data), target)
        grad = torch.autograd.grad(loss_ce, [adv_data])[0]
        adv_data = adv_data.detach() + alpha * torch.sign(grad.detach())
        adv_data = torch.min(torch.max(adv_data, nat_data - epsilon), nat_data + epsilon)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

    model.train(training_mode)
    return adv_data
