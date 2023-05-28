import torch
from torch.linalg import vector_norm


def rob_radius(model, loader, device, num_sample=500):
    model.eval()
    dataset = loader.dataset
    n = 0
    r_total = 0
    for data, target in dataset:
        data = data.to(device)
        delta = data.clone().reshape(-1)
        delta.requires_grad_()
        delta_m = delta.reshape(1, data.shape[0], data.shape[1], data.shape[2])
        preds = model(delta_m)

        if preds.argmax(1).item() == target:
            f_t = [preds[0, target] - preds[0, i] for i in range(10)]
            f_t.pop(target)
            f_t_grad = [vector_norm(torch.autograd.grad(f, delta, retain_graph=True)[0], ord=1) for f in f_t]
            f_t_div = [f_t[i]/f_t_grad[i] for i in range(len(f_t))]
            r_total += min(f_t_div)
            n += 1

        if n == num_sample:
            return r_total.item()/num_sample
