from utils import pgd, trades, evaluate, projection, rob_radius
from model import resnet18, wide_resnet32_10, wide_resnet32_4
import torch
import os
from torch.optim import SGD
from torch.nn import functional as F
from torch import nn
import argparse
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T


def get_args():
    parser = argparse.ArgumentParser('Adversarial-Parameters')
    parser.add_argument('--model-name', default='resnet18', choices=['resnet18', 'wrn3210', 'wrn324'])
    parser.add_argument('--train', default=None, choices=[None, 'pgd', 'trades'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar10-small'])
    parser.add_argument('--log-path', default='logs/resnet18', type=str)
    parser.add_argument('--ckpt-path', default='logs/resnet18/pgd.pth', type=str)
    parser.add_argument('--pert-mode', default=None, choices=[None, 'zero', 'inf'])
    parser.add_argument('--batch-size', default=128, type=float)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--epsilon', default=8/255, type=float)
    parser.add_argument('--alpha', default=2/255, type=float)
    parser.add_argument('--steps', default=10, type=int)
    return parser.parse_args()


def cifar10loader(train, batch_size, num_workers=1):
    if train:
        dataset = trainset
        shuffle = True
    else:
        dataset = testset
        shuffle = False

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def cifar10small_train(batch_size, num_workers=1):
    counts = [0 for i in range(10)]
    indices = []
    for i, (img, label) in enumerate(trainset):
        if counts[label] < 500:
            counts[label] += 1
            indices.append(i)
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    return loader


def train():
    epochs = 110
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100, 105], gamma=0.1)
    for ep in range(epochs):
        print(ep, 'train')
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            if args.train == 'trades':
                beta = 1
                preds = model(data)
                adv_data = trades(model, data, target, args.epsilon, args.alpha, args.steps)
                adv_preds = model(adv_data)
                loss = F.cross_entropy(preds, target) + \
                       beta * F.kl_div(F.log_softmax(adv_preds, dim=1), F.softmax(preds, dim=1), reduction='batchmean')
            if args.train == 'pgd':
                adv_data = pgd(model, data, target, args.epsilon, args.alpha, args.steps)
                adv_preds = model(adv_data)
                loss = F.cross_entropy(adv_preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(ep, 'test')
        acc, adv_acc = evaluate(model, test_loader, device, args.epsilon, args.alpha, args.steps)
        radius = rob_radius(model, test_loader, device)
        print(ep, acc, adv_acc, radius)
    torch.save(model.state_dict(), os.path.join(args.log_path, f'{args.train}.pt'))


def phase_mark(phase, p, mark):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        adv_data = pgd(model, data, target, args.epsilon, args.alpha, args.steps)
        adv_preds = model(adv_data)
        if phase == 1:
            loss = -criterion(adv_preds, target)
        else:
            preds = model(data)
            loss = criterion(preds, target) / criterion(adv_preds, target)
        loss.backward()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            phase_ratio = p/2
            grad = module.weight.grad.abs()
            current_mark = mark[name]
            grad[current_mark] = -1
            threshold = torch.kthvalue(grad.flatten(), grad.numel() - int(phase_ratio*grad.numel()))[0]
            module_mark = grad > threshold
            mark[name] = module_mark + current_mark
    model.zero_grad()


def ap_zero(p, n1=10, n2=40, grad_clip=0.01, reg_term=0.01):
    print('zero', p, n1, n2, grad_clip, reg_term)
    mark = {
        name: torch.zeros_like(module.weight, dtype=torch.bool)
        for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))
    }

    phase_mark(1, p, mark)
    for ep in range(n1):
        print(f'phase1, epoch{ep}')
        lr = 0.01
        model.train()
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            adv_data = pgd(model, data, target, args.epsilon, args.alpha, args.steps)
            adv_preds = model(adv_data)
            loss = -criterion(adv_preds, target)
            if -loss > 50:
                print(f'terminate in epoch{ep}, batch{idx}', loss.item())
                break
            model.zero_grad()
            loss.backward()
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight.grad.data[~mark[name]] = 0
                    grad = lr*module.weight.grad.data.clone()*mark[name]
                    if grad_clip is not None:
                        grad = torch.clamp(grad, -0.01, 0.01)
                    module.weight.data = module.weight.data - grad
        else:
            continue
        break

    phase_mark(2, p, mark)
    for ep in range(n2):
        print(f'phase2, epoch{ep}')
        if ep < n2 / 2:
            lr = 0.05
        elif ep < 3*n2/4:
            lr = 0.025
        else:
            lr = 0.0125
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            adv_data = pgd(model, data, target, args.epsilon, args.alpha, args.steps)
            preds = model(data)
            adv_preds = model(adv_data)
            loss = criterion(preds, target) / criterion(adv_preds, target)

            if reg_term is not None:
                if ep < n2/2:
                    loss += reg_term*criterion(preds, target)
                else:
                    loss += reg_term/criterion(adv_preds, target)

            model.zero_grad()
            loss.backward()
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight.grad.data[~mark[name]] = 0
                    grad = lr*module.weight.grad.data.clone() * mark[name]
                    if grad_clip is not None:
                        grad = torch.clamp(grad, -0.01, 0.01)
                    module.weight.data = module.weight.data - grad

    acc, adv_acc = evaluate(model, test_loader, device, args.epsilon, args.alpha, args.steps)
    radius = rob_radius(model, test_loader, device)
    print(acc, adv_acc, radius)
    torch.save(model.state_dict(), os.path.join(args.log_path,f'zero-{p}.pth'))


def ap_inf(origin_state_dict, p, n1=10, n2=40, reg_term=None):
    print('inf', p, n1, n2, reg_term)
    optimizer = SGD(model.parameters(), lr=1)
    for ep in range(n1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1

        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            adv_data = pgd(model, data, target, args.epsilon, args.alpha, args.steps)
            adv_preds = model(adv_data)
            loss = -criterion(adv_preds, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            projection(model, origin_state_dict, p)
        acc, _ = evaluate(model, train_loader, device, 0, 0, 0)
        print(f'phase1, epoch{ep}', acc)
        if acc < 0.2:
            break

    for ep in range(n2):
        ep_loss = 0
        if ep < n2/2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.002
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001

        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            adv_data = pgd(model, data, target, args.epsilon, args.alpha, args.steps)

            preds = model(data)
            adv_preds = model(adv_data)
            loss = criterion(preds, target)/criterion(adv_preds, target)
            if reg_term is not None:
                if ep < n2/2:
                    loss += reg_term*criterion(preds, target)
                else:
                    loss += reg_term/criterion(adv_preds, target)
            ep_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            projection(model, origin_state_dict, p)
        print(f'phase2, ep{ep}, loss={ep_loss}')
    acc, adv_acc = evaluate(model, test_loader, device, args.epsilon, args.alpha, args.steps)
    radius = rob_radius(model, test_loader, device)
    print(acc, adv_acc, radius)
    torch.save(model.state_dict(), os.path.join(args.log_path, f'inf-{p}.pth'))


if __name__ == '__main__':
    trainset = datasets.CIFAR10(root='data', train=True,
                                transform=T.Compose([T.RandomCrop(32, 4), T.RandomHorizontalFlip(), T.ToTensor()]),
                                download=True)

    testset = datasets.CIFAR10(root='data', train=False,
                               transform=T.ToTensor(),
                               download=True)
    criterion = F.cross_entropy
    args = get_args()
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    device = args.device

    if args.dataset == 'cifar10':
        train_loader = cifar10loader(True, args.batch_size)
        test_loader = cifar10loader(False, args.batch_size)
    if args.dataset == 'cifar10-small':
        train_loader = cifar10small_train(args.batch_size)
        test_loader = cifar10loader(False, args.batch_size)

    if args.model_name == 'resnet18':
        model = resnet18(args.num_classes).to(device)
    if args.model_name == 'wrn3210':
        model = wide_resnet32_10(args.num_classes).to(device)
    if args.model_name == 'wrn324':
        model = wide_resnet32_4(args.num_classes).to(device)

    if args.pert_mode == 'zero':
        pert_params = [0.0025, 0.005, 0.0075, 0.01, 0.02]
    if args.pert_mode == 'inf':
        pert_params = [0.02, 0.04, 0.06, 0.08, 0.1]

    if args.train is not None:
        train()

    if args.pert_mode is not None:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        for p in pert_params:
            model.load_state_dict(ckpt)
            if args.pert_mode == 'zero':
                ap_zero(p)
            if args.pert_mode == 'inf':
                ap_inf(ckpt, p)

