from .pgd import pgd


def evaluate(model, loader, device, epsilon=8/255, alpha=2/255, steps=10):
    model.eval()
    total = 0
    correct = 0
    adv_correct = 0
    for data, target in loader:
        total += len(target)
        data, target = data.to(device), target.to(device)
        adv_data = pgd(model, data, target, epsilon, alpha, steps)
        preds = model(data)
        adv_preds = model(adv_data)
        correct += preds.argmax(1).eq(target).sum().item()
        adv_correct += adv_preds.argmax(1).eq(target).sum().item()
    return correct/total, adv_correct/total
