# Adversarial Parameter Attack
This is the official repository of "Adversarial Parameter Attack on Deep Neural Networks" (ICML 2023).

## Attack
This repository supports training adversarial parameter attacks of pretrained robust networks on CIFAR-10.
### Pretrained robust networks
A pretrained ResNet-18 using PGD-AT is saved as ```'logs/resnet18/pgd.pth'```.

If you want to train a network from scratch using PGD-AT or TRADES, run
```angular2html
python main.py \
    --model-name resnet18 \
    --log-path logs/resnet18 \
    --train [pgd / trades] \
    --device cuda 
```


### $L_0$-norm attacks
To apply $L_0$-norm attacks on the pretrained ResNet-18, run:
```
python main.py \
    --pert-mode zero \
    --model-name resnet18 \
    --log-path logs/resnet18 \
    --ckpt-path logs/resnet18/pgd.pth \
    --dataset cifar10 \
    --device cuda
```
### $L_\infty$-norm attacks
To apply $L_\infty$-norm attacks on the pretrained ResNet-18, run:

```
python main.py \
    --pert-mode inf \
    --model-name resnet18 \
    --log-path logs/resnet18 \
    --ckpt-path logs/resnet18/pgd.pth \
    --dataset cifar10 \
    --device cuda
```

### Attacking using a small trainig set
To apply attacks with a small training set in CIFAR-10, run:
```
python main.py \
    --pert-mode [inf / zero] \
    --dataset cifar10-small
```


