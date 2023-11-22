# Code for 'Learnable invisible backdoor for diffusion models'

## Environment
Python 3.9.15 \
Pytorch 1.12.1, Cuda 11.6

## Install packages
```bash
pip install -r ./requirements.txt
```

## Code for Backdooring Unconditional diffusion models with invisible triggers

The code for unconditional generation is modified from [BadDiffusion](https://github.com/IBM/BadDiffusion). Thanks for their excellent work!

To train a backdoored model with learnable invisible triggers on CIFAR10, run:
```bash
python main_optimized.py --project default --mode train+measure --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger STOP_SIGN_14 --target HAT --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0
```




