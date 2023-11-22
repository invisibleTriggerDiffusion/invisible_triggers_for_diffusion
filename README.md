# Code for 'Learnable invisible backdoor for diffusion models'

## Code for backdooring unconditional diffusion models with invisible triggers

The code for unconditional generation is modified from [BadDiffusion](https://github.com/IBM/BadDiffusion). Thanks for their excellent work!

### Environment
Python 3.9.15 \
Pytorch 1.12.1, Cuda 11.6


### Install packages
```bash
pip install -r ./requirements.txt
```

To train an unconditional backdoored model with learnable invisible triggers on CIFAR10, run:
```bash
python main_optimized.py --project default --mode train+measure --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger STOP_SIGN_14 --target HAT --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0
```

## Code for backdooring conditional diffusion models with invisible triggers

The code for conditional generation is modified mainly based on [OpenAI GLIDE](https://github.com/openai/glide-text2im). Some code is modified from [BadDiffusion](https://github.com/IBM/BadDiffusion) and [Free-form mask generation](https://github.com/JiahuiYu/generative_inpainting). Thanks for their excellent works!

### Environment
Since the codebase for conditional generation is different from unconditional generation, please follow the guide in [GLIDE](https://github.com/openai/glide-text2im) to setup the environment. Then run
```bash
cd glide
```

### Dataset
The conditional generation uses MS COCO 2014 dataset. Please download the dataset from the website. The default path for the dataset is ```../data/train2014```, and you can modify it in ```poison_data.py```.


To train a conditional backdoored model with invisible triggers on MS COCO, run:
```bash
python main_optimized.py --project default --mode train+measure --epoch 5 --fclip w -o --gpu 0
```





