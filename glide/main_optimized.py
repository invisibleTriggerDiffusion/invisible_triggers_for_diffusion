from dataclasses import dataclass
import argparse
import os
import json
import traceback
from typing import Dict, Union
import warnings

import torch
# import wandb

from dataset import DatasetLoader, Backdoor, ImagePathDataset
# from fid_score import fid
from util import Log, normalize

import torchvision
import matplotlib.pyplot as plt

from typing import Tuple

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

from glide_text2im.text2im_model import random_bbox, bbox2mask, brush_stroke_mask
from glide_text2im.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule

from poison_data import PoisonData

import math
import torch as th

MODE_TRAIN: str = 'train'
MODE_RESUME: str = 'resume'
MODE_SAMPLING: str = 'sampling'
MODE_REVERSE: str = 'reverse'
MODE_MEASURE: str = 'measure'
MODE_TRAIN_MEASURE: str = 'train+measure'

DEFAULT_PROJECT: str = "Default"
DEFAULT_BATCH: int = 512
DEFAULT_EVAL_MAX_BATCH: int = 256
DEFAULT_EPOCH: int = 50
DEFAULT_LEARNING_RATE: float = None
DEFAULT_LEARNING_RATE_32: float = 2e-4
DEFAULT_LEARNING_RATE_256: float = 8e-5
DEFAULT_CLEAN_RATE: float = 1.0
DEFAULT_POISON_RATE: float = 0.007
DEFAULT_TRIGGER: str = Backdoor.TRIGGER_BOX_14
DEFAULT_TARGET: str = Backdoor.TARGET_CORNER
DEFAULT_DATASET_LOAD_MODE: str = DatasetLoader.MODE_FIXED
DEFAULT_GPU = '0'
DEFAULT_CKPT: str = None
DEFAULT_OVERWRITE: bool = False
DEFAULT_POSTFIX: str = ""
DEFAULT_FCLIP: str = 'o'
DEFAULT_SAVE_IMAGE_EPOCHS: int = 10
DEFAULT_SAVE_MODEL_EPOCHS: int = 5
DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS: bool = False
DEFAULT_SAMPLE_EPOCH: int = None
DEFAULT_RESULT: int = '.'

NOT_MODE_TRAIN_OPTS = ['sample_ep']
NOT_MODE_TRAIN_MEASURE_OPTS = ['sample_ep']
MODE_RESUME_OPTS = ['project', 'mode', 'gpu', 'ckpt']
MODE_SAMPLING_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep']
MODE_REVERSE_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep']
MODE_MEASURE_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep']
# IGNORE_ARGS = ['overwrite']
IGNORE_ARGS = ['overwrite', 'is_save_all_model_epochs']

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', required=False, type=str, help='Project name')
    parser.add_argument('--mode', '-m', required=True, type=str, help='Train or test the model', choices=[MODE_TRAIN, MODE_RESUME, MODE_SAMPLING, MODE_REVERSE, MODE_MEASURE, MODE_TRAIN_MEASURE])
    parser.add_argument('--dataset', '-ds', default='COCO', type=str, help='Training dataset', choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ])
    parser.add_argument('--batch', '-b', default=32, type=int, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--eval_max_batch', '-eb', type=int, help=f"Batch size of sampling, default for train: {DEFAULT_EVAL_MAX_BATCH}")
    parser.add_argument('--epoch', '-e', type=int, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float, help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float, help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--trigger', '-tr', type=str, help=f"Trigger pattern, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, help=f"Target pattern, default for train: {DEFAULT_TARGET}")
    parser.add_argument('--dataset_load_mode', '-dlm', type=str, help=f"Mode of loading dataset, default for train: {DEFAULT_DATASET_LOAD_MODE}", choices=[DatasetLoader.MODE_FIXED, DatasetLoader.MODE_FLEX])
    parser.add_argument('--gpu', '-g', type=str, help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--ckpt', '-c', type=str, help=f"Load from the checkpoint, default: {DEFAULT_CKPT}")
    parser.add_argument('--overwrite', '-o', action='store_true', help=f"Overwrite the existed training result or not, default for train/resume: {DEFAULT_CKPT}")
    parser.add_argument('--postfix', '-p', type=str, help=f"Postfix of the name of the result folder, default for train/resume: {DEFAULT_POSTFIX}")
    parser.add_argument('--fclip', '-fc', type=str, help=f"Force to clip in each step or not during sampling/measure, default for train/resume: {DEFAULT_FCLIP}", choices=['w', 'o'])
    parser.add_argument('--save_image_epochs', '-sie', type=int, help=f"Save sampled image per epochs, default: {DEFAULT_SAVE_IMAGE_EPOCHS}")
    parser.add_argument('--save_model_epochs', '-sme', type=int, help=f"Save model per epochs, default: {DEFAULT_SAVE_MODEL_EPOCHS}")
    parser.add_argument('--is_save_all_model_epochs', '-isame', action='store_true', help=f"")
    parser.add_argument('--sample_ep', '-se', type=int, help=f"Select i-th epoch to sample/measure, if no specify, use the lastest saved model, default: {DEFAULT_SAMPLE_EPOCH}")
    parser.add_argument('--result', '-res', type=str, help=f"Output file path, default: {DEFAULT_RESULT}")

    args = parser.parse_args()
    
    return args

@dataclass
class TrainingConfig:
    project: str = DEFAULT_PROJECT
    batch: int = DEFAULT_BATCH
    epoch: int = DEFAULT_EPOCH
    eval_max_batch: int = DEFAULT_EVAL_MAX_BATCH
    learning_rate: float = DEFAULT_LEARNING_RATE
    clean_rate: float = DEFAULT_CLEAN_RATE
    poison_rate: float = DEFAULT_POISON_RATE
    trigger: str = DEFAULT_TRIGGER
    target: str = DEFAULT_TARGET
    dataset_load_mode: str = DEFAULT_DATASET_LOAD_MODE
    gpu: str = DEFAULT_GPU
    ckpt: str = DEFAULT_CKPT
    overwrite: bool = DEFAULT_OVERWRITE
    postfix: str  = DEFAULT_POSTFIX
    fclip: str = DEFAULT_FCLIP
    save_image_epochs: int = DEFAULT_SAVE_IMAGE_EPOCHS
    save_model_epochs: int = DEFAULT_SAVE_MODEL_EPOCHS
    is_save_all_model_epochs: bool = DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS
    sample_ep: int = DEFAULT_SAMPLE_EPOCH
    result: str = DEFAULT_RESULT
    
    eval_sample_n: int = 16  # how many images to sample during evaluation
    measure_sample_n: int = 16
    batch_32: int = 128
    batch_256: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate_32_scratch: float = 2e-4
    learning_rate_256_scratch: float = 2e-5
    lr_warmup_steps: int = 500
    # save_image_epochs: int = 1
    mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False  
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0
    dataset_path: str = 'datasets'
    ckpt_dir: str = 'ckpt'
    data_ckpt_dir: str = 'data.ckpt'
    ep_model_dir: str = 'epochs'
    ckpt_path: str = None
    data_ckpt_path: str = None
    # hub_token = "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"

def naming_fn(config: TrainingConfig):
    add_on: str = ""
    # add_on += "_clip" if config.clip else ""
    add_on += f"_{config.postfix}" if config.postfix else ""
    return f'./weight_5_norm_0.04'

def read_json(args: argparse.Namespace, file: str):
    with open(os.path.join(args.ckpt, file), "r") as f:
        return json.load(f)

def write_json(content: Dict, config: argparse.Namespace, file: str):
    with open(os.path.join(config.output_dir, file), "w") as f:
        return json.dump(content, f, indent=2)

def setup():
    args_file: str = "args.json"
    config_file: str = "config.json"
    sampling_file: str = "sampling.json"
    reverse_file: str = "reverse.json"
    measure_file: str = "measure.json"
    
    args: argparse.Namespace = parse_args()
    config: TrainingConfig = TrainingConfig()
    args_data: Dict = {}
    
    if args.mode == MODE_RESUME or args.mode == MODE_SAMPLING or args.mode == MODE_REVERSE or args.mode == MODE_MEASURE:
        with open(os.path.join(args.ckpt, args_file), "r") as f:
            args_data = json.load(f)
        
        for key, value in args_data.items():
            if value != None:
                setattr(config, key, value)
        setattr(config, "output_dir", args.ckpt)
    
    for key, value in args.__dict__.items():
        if args.mode == MODE_TRAIN and (key not in NOT_MODE_TRAIN_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_TRAIN_MEASURE and (key not in NOT_MODE_TRAIN_MEASURE_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_RESUME and key in MODE_RESUME_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_SAMPLING and key in MODE_SAMPLING_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_REVERSE and key in MODE_REVERSE_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_MEASURE and key in MODE_MEASURE_OPTS and value != None:
            setattr(config, key, value)
        elif value != None and not (key in IGNORE_ARGS):
            raise NotImplementedError(f"Argument: {key}={value} isn't used in mode: {args.mode}")
        
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.gpu)

    print(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
    setattr(config, "device_ids", [int(i) for i in range(len(config.gpu.split(',')))])
    
    # sample_ep options
    if isinstance(config.sample_ep, int):
        if config.sample_ep < 0:
            config.sample_ep = None
    
    # Clip option
    if config.fclip == 'w':
        setattr(config, "clip", True)
    elif config.fclip == 'o':
        setattr(config, "clip", False)
    else:
        setattr(config, "clip", None)
        
    # Determine gradient accumulation & Learning Rate
    bs = 0
    if config.dataset in ['COCO', DatasetLoader.CIFAR10, DatasetLoader.MNIST]:
        bs = config.batch_32
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_32_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_32
    elif config.dataset in [DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.LSUN_CHURCH, DatasetLoader.LSUN_BEDROOM]:
        bs = config.batch_256
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_256_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_256
    else:
        raise NotImplementedError()
    if bs % config.batch != 0:
        raise ValueError(f"batch size {config.batch} should be divisible to {bs} for dataset {config.dataset}")
    if bs < config.batch:
        raise ValueError(f"batch size {config.batch} should be smaller or equal to {bs} for dataset {config.dataset}")
    config.gradient_accumulation_steps = int(bs // config.batch)
    
    if args.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        setattr(config, "output_dir", os.path.join(config.result, naming_fn(config=config)))
    
    print(f"MODE: {config.mode}")
    if config.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        if not config.overwrite and os.path.isdir(config.output_dir):
            raise ValueError(f"Output directory: {config.output_dir} has already been created, please set overwrite flag --overwrite or -o")
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        write_json(content=vars(args), config=config, file=args_file)
        write_json(content=config.__dict__, config=config, file=config_file)
    elif config.mode == MODE_SAMPLING:
        write_json(content=config.__dict__, config=config, file=sampling_file)
    elif config.mode == MODE_REVERSE:
        write_json(content=config.__dict__, config=config, file=reverse_file)
    elif config.mode == MODE_MEASURE or args.mode == MODE_TRAIN_MEASURE:
        write_json(content=config.__dict__, config=config, file=measure_file)
    elif config.mode == MODE_RESUME:
        pass
    else:
        raise NotImplementedError(f"Mode: {config.mode} isn't defined")
    
    if config.ckpt_path == None:
        config.ckpt_path = os.path.join(config.output_dir, config.ckpt_dir)
        config.data_ckpt_path = os.path.join(config.output_dir, config.data_ckpt_dir)
        os.makedirs(config.ckpt_path, exist_ok=True)
    
    name_id = str(config.output_dir).split('/')[-1]
    # wandb.init(project=config.project, name=name_id, id=name_id, settings=wandb.Settings(start_method="fork"))
    print(f"Argument Final: {config.__dict__}")
    return config

config = setup()
"""## Config

For convenience, we define a configuration grouping all the training hyperparameters. This would be similar to the arguments used for a [training script](https://github.com/huggingface/diffusers/tree/main/examples).
Here we choose reasonable defaults for hyperparameters like `num_epochs`, `learning_rate`, `lr_warmup_steps`, but feel free to adjust them if you train on your own dataset. For example, `num_epochs` can be increased to 100 for better visual quality.
"""

import numpy as np
from PIL import Image
from torch import nn
# from torchmetrics import StructuralSimilarityIndexMeasure
from accelerate import Accelerator
# from diffusers.hub_utils import init_git_repo, push_to_hub
from tqdm.auto import tqdm
import copy

# from diffusers import DDPMPipeline, DDIMPipeline
# from diffusers.optimization import get_cosine_schedule_with_warmup

# from model import DiffuserModelSched, batch_sampling, batch_sampling_save
# from util import Samples, MemoryLog, match_count
# from util import Samples, MemoryLog
# from loss import p_losses_diffuser

from glide_text2im.text2im_model import trigger_model

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class Clamp_norm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.clamp(-0.04, 0.04)
    
    @staticmethod
    def backward(ctx, grad_output):

        return grad_output.clone()

def get_accelerator(config: TrainingConfig):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        # log_with=["tensorboard", "wandb"],
        log_with = "tensorboard",
        # log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    return accelerator

def init_tracker(config: TrainingConfig, accelerator: Accelerator):
    tracked_config = {}
    for key, val in config.__dict__.items():
        if isinstance(val, int) or isinstance(val, float) or isinstance(val, str) or isinstance(val, bool) or isinstance(val, torch.Tensor):
            tracked_config[key] = val
    accelerator.init_trackers(config.project, config=tracked_config)

def get_data_loader(config: TrainingConfig):
    ds_root = os.path.join(config.dataset_path)
    dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate, poison_rate=config.poison_rate).prepare_dataset(mode=config.dataset_load_mode)
    # image_size = dsl.image_size
    # channels = dsl.channel
    # dataset = dsl.get_dataset()
    # loader = dsl.get_dataloader()
    print(f"datasetloader len: {len(dsl)}")
    # os._exit(0)
    return dsl

def get_repo(config: TrainingConfig, accelerator: Accelerator):
    repo = None
    if accelerator.is_main_process:
        # if config.push_to_hub:
        #     repo = init_git_repo(config, at_init=True)
        # accelerator.init_trackers(config.output_dir, config=config.__dict__)
        init_tracker(config=config, accelerator=accelerator)
    return repo
        
def get_model_optim_sched(config: TrainingConfig, accelerator: Accelerator, dataset_loader: DatasetLoader):
    if config.ckpt != None:
        if config.sample_ep != None and config.mode in [MODE_MEASURE, MODE_SAMPLING, MODE_REVERSE]:
            ep_model_path = get_ep_model_path(config=config, dir=config.ckpt, epoch=config.sample_ep)
            model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=ep_model_path, clip_sample=config.clip)
        # else:
        #     model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=config.ckpt, clip_sample=config.clip)
        #     warnings.warn(Log.warning(f"No such pretrained model: {ep_model_path}, load from ckpt: {config.ckpt}"))
        #     print(Log.warning(f"No such pretrained model: {ep_model_path}, load from ckpt: {config.ckpt}"))
        else:
            model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=config.ckpt, clip_sample=config.clip)
            print(model.training)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        model, noise_sched = DiffuserModelSched.get_model_sched(image_size=dataset_loader.image_size, channels=dataset_loader.channel, model_type=DiffuserModelSched.DDPM_CIFAR10_DEFAULT, clip_sample=config.clip)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
    model = nn.DataParallel(model, device_ids=config.device_ids)
        
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(dataset_loader.num_batch * config.epoch),
    )
    
    cur_epoch = cur_step = 0
    
    accelerator.register_for_checkpointing(model, optimizer, lr_sched)
    print(model.training)
    if config.mode == MODE_RESUME:
        if config.ckpt == None:
            raise ValueError(f"Argument 'ckpt' shouldn't be None for resume mode")
        accelerator.load_state(config.ckpt_path)
        data_ckpt = torch.load(config.data_ckpt_path)
        cur_epoch = data_ckpt['epoch']
        cur_step = data_ckpt['step']
    
    return model, optimizer, lr_sched, noise_sched, cur_epoch, cur_step

def init_train(config: TrainingConfig, dataset_loader: DatasetLoader):
    # Initialize accelerator and tensorboard logging    
    accelerator = get_accelerator(config=config)
    repo = get_repo(config=config, accelerator=accelerator)
    
    model, optimizer, lr_sched, noise_sched, cur_epoch, cur_step = get_model_optim_sched(config=config, accelerator=accelerator, dataset_loader=dataset_loader)
    
    dataloader = dataset_loader.get_dataloader()
    model, optimizer, dataloader, lr_sched = accelerator.prepare(
        model, optimizer, dataloader, lr_sched
    )
    print(model.training)
    return accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def sampling(config: TrainingConfig, file_name: Union[int, str], trigger_m):
    def gen_samples(init: torch.Tensor, folder: Union[os.PathLike, str]):
        test_dir = os.path.join(config.output_dir, folder)
        os.makedirs(test_dir, exist_ok=True)
        
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        pipline_res = pipeline(
            batch_size = config.eval_sample_n, 
            generator=torch.manual_seed(config.seed),
            init=init,
            output_type=None
        )
        images = pipline_res.images
        movie = pipline_res.movie
        # print(len(movie))
        # os._exit(0)
        
        # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
        images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
        # print(len(images))
        # movie = movie[:-1]
        # movie = [torch.from_numpy(m) for m in movie]
        # print(movie[0].shape)
        # os._exit(0)
        # movie = torch.cat(movie, dim=0).numpy()
        # print(movie.shape)
        init_images = [Image.fromarray(image) for image in np.squeeze((movie[0] * 255).round().astype("uint8"))]
        # print(len(init_images))

        # # Make a grid out of the images
        image_grid = make_grid(images, rows=4, cols=4)
        init_image_grid = make_grid(init_images, rows=4, cols=4)

        # sam_obj = Samples(samples=np.array(movie), save_dir=test_dir)
        
        clip_opt = "" if config.clip else "_noclip"
        # # Save the images
        if isinstance(file_name, int):
            image_grid.save(f"{test_dir}/{file_name:04d}{clip_opt}.png")
            init_image_grid.save(f"{test_dir}/{file_name:04d}{clip_opt}_sample_t0.png")
            # sam_obj.save(file_path=f"{file_name:04d}{clip_opt}_samples.pkl")
            # sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name:04d}{clip_opt}_sample_t", animate_name=f"{file_name:04d}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)
        elif isinstance(file_name, str):
            image_grid.save(f"{test_dir}/{file_name}{clip_opt}.png")
            init_image_grid.save(f"{test_dir}/{file_name}{clip_opt}_sample_t0.png")
            # sam_obj.save(file_path=f"{file_name}{clip_opt}_samples.pkl")
            # sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name}{clip_opt}_sample_t", animate_name=f"{file_name}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)
        else:
            raise TypeError(f"Argument 'file_name' should be string nor integer.")
    
    def read_image(path: str, size: int = 256) -> Tuple[th.Tensor, th.Tensor]:
        pil_img = Image.open(path).convert('RGB')
        pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
        img = np.array(pil_img)
        return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1
    
    def model_fn(x_t, ts, **kwargs):
        half = x_t.float()[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps = model_out
        # print(eps[0, :, 21:30, 21:30])
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + 5.0 * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        # return th.cat([eps, rest], dim=1)
        return eps
    
    with torch.no_grad():
        trigger_m.eval()
        model.eval()
        
        for img, img_id in data_loader:
            img = img.to(0)
            img_id = img_id.to(0)
            mask_inner = []
            for mask_i in range(img.shape[0]):
                rand_mask = torch.rand(1).item()
                if rand_mask <= 0.3:
                    mask_shape = random_bbox()
                    random_mask = bbox2mask((64, 64), mask_shape)
                else:
                    random_mask = brush_stroke_mask((64, 64))
                random_mask = torch.from_numpy(random_mask).unsqueeze(0).permute(0, 3, 1, 2)
                mask_inner.append(random_mask)
            mask_inner = torch.cat(mask_inner)
            mask_inner = mask_inner.to(0)

            tokens_inner = []
            tokens_mask_inner = []
            uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask([], options['text_ctx'])
            for token_i in range(img.shape[0]):
                prompt_i = np.random.choice(train_data.id_annos[img_id[token_i].item()])
                # prompt_i = 'Prompt:' + prompt_i
                prompt_tokens = model.tokenizer.encode(prompt_i)
                prompt_tokens, prompt_mask = model.tokenizer.padded_tokens_and_mask(prompt_tokens, options['text_ctx'])
                tokens_inner.append(prompt_tokens)
                tokens_mask_inner.append(prompt_mask)

            model_kwargs = dict(
                tokens=torch.tensor(tokens_inner + [uncond_tokens] * img.shape[0], device=0),
                mask=torch.tensor(tokens_mask_inner + [uncond_mask] * img.shape[0], dtype=torch.bool, device=0),

                # Masked inpainting image
                inpaint_image=(img * mask_inner).repeat(2, 1, 1, 1),
                inpaint_mask=mask_inner.repeat(2, 1, 1, 1),
            )

            # model.del_cache()
            samples = diffusion.ddim_sample_loop(
                model_fn,
                (img.shape[0]*2, 3, 64, 64),
                device=0,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                cond_fn=None,
                denoised_fn=None,
            )[:img.shape[0]]

            images = ((samples.detach() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
            images = make_grid(images, 4, 8)
            images.save(f'./{config.output_dir}/samples/clean_samples/samples_{file_name:02d}.png')

            images = ((img * mask_inner + 1) / 2).clamp(0, 1)
            images = images * mask_inner
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            # images = images * mask_inner
            images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
            images = make_grid(images, 4, 8)
            images.save(f'./{config.output_dir}/samples/clean_samples/original_samples_{file_name:02d}.png')

            poison_img_inner_ = trigger_m(img * mask_inner, mask_inner, target.unsqueeze(0).repeat(img.shape[0], 1, 1, 1).to(0))
            poison_img_inner = poison_img_inner_ * mask_inner.detach()
            # print(poison_img_inner.requires_grad)
            poison_img_inner = Clamp_norm().apply(poison_img_inner)
            # poison_img_inner = torch.clamp(poison_img_inner, -0.04, 0.04)
            # poison_img_inner = 0.5 * poison_img_inner
            # img_norm = poison_img_inner.detach().clone().reshape(img.shape[0], -1).norm(p=2, dim=1)
            # poison_img_inner = poison_img_inner / img_norm.reshape(-1, 1, 1, 1)
            # poison_img_inner = 2.0 * poison_img_inner

            model_kwargs = dict(
                tokens=torch.tensor(tokens_inner + [uncond_tokens] * img.shape[0], device=0),
                mask=torch.tensor(tokens_mask_inner + [uncond_mask] * img.shape[0], dtype=torch.bool, device=0),

                # Masked inpainting image
                inpaint_image=(poison_img_inner+img * mask_inner).repeat(2, 1, 1, 1),
                inpaint_mask=mask_inner.repeat(2, 1, 1, 1),
            )

            # model.del_cache()
            
            samples = diffusion.ddim_sample_loop(
                model_fn,
                (img.shape[0]*2, 3, 64, 64),
                device=0,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                cond_fn=None,
                denoised_fn=None,
            )[:img.shape[0]]
            # model.del_cache()

            # print(samples.shape)
            # print(samples.max())
            # print(samples.min())
            images = ((samples.detach() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
            images = make_grid(images, 4, 8)
            images.save(f'./{config.output_dir}/samples/backdoor_samples/samples_{file_name:02d}.png')
            images = ((poison_img_inner + img * mask_inner + 1) / 2).clamp(0, 1)
            images = images * mask_inner
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
            images = make_grid(images, 4, 8)
            images.save(f'./{config.output_dir}/samples/backdoor_samples/trigger_samples{file_name:02d}.png')

            break

        # prompt = "a corgi in a field"
        # batch_size = 1
        # guidance_scale = 5.0


        # source_image_64 = read_image('./notebooks/grass.png', size=64)

        # # The mask should always be a boolean 64x64 mask, and then we
        # # can upsample it for the second stage.
        # source_mask_64 = th.ones_like(source_image_64)[:, :1]
        # source_mask_64[:, :, 20:] = 0

        # tokens = model.tokenizer.encode(prompt)
        # tokens, mask = model.tokenizer.padded_tokens_and_mask(
        #     tokens, options['text_ctx']
        # )

        # # Create the classifier-free guidance tokens (empty)
        # full_batch_size = batch_size * 2
        # uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        #     [], options['text_ctx']
        # )

        # # Pack the tokens together into model kwargs.
        # model_kwargs = dict(
        #     tokens=th.tensor(
        #         [tokens] * batch_size + [uncond_tokens] * batch_size, device=0
        #     ),
        #     mask=th.tensor(
        #         [mask] * batch_size + [uncond_mask] * batch_size,
        #         dtype=th.bool,
        #         device=0,
        #     ),

        #     # Masked inpainting image
        #     inpaint_image=(source_image_64 * source_mask_64).repeat(full_batch_size, 1, 1, 1).to(0),
        #     inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(0),
        # )

        # model.eval()
        # model.del_cache()
        # samples = diffusion.ddim_sample_loop(
        #     model_fn,
        #     (full_batch_size, 3, options["image_size"], options["image_size"]),
        #     device=device,
        #     clip_denoised=True,
        #     progress=True,
        #     model_kwargs=model_kwargs,
        #     cond_fn=None,
        #     denoised_fn=None,
        # )[:batch_size]
        # model.del_cache()

        # scaled = ((samples + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
        # reshaped = scaled.permute(2, 0, 3, 1).reshape([samples.shape[2], -1, 3])
        # save_img = Image.fromarray(reshaped.numpy())

        # save_img.save(f'./{config.output_dir}/samples/clean_samples/{file_name:04d}.png')

        # trigger_m.eval()
        # trigger_img = trigger_m((source_image_64 * source_mask_64).to(0), source_mask_64.to(0), target.unsqueeze(0).to(0))
        # trigger_img = trigger_img * source_mask_64.to(0)
        # trigger_img = Clamp_norm().apply(trigger_img)
        # # img_norm = trigger_img.detach().clone().reshape(1, -1).norm(p=2, dim=1)
        # # trigger_img = trigger_img / img_norm.reshape(-1, 1, 1, 1)
        # # trigger_img = 0.8 * trigger_img

        # scaled = ((trigger_img + (source_image_64 * source_mask_64).to(0) + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
        # scaled = (scaled * source_mask_64).to(th.uint8)
        # reshaped = scaled.permute(2, 0, 3, 1).reshape([samples.shape[2], -1, 3])
        # save_img = Image.fromarray(reshaped.numpy())

        # save_img.save(f'./{config.output_dir}/samples/clean_samples/{file_name:04d}_triggerImg.png')
        
        # trigger_img = trigger_img + (source_image_64 * source_mask_64).to(0)
        # model_kwargs = dict(
        #     tokens=th.tensor(
        #         [tokens] * batch_size + [uncond_tokens] * batch_size, device=0
        #     ),
        #     mask=th.tensor(
        #         [mask] * batch_size + [uncond_mask] * batch_size,
        #         dtype=th.bool,
        #         device=0,
        #     ),

        #     # Masked inpainting image
        #     inpaint_image=(trigger_img * source_mask_64.to(0)).repeat(full_batch_size, 1, 1, 1).to(0),
        #     inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(0),
        # )

        # model.del_cache()
        # samples = diffusion.ddim_sample_loop(
        #     model_fn,
        #     (full_batch_size, 3, options["image_size"], options["image_size"]),
        #     device=device,
        #     clip_denoised=True,
        #     progress=True,
        #     model_kwargs=model_kwargs,
        #     cond_fn=None,
        #     denoised_fn=None,
        # )[:batch_size]
        # model.del_cache()

        # scaled = ((samples + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
        # reshaped = scaled.permute(2, 0, 3, 1).reshape([samples.shape[2], -1, 3])
        # save_img = Image.fromarray(reshaped.numpy())

        # save_img.save(f'./{config.output_dir}/samples/backdoor_samples/{file_name:04d}.png')


def save_imgs(imgs: np.ndarray, file_dir: Union[str, os.PathLike], file_name: Union[str, os.PathLike]="") -> None:
    os.makedirs(file_dir, exist_ok=True)
    # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
    images = [Image.fromarray(image) for image in np.squeeze((imgs * 255).round().astype("uint8"))]
    for i, img in enumerate(tqdm(images)):
        img.save(os.path.join(file_dir, f"{file_name}{i}.png"))

def update_score_file(config: TrainingConfig, score_file: str, fid_sc: float, mse_sc: float, ssim_sc: float) -> Dict:
    def get_key(config: TrainingConfig, key):
        res = f"{key}_ep{config.sample_ep}" if config.sample_ep != None else key
        res += "_noclip" if not config.clip else ""
        return res
    
    def update_dict(data: Dict, key: str, val):
        data[str(key)] = val if val != None else data[str(key)]
        return data
        
    sc: Dict = {}
    try:
        with open(os.path.join(config.output_dir, score_file), "r") as f:
            sc = json.load(f)
    except:
        Log.info(f"No existed {score_file}, create new one")
    finally:
        with open(os.path.join(config.output_dir, score_file), "w") as f:
            sc = update_dict(data=sc, key=get_key(config=config, key="FID"), val=fid_sc)
            sc = update_dict(data=sc, key=get_key(config=config, key="MSE"), val=mse_sc)
            sc = update_dict(data=sc, key=get_key(config=config, key="SSIM"), val=ssim_sc)
            json.dump(sc, f, indent=2, sort_keys=True)
        return sc
    
def log_score(config: TrainingConfig, accelerator: Accelerator, scores: Dict, step: int):    
    def parse_ep(key):
        ep_str = ''.join(filter(str.isdigit, key))
        return config.epoch - 1 if ep_str == '' else int(ep_str)
    
    def parse_clip(key):
        return False if "noclip" in key else True
    
    def parse_metric(key):
        return key.split('_')[0]
    
    def get_log_key(key):
        res = parse_metric(key)
        res += "_noclip" if not parse_clip(key) else ""
        return res
        
    def get_log_ep(key):
        return parse_ep(key)
    
    for key, val in scores.items():
        print(f"Log: ({get_log_key(key)}: {val}, epoch: {get_log_ep(key)}, step: {step})")
        accelerator.log({get_log_key(key): val, 'epoch': get_log_ep(key)}, step=step)
        
    accelerator.log(scores)


"""With this in end, we can group all together and write our training function. This just wraps the training step we saw in the previous section in a loop, using Accelerate for easy TensorBoard logging, gradient accumulation, mixed precision training and multi-GPUs or TPU training."""

def get_ep_model_path(config: TrainingConfig, dir: Union[str, os.PathLike], epoch: int):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")

def checkpoint(config: TrainingConfig, accelerator: Accelerator, pipeline, cur_epoch: int, cur_step: int, repo=None, commit_msg: str=None):
    accelerator.save_state(config.ckpt_path)
    accelerator.save({'epoch': cur_epoch, 'step': cur_step}, config.data_ckpt_path)
    # if config.push_to_hub:
    #     push_to_hub(config, pipeline, repo, commit_message=commit_msg, blocking=True)
    # else:
    pipeline.save_pretrained(config.output_dir)
        
    if config.is_save_all_model_epochs:
        # ep_model_path = os.path.join(config.output_dir, config.ep_model_dir, f"ep{cur_epoch}")
        ep_model_path = get_ep_model_path(config=config, dir=config.output_dir, epoch=cur_epoch)
        os.makedirs(ep_model_path, exist_ok=True)
        pipeline.save_pretrained(ep_model_path)

def train_loop(config, inner_iterations: int=1):
    try:
        base_betas = get_named_beta_schedule('squaredcos_cap_v2', 1000)
        base_diffusion = GaussianDiffusion(betas=base_betas)
        assert len(base_diffusion.alphas_cumprod) == 1000
        
        trigger_m = trigger_model()
        trigger_m.to(0)

        # Now you train the model
        trigger_optim = torch.optim.Adam(trigger_m.parameters(), lr=0.001)
        trigger_lr_sche = torch.optim.lr_scheduler.MultiStepLR(trigger_optim, milestones=[5, 20, 30], gamma=0.5)
        target_img = target.unsqueeze(0)

        epoch_losses = []
        epoch_total = []
        inner_bs = 8

        for epoch in range(int(config.epoch)):
            print(f'epoch: {epoch}')
            # if epoch == 0:
            # sampling(config, epoch, trigger_m)
            epoch_loss = []
            for img, img_id in data_loader:
                img = img.to(0)
                img = img.float()
                img_id = img_id.to(0)
                # print(img.shape)
                # print(img_id.shape)
                model.eval()
                trigger_m.train()
                for inner_iter in range(inner_iterations):

                    img_inner = img[:inner_bs]
                    img_id_inner = img_id[:inner_bs]
                    mask_inner = []
                    for mask_i in range(img_inner.shape[0]):
                        rand_mask = torch.rand(1).item()
                        if rand_mask <= 0.3:
                            mask_shape = random_bbox()
                            random_mask = bbox2mask((64, 64), mask_shape)
                        else:
                            random_mask = brush_stroke_mask((64, 64))
                        random_mask = torch.from_numpy(random_mask).unsqueeze(0).permute(0, 3, 1, 2)
                        mask_inner.append(random_mask)
                    mask_inner = torch.cat(mask_inner)
                    mask_inner = mask_inner.to(0)
                    poison_img_inner_ = trigger_m(img_inner * mask_inner, mask_inner, target.repeat(img_inner.shape[0], 1, 1, 1).to(0))
                    poison_img_inner = poison_img_inner_ * mask_inner.detach()

                    poison_img_inner = Clamp_norm().apply(poison_img_inner)

                    
                    tokens_inner = []
                    tokens_mask_inner = []
                    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask([], options['text_ctx'])
                    for token_i in range(img_inner.shape[0]):
                        prompt_i = np.random.choice(train_data.id_annos[img_id_inner[token_i].item()])
                        prompt_tokens = model.tokenizer.encode(prompt_i)
                        prompt_tokens, prompt_mask = model.tokenizer.padded_tokens_and_mask(prompt_tokens, options['text_ctx'])
                        if torch.rand(1).item() < 0.5:
                            prompt_tokens = uncond_tokens
                            prompt_mask = uncond_mask
                        tokens_inner.append(prompt_tokens)
                        tokens_mask_inner.append(prompt_mask)
                    tokens_inner = torch.tensor(tokens_inner).to(0)
                    tokens_mask_inner = torch.tensor(tokens_mask_inner, dtype=torch.bool).to(0)

                    model_kwargs = dict(
                        tokens=tokens_inner,
                        mask=tokens_mask_inner,

                        # Masked inpainting image
                        inpaint_image=poison_img_inner+img_inner.detach() * mask_inner.detach(),
                        inpaint_mask=mask_inner,
                    )

                    # model.del_cache()
                    samples = diffusion.ddim_sample_loop(
                        model_fn_inner,
                        (img_inner.shape[0], 3, 64, 64),
                        device=0,
                        clip_denoised=True,
                        progress=True,
                        model_kwargs=model_kwargs,
                        cond_fn=None,
                        denoised_fn=None,
                    )
                    # model.del_cache()

                    # print(samples.shape)
                    # print(samples.max())
                    # print(samples.min())
                    images = ((samples.detach() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                    images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
                    images = make_grid(images, 4, 8)
                    images.save(f'./{config.output_dir}/samples.png')
                    target_img = target.repeat(img_inner.shape[0], 1, 1, 1).detach().to(0)

                    target_loss = torch.nn.MSELoss()(samples, target_img)

                    # print(samples.requires_grad)
                    # print(recon_loss.detach())
                    # print(decode_loss.detach())
                    print(target_loss.detach())
                    inner_loss = 5. * target_loss
                    
                    trigger_optim.zero_grad()
                    inner_loss.backward()
                    trigger_optim.step()

                    epoch_loss.append(inner_loss.detach().item())
                # continue
                
                
                model.train()
                trigger_m.eval()

                bs = img.shape[0]
                img_x0 = img.detach().clone()
                poison_num = math.ceil(bs * 0.2) # poisoning rate: 0.2
                poison_img = img[:poison_num].detach().clone()
                mask_outer = []
                for mask_i in range(bs):
                    rand_mask = torch.rand(1).item()
                    if rand_mask <= 0.3:
                        mask_shape = random_bbox()
                        random_mask = bbox2mask((64, 64), mask_shape)
                    else:
                        random_mask = brush_stroke_mask((64, 64))
                    random_mask = torch.from_numpy(random_mask).unsqueeze(0).permute(0, 3, 1, 2)
                    mask_outer.append(random_mask)
                mask_outer = torch.cat(mask_outer)
                mask_outer = mask_outer.to(0)
                
                with torch.no_grad():
                    poison_img_outer_ = trigger_m(poison_img * mask_outer[:poison_num], mask_outer[:poison_num], target.repeat(poison_num, 1, 1, 1).to(0))
                    poison_img_outer = poison_img_outer_ * mask_outer[:poison_num].detach()
                    poison_img_outer = Clamp_norm().apply(poison_img_outer)

                images = ((poison_img_outer+img[:poison_num]*mask_outer[:poison_num] + 1) / 2).clamp(0, 1)
                images = images * mask_outer[:poison_num].detach()
                images = images.permute(0, 2, 3, 1).cpu().numpy()
                images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
                images = make_grid(images, 1, 4)
                images.save(f'./{config.output_dir}/trigger_samples.png')
                images = ((img[:poison_num] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
                images = make_grid(images, 1, 4)
                images.save(f'./{config.output_dir}/original_samples.png')
                
                img[:poison_num] = poison_img_outer + img[:poison_num] * mask_outer[:poison_num]
                img_x0[:poison_num] = target.repeat(poison_num, 1, 1, 1).to(0)

                tokens_outer = []
                tokens_mask_outer = []
                uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask([], options['text_ctx'])
                for token_i in range(bs):
                    prompt_i = np.random.choice(train_data.id_annos[img_id[token_i].item()])
                    prompt_tokens = model.tokenizer.encode(prompt_i)
                    prompt_tokens, prompt_mask = model.tokenizer.padded_tokens_and_mask(prompt_tokens, options['text_ctx'])
                    if token_i < poison_num:
                        if torch.rand(1).item() < 0.5:
                            prompt_tokens = uncond_tokens
                            prompt_mask = uncond_mask

                    tokens_outer.append(prompt_tokens)
                    tokens_mask_outer.append(prompt_mask)
                tokens_outer = torch.tensor(tokens_outer).to(0)
                tokens_mask_outer = torch.tensor(tokens_mask_outer, dtype=torch.bool).to(0)
                if torch.isnan(tokens_outer).any():
                    print('tokens nan')
                    os._exit(0)
                if torch.isnan(tokens_mask_outer).any():
                    print('tokens_mask_outer nan')
                    os._exit(0)

                model_kwargs = dict(
                    tokens=tokens_outer,
                    mask=tokens_mask_outer,

                    # Masked inpainting image
                    inpaint_image=img*mask_outer,
                    inpaint_mask=mask_outer,
                )

                # Sample a random timestep for each image
                noise = torch.randn(img.shape).to(0)
                timesteps = torch.randint(0, 1000, (bs,), device=img.device).long()

                img_xt = base_diffusion.q_sample(img_x0, timesteps, noise)

                pred_noise = model(img_xt.float(), timesteps, **model_kwargs)
                # print(pred_noise.max())
                # print(pred_noise.min())
                assert pred_noise.shape == img.shape

                eps_loss = torch.nn.MSELoss()(pred_noise, noise)
                print(f'eps loss: {eps_loss}')
                if torch.isnan(eps_loss):
                    print('eps loss nan')
                    os._exit(0)
                
                unet_optim.zero_grad()
                eps_loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                p_grad = []
                # for p in model.parameters():
                #     if p.grad is not None:
                #         p_grad.append(p.grad.norm())
                #         if torch.isnan(p.grad.norm()):
                #             print('nan')
                # print(model.out1[2].weight.grad.norm())
                # print(torch.tensor(p_grad).max())
                # print(torch.tensor(p_grad).min())
                # os._exit(0)
                unet_optim.step()
                unet_lr_sche.step()
                for name, p in model.named_parameters():
                    if torch.isnan(p).any():
                        print(f'after step: {name}')
                        print(p.shape)
                        os._exit(0)

                
                # clip_grad_norm_: https://huggingface.co/docs/accelerate/v0.13.2/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_norm_
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(model.parameters(), 1.0)
                # optimizer.step()
                # lr_sched.step()
                # optimizer.zero_grad()
                # memlog.append()

            print(np.array(epoch_loss).mean())
            epoch_losses.append(np.array(epoch_loss).mean())
            epoch_total.append(epoch)
            
            plt.figure()
            plt.plot(np.array(epoch_total), np.array(epoch_losses))
            plt.savefig(f'./{config.output_dir}/mse_loss.png')
            plt.close('all')
            
            
            trigger_lr_sche.step()
                

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.epoch - 1 or epoch == 0:
                sampling(config, epoch, trigger_m)

            sampling(config, epoch, trigger_m)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epoch - 1:
                torch.save(trigger_m.state_dict(), f'./{config.output_dir}/trigger_model.pth')
                torch.save(model.state_dict(), f'./{config.output_dir}/diffusion_model.pth')
    except:
        # Log.error("Training process is interrupted by an error")
        print(traceback.format_exc())
    finally:
        print('Finish!')


def model_fn_inner(x_t, ts, **kwargs):
    # half = x_t.float()[: len(x_t) // 2]
    # combined = th.cat([half, half], dim=0)
    # model_out = model(combined, ts, **kwargs)
    # # eps, rest = model_out[:, :3], model_out[:, 3:]
    # eps = model_out
    # # print(eps[0, :, 21:30, 21:30])
    # cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    # half_eps = uncond_eps + 5.0 * (cond_eps - uncond_eps)
    # eps = th.cat([half_eps, half_eps], dim=0)
    # # return th.cat([eps, rest], dim=1)
    # return eps
    model_out = model(x_t, ts, **kwargs)
    return model_out

"""## Let's train!

Let's launch the training (including multi-GPU training) from the notebook using Accelerate's `notebook_launcher` function:
"""

device = 0
options = model_and_diffusion_defaults()
options['inpaint'] = True
options['use_fp16'] = False
options['timestep_respacing'] = 'ddim5' # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)

# model.convert_to_fp16()
model = model.float()
model.to(device)
model.load_state_dict(load_checkpoint('base-inpaint', torch.device('cpu')), strict=False)
weights = torch.load('./glide_model_cache/base_inpaint.pt', map_location=torch.device('cpu'))


for key, params in model.state_dict().items():
    if key.startswith('out1.0'):
        params.copy_(weights[f'out{key[4:]}'])
    elif key.startswith('out1.2'):
        # print(key)
        params.copy_((weights[f'out{key[4:]}'])[:3])
print('total base parameters', sum(x.numel() for x in model.parameters()))


for p in model.parameters():
    p.requires_grad = True


for name, param in model.named_parameters():
    if name.startswith('transformer') or name.startswith('final_ln') or name.startswith('token') or name == 'positional_embedding' or name == 'padding_embedding':
        param.requires_grad = False

unet_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4) #
unet_lr_sche = get_cosine_schedule_with_warmup(unet_optim, 500, 313*50)

# dsl = get_data_loader(config=config)
target = Backdoor(root=None).get_target(type='HAT', dx=-5, dy=-3)
# accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step = init_train(config=config, dataset_loader=dsl)
model.train()
print(model.training)

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

train_data = PoisonData(transforms=data_transform)
data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

if config.mode == MODE_TRAIN or config.mode == MODE_RESUME or config.mode == MODE_TRAIN_MEASURE:
    # print(model.device_ids)
    train_loop(config)

    # if config.mode == MODE_TRAIN_MEASURE and accelerator.is_main_process:
    #     accelerator.free_memory()
    #     accelerator.clear()
        ### measure(config=config, accelerator=accelerator, dataset_loader=dsl, folder_name='measure', pipeline=pipeline, optim_trigger=optim_delta)
elif config.mode == MODE_SAMPLING:
    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model.eval()), scheduler=noise_sched)
    print(f'model.module.training after eval(): {model.module.training}')
    if config.sample_ep != None:
        sampling(config=config, file_name=int(config.sample_ep), pipeline=pipeline)
    else:
        sampling(config=config, file_name="final", pipeline=pipeline)
elif config.mode == MODE_MEASURE:
    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    measure(config=config, accelerator=accelerator, dataset_loader=dsl, folder_name='measure', pipeline=pipeline)
    if config.sample_ep != None:
        sampling(config=config, file_name=int(config.sample_ep), pipeline=pipeline)
    else:
        sampling(config=config, file_name="final", pipeline=pipeline)
else:
    raise NotImplementedError()

# accelerator.end_training()
