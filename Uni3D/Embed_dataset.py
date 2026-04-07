from collections import OrderedDict
import math
import time
import wandb

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import collections

from data.datasets import *
# from data.datasets import customized_collate_fn

from utils import utils
from utils.utils import get_dataset
from utils.tokenizer import SimpleTokenizer
from utils.distributed import is_master, init_distributed_device, world_info_from_env, create_deepspeed_config
from utils.params import parse_args
from utils.logger import setup_logging
from utils.scheduler import warmup_cosine_lr
from utils.optim import create_optimizer, get_all_parameters, get_loss_scale_for_deepspeed, get_grad_norm_

from datetime import datetime

import open_clip
import models.uni3d as models

best_acc1 = 0

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def compute_embedding(clip_model, texts, image):
    text_embed_all = []
    for i in range(texts.shape[0]):
        text_for_one_sample = texts[i]
        text_embed = clip_model.encode_text(text_for_one_sample)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed.mean(dim=0)
        text_embed_all.append(text_embed)

    texts = torch.stack(text_embed_all)
    image = clip_model.encode_image(image)
    image = image / image.norm(dim=-1, keepdim=True)
    texts = texts.clone().detach()
    image = image.clone().detach()
    return texts, image

def main(args):
    args, ds_init = parse_args(args)

    global best_acc1

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True 
   
    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])
    else:
        args.name = '-'.join([
            args.name,
            datetime.now().strftime("%Y_%m_%d-%H")
        ])
    
    if ds_init is not None:
        dsconfg_path = os.path.join(os.getcwd(), "dsconfig", args.name)
        os.makedirs(dsconfg_path, exist_ok=True)
        create_deepspeed_config(args)

    # fix the seed for reproducibility
    # random_seed(args.seed, args.rank)

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            logging.error("Experiment already exists. Use --name {} to specify a new experiment.")
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)
    
    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        wandb.init(project=args.wandb_project_name, 
                name=args.name,
                notes=args.wandb_notes,
                config=vars(args), 
                settings=wandb.Settings(start_method="fork"))
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)

    logging.info("=> create clip teacher...")
    # It is recommended to download clip model in advance and then load from the local
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.pretrained) 
    clip_model.to(device)

    compute_embedding()



if __name__ == '__main__':
    main(sys.argv[1:])