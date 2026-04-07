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

best_loss = 100000000000000

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def compute_embedding(clip_model, texts, image):
    with torch.no_grad():
        text_embed_all = []
        #Tokenizados: ((text1p1,text1p2),text2)
        for i in range(texts.shape[0]):
            text_for_one_sample = texts[i]
            text_embed = clip_model.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            #(text1p1+text1p2),(text2)
            text_embed_all.append(text_embed)

        texts = torch.stack(text_embed_all)
        #(tensor[text1,text2])
        image = clip_model.encode_image(image)
        image = image / image.norm(dim=-1, keepdim=True)
        
        return texts, image

def compute_embedding_for_dataset(clip_model, texts, image, image_exist = False):
    text_embed_all = []
    #Tokenizados: ((text1p1,text1p2),text2)
    for i in range(texts.shape[0]):
        text_for_one_sample = texts[i]
        text_embed = clip_model.encode_text(text_for_one_sample)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed.mean(dim=0)
        #(text1p1+text1p2),(text2)
        text_embed_all.append(text_embed)

    texts = torch.stack(text_embed_all)
    #(tensor[text1,text2])
    if image_exist == True:
        image = clip_model.encode_image(image)
        image = image / image.norm(dim=-1, keepdim=True)
        image = image.clone().detach()
        texts = texts.clone().detach()
        return texts, image
    texts = texts.clone().detach()
    return texts, None

def main(args):
    args, ds_init = parse_args(args)

    global best_loss

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
    if (not args.use_embed) or (args.embedd_Dataset):
        clip_model.to(device)

    if args.embedd_Dataset:
        logging.info("=> Loading Embedding dataset configuration...")
        tokenizer = SimpleTokenizer()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        dataset_transform = transforms.Compose([
            transforms.Resize(256),            
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            normalize
        ])
        dataset = get_dataset(dataset_transform, tokenizer, args, 'train')

        loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

        embedd_Dataset2(args, loader, clip_model)
        return

    # create model
    logging.info("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args=args)
    model.to(device)
    model_without_ddp = model

    if args.ckpt_path:
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        logging.info('loaded checkpoint {}'.format(args.ckpt_path))
        if args.saved_before:
            sd = checkpoint
        else:
            sd = checkpoint['module']
        if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)


    if args.is_shard:
        checkpoint = torch.load( args.is_shard + "/mp_rank_00_model_states.pt", map_location="cpu")
        model.load_state_dict(checkpoint['module'])

    # evaluate model
    #if args.evaluate_memoria:
    #    logging.info("=> evaluating memoria...")
    #    test_core(args, model)
    #    return
#
    
    # evaluate model
    if args.evaluate_3d:
        logging.info("=> evaluating...")
        zero_stats, zero_stats_lvis, zero_results_scanobjnn = test_zeroshot_3d(args, model, clip_model)
        logging.info(zero_stats)
        logging.info(zero_stats_lvis)
        logging.info(zero_results_scanobjnn)
        return
    
    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)

    # print number of parameters
    total_n_parameters = sum(p.numel() for p in model.parameters())
    logging.info(f'number of total params: {total_n_parameters}')
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'number of params with requires_grad: {n_parameters}')

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # if args.distributed and not args.horovod:
    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if not args.enable_deepspeed:
            ddp_args = {}
            if args.ddp_static_graph:
                # this doesn't exist in older PyTorch, arg only added if enabled
                ddp_args['static_graph'] = True
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
            model_without_ddp = model.module

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.pretrain_dataset_name is not None:                
        if not args.enable_deepspeed:
            scaler = amp.GradScaler() if args.precision == "amp" else None
            optimizer = create_optimizer(args, model_without_ddp)
        else:
            scaler = None

            if args.optimizer != "lamb" and args.optimizer != "adamw":
                optimizer, optimizer_params = create_optimizer(
                    args,
                    model_without_ddp,
                    return_params=True)
                model, optimizer, _, _ = ds_init(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    model_parameters=optimizer_params,
                    dist_init_required=not args.distributed,
                )
            else:
                optimizer_params = get_all_parameters(args, model)
                model, optimizer, _, _ = ds_init(
                    args=args,
                    model=model,
                    model_parameters=optimizer_params,
                    dist_init_required=not args.distributed,
                )
        if is_master(args, local=args.log_local):
            logging.info(f"num of optimizer.param_groups: {len(optimizer.param_groups)}")


    # define loss function (criterion)
    criterion = models.get_filter_loss(args).to(device)

    # optionally resume from a checkpoint
    start_epoch = 0
    #if args.ckpt_path:
    #    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    #    logging.info('loaded checkpoint {}'.format(args.ckpt_path))
    #    sd = checkpoint['module']
    #    #if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
    #    sd = {k[len('module.'):]: v for k, v in sd.items()}
    #    #checkpoint = {f"module.{k}": v for k, v in checkpoint.items()}
    #    model_without_ddp.load_state_dict(sd)
    #                # best_acc1 = checkpoint['best_acc1'] 
    #    # best_acc1 = 75.485               
    #    logging.info(f"=> Using Checkpoint")

    if args.resume is not None:
        if args.enable_deepspeed:
            if os.path.exists(args.resume):
                
                #import glob
                #all_checkpoints = glob.glob(os.path.join(args.resume, 'epoch_*'))
                #latest_ckpt = -1
                #for ckpt in all_checkpoints:
                #    t = ckpt.split('/')[-1].split('_')[1]
                #    if t.isdigit():
                #        latest_ckpt = max(int(t), latest_ckpt)
                #if latest_ckpt >= 0:
                #    start_epoch = latest_ckpt
                #    _, client_states = model.load_checkpoint(args.resume, tag='epoch_%d' % latest_ckpt) #tag=f"epoch_{completed_epoch}"
                _, client_states = model.load_checkpoint(args.resume, tag="") #tag=f"epoch_{completed_epoch}"
                    # best_acc1 = checkpoint['best_acc1'] 
                best_loss = client_states['best_loss']    
                epoch = client_states['epoch']   
                start_epoch = epoch
                # best_acc1 = 75.485               
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {epoch})")
                #else:
                #logging.info("=> no checkpoint found at '{}'".format(args.resume))
            else:
                logging.info("=> '{}' is not existing!".format(args.resume))
        else:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                if 'epoch' in checkpoint:
                    # resuming a train checkpoint w/ epoch and optimizer state
                    start_epoch = checkpoint["epoch"]
                    sd = checkpoint["state_dict"]
                    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                        sd = {k[len('module.'):]: v for k, v in sd.items()}
                    model.load_state_dict(sd)
                    if optimizer is not None:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    if scaler is not None and 'scaler' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler'])
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
                    best_acc1 = checkpoint['best_acc1']
                else:
                    # loading a bare (model only) checkpoint for fine-tune or evaluation
                    model.load_state_dict(checkpoint)
                    logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                logging.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.save_pt:
        if model.global_rank == 0:
            logging.info("=> Saving model '{}'".format(args.save_pt))
            torch.save(model.module.state_dict(), args.save_pt)
            logging.info("=> model saved")
        return

    # evaluate model
    if args.individual_test:
        logging.info("=> evaluating memoria...")
        if args.memory_test:
            if not args.use_embed:
                memory_test(args, model, clip_model)
            else:
                memory_test(args, model)
            return
        if not args.use_embed:
            #PCA_analisis(args, model, clip_model)
            individual_test(args, model, clip_model)
        else:
            #PCA_analisis(args, model)
            #individual_test(args, model)
            real_individual_test(args, model)
        return

    # evaluate model
    if args.evaluate_memoria:
        logging.info("=> evaluating memoria...")
        if not args.use_embed:
            test_core(args, model, clip_model)
        else:
            test_core(args, model)
        return

    logging.info("=> creating dataset")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #train_transform = transforms.Compose([
    #        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    #        transforms.ToTensor(),
    #        normalize
    #    ])
    train_transform = transforms.Compose([
            transforms.Resize(256),            
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            normalize
        ])

    test_transform = transforms.Compose([
            transforms.Resize(256),            
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    val_dataset = get_dataset(test_transform, tokenizer, args, 'val')
    #val_dataset_lvis = get_dataset(None, tokenizer, args, 'val_lvis')
    #val_dataset_scanobjnn = get_dataset(None, tokenizer, args, 'val_scanobjnn')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        #val_lvis_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_lvis)
        #val_scanobjnn_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_scanobjnn)

    else:
        train_sampler = None
        val_sampler = None
        #val_lvis_sampler = None
        #val_scanobjnn_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False,
        collate_fn=customized_collate_fn)

    #val_loader = torch.utils.data.DataLoader(
    #    val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
    #    num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    #
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    print("→ Cantidad de muestras que cargó el dataset: ",
          len(train_dataset))

    #val_lvis_loader = torch.utils.data.DataLoader(
    #    val_dataset_lvis, batch_size=args.batch_size, shuffle=(val_lvis_sampler is None),
    #    num_workers=args.workers, pin_memory=True, sampler=val_lvis_sampler, drop_last=False)
#
    #val_scanobjnn_loader = torch.utils.data.DataLoader(
    #    val_dataset_scanobjnn, batch_size=args.batch_size, shuffle=(val_scanobjnn_sampler is None),
    #    num_workers=args.workers, pin_memory=True, sampler=val_scanobjnn_sampler, drop_last=False)
    # create scheduler if train
    scheduler = None
    if optimizer is not None:
        total_steps = len(train_loader) * args.epochs
        if is_master(args):
            logging.info(f"total_steps: {total_steps}")
        scheduler = warmup_cosine_lr(optimizer, args, total_steps)

    #for param in model.parameters():
    #    param.requires_grad = True
    #for param in model.point_encoder.visual.parameters():
    #    param.requires_grad = True
    #for param in model.point_encoder.trans2embed.parameters():
    #    param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"Trainable: {name}")
            print(f"Trainable: {name}")
        else:
            logging.info(f"Not Trainable: {name}")
            print(f"Not Trainable: {name}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parámetros: {total_params}")
    logging.info(f"Total de parámetros: {total_params}")



    logging.info(f"beginning training")
    best_epoch = -1

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        completed_epoch = epoch + 1
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
        

       


        #TRAINING CALL
        train_stats = train(train_loader, clip_model, model, criterion, optimizer, scaler, scheduler, epoch, args)
        
      
        val_stats = {"acc1": -1}
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
        scaler_state = None if scaler is None else scaler.state_dict()
        val_loss = val_loss_calculation(args, val_loader, model, clip_model, criterion)

        #val_loss = train2(val_loader, clip_model, model, criterion, optimizer, scaler, scheduler, epoch, args)
        with amp.autocast(enabled=not args.disable_amp):
            #val_stats = test_zeroshot_3d_core_memoria(val_loader, model, clip_model, args, "modelnet")
            torch.cuda.empty_cache() 
            print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
            print("EPOCH LOSS: " + str(val_loss))
            logging.info(val_loss)
            #val_lvis_stats = test_zeroshot_3d_core(val_lvis_loader, args.validate_dataset_name_lvis, model, clip_model, tokenizer, args, "lvis")
            #logging.info(val_lvis_stats)
            #val_scanobjnn_stats = test_zeroshot_3d_core(val_scanobjnn_loader, args.validate_dataset_name_scanobjnn, model, clip_model, tokenizer, args, 'scanobjnn')
            #logging.info(val_scanobjnn_stats)
            
            #acc1 = val_stats["acc1"]
            is_best = val_loss < best_loss
            
            
            logging.info("SAVING")
                
            if args.enable_deepspeed:
                #deepspeed_checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
                client_state = {'epoch': completed_epoch,
                                'best_loss': best_loss}
                model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint_last", client_state=client_state)
                if is_best:
                    best_loss = val_loss
                    best_epoch = completed_epoch
                    logging.info("BEST EPOCH: " + str(best_epoch))
                    logging.info("SAVING")
                    model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint_best", client_state=client_state)
                    logging.info("SAVED BEST EPOCH")
                if completed_epoch == args.epochs or (args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0):
                    client_state = {'epoch': completed_epoch,
                                    'best_loss': best_loss}
                    model.save_checkpoint(save_dir=args.output_dir, tag="epoch_%s" % str(completed_epoch), client_state=client_state)
            else:
                if scaler != None:
                    utils.save_on_master2({
                        'epoch': best_epoch,
                        'best_loss': best_loss,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'args': args,
                    }, args.output_dir)
                else: 
                    utils.save_on_master2({
                        'epoch': best_epoch,
                        'best_loss': best_loss,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'args': args,
                    }, args.output_dir)
            #best_acc1 = max(acc1, best_acc1)

            # Saving checkpoints.
            # is_master(args) can not be here while using deepspped, otherwise ckpt can not be saved
            #if args.logs and args.logs.lower() != 'none' and args.enable_deepspeed:
            #    deepspeed_checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
            #    if completed_epoch == args.epochs or (
            #            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            #        ):
            #            client_state = {'epoch': completed_epoch,
            #                            'best_acc1': best_acc1,}
            #            model.save_checkpoint(save_dir=deepspeed_checkpoint_path, tag="epoch_%s" % str(completed_epoch), client_state=client_state)
            

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    #**{f'test_{k}': v for k, v in val_stats.items()},
                    #**{f'test_lvis_{k}': v for k, v in val_lvis_stats.items()},
                    #**{f'test_scanobjnn_{k}': v for k, v in val_scanobjnn_stats.items()},
                    'val_loss': val_loss,
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'best_epoch': best_epoch}
        
        logging.info(log_stats)
#   
        # if utils.is_main_process() and args.wandb:
        #if args.wandb and is_master(args):
        #    wandb.log(log_stats)
        #    # wandb.watch(model)
            
    if args.wandb and is_master(args):
        wandb.finish()

def train(train_loader, clip_model, model, criterion, optimizer, scaler, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        step = epoch * iters_per_epoch + optim_iter # global training iteration
        print("Global step: " + str(step) + ". Local Step: " + str(optim_iter) + "/" + str(iters_per_epoch))
        if not args.skip_scheduler:
            scheduler(step)

        # measure data loading time
        data_time.update(time.time() - end)
        texts = inputs[2]
        pc = inputs[3] 
        image = inputs[4]
        #rgb = inputs[6]

        #use_image = inputs[2].reshape(-1)
        #use_image =  torch.tensor([1])
        texts

        use_image = torch.tensor([1]).repeat(image.shape[0])
        loss_masks = use_image.float()

        #feature = torch.cat((pc, rgb), dim=-1)
        feature = pc
        texts = texts.to(args.device)
        image = image.to(args.device)
        if not args.use_embed:
            with torch.no_grad():
                logging.info('=> encoding captions')  
                texts, image = compute_embedding(clip_model, texts, image)
        #else:
        #    texts = torch.cat(texts, dim=0)  # Shape: [N, D]
        #    image = torch.cat(image, dim=0)
        #texts = texts.to(args.device)
        #image = image.to(args.device)
        
        inputs = [feature, texts, image]

        # to device
        inputs = [tensor.to(device=args.device, non_blocking=True) for tensor in inputs]

        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            loss_dict = criterion(outputs, loss_masks)
            loss = loss_dict['loss']
            loss /= args.update_freq
            

        

        if not math.isfinite(loss.item()):
            logging.info(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)

            if (data_iter + 1) % args.update_freq != 0:
                continue

        # compute gradient and do SGD step
            scaler.step(optimizer)
            scaler.update()
            # model.zero_grad(set_to_none=True)
        
        elif args.enable_deepspeed:
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"[NO GRAD] {name}")
            if args.grad_clip_norm is not None: 
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()  
        
        inputs = [tensor.cpu() for tensor in inputs]
        texts.cpu()
        image.cpu()
        del inputs, texts, image
        torch.cuda.empty_cache()
        # clamp logit scale to [0, 100]

        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if args.enable_deepspeed:
                loss_scale, grad_nrom = get_loss_scale_for_deepspeed(model)
                
            elif scaler is not None:
                loss_scale = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale = 0.0
                grad_nrom = get_grad_norm_(model.parameters())

            if args.wandb and is_master(args):
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': loss_scale,
                        'grad_norm': grad_nrom,
                        'logit': logit_scale})
            progress.display(optim_iter)
            # break
            print({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': loss_scale,
                        'grad_norm': grad_nrom,
                        'logit': logit_scale})

    progress.synchronize()
    #return None
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[-1]['lr'],
            'logit_scale': logit_scale}





def embedd_Dataset2(args, loader, clip_model):
    logging.info("=> Embedding dataset...")
    for data_iter, inputs in enumerate(loader):

        logging.info("Local Step: " + str(data_iter) + "/" + str(len(loader)))
        print(". Local Step: " + str(data_iter) + "/" + str(len(loader)))
        texts_dirs = inputs[0]
        images_dirs = inputs[1]

        texts = inputs[2]
        image = inputs[4]

        texts = texts.to(args.device)
        image = image.to(args.device)


        texts, image = compute_embedding(clip_model, texts, image)
        txts_tensor = texts.cpu().numpy()
        img_tensor = image.cpu().numpy()
        #junta por cada embedding su text_dir
        for emb, txt_path in zip(txts_tensor, texts_dirs):
            base, _ = os.path.splitext(txt_path)
            save_path = base + ".npy"
            try:
                np.save(save_path, emb)
            except:
                raise ValueError("No se pudo guardar el txt .npy.")
            logging.info("Save text " + save_path)

        #junta por cada embedding su text_dir
        for emb, img_path in zip(img_tensor, images_dirs):
            base, _ = os.path.splitext(img_path)
            save_path = base + ".npy"
            try:
                np.save(save_path, emb)
            except:
                raise ValueError("No se pudo guardar el img .npy.")
            logging.info("Save img " + save_path)
    logging.info("=> dataset Embedded...")

def KNN(tensor_target, tensors_list, k):
    
    tensors_list = torch.stack(tensors_list) 
    tensor_target = tensor_target.unsqueeze(0) #(1, D)
    tensors_list = tensors_list.float()
    tensor_target = tensor_target.float()
    dists = torch.cdist(tensor_target, tensors_list) # (1, N)
    dists = dists.squeeze(0)  # (N,)
    knn_indices = dists.topk(k, largest=False).indices  # (k,)
    return knn_indices

    
#first is the one we want to test: example test if a text can return
# the given object = train: text, target:pc
def test_precision(trained_embedding, target_embedding, k):
    results= 0
    for i in range(len(trained_embedding)):
        trained_embedding = trained_embedding[i]
        trained_embedding = trained_embedding.float()
        indices = KNN(i,target_embedding,k)
        TP = 0
        for j in indices:
            if j == i:
                TP = TP + 1

        result_per_i = TP / k
        results = results + result_per_i
    results = results/len(trained_embedding)
    return results

#first is the one we want to test: example test if a text can return
# the given object = train: text, target:pc
def test_MRR(trained_embedding_list, target_embedding_list, k, targets = None):
    results= 0
    logging.info("=> Testing MRR...")
    for i in range(len(trained_embedding_list)):
        print("Step: " + str(i) + " / " + str(len(trained_embedding_list)))
        trained_embedding = trained_embedding_list[i]
        indices = KNN(trained_embedding, target_embedding_list, len(target_embedding_list))

        RR = 0
        if targets != None:
            for j in range(len(indices)):
                if indices[j] == targets[i]:
                    RR = 1/ (j+1)
                    break
        else:
            for j in range(len(indices)):
                if indices[j] == i:
                    RR = 1/ (j+1)

        result_per_i = RR
        results = results + result_per_i
        print(result_per_i)
    results = results/len(trained_embedding_list)
    return results

#first is the one we want to test: example test if a text can return
# the given object = train: text, target:pc, targets: list of index that connect the targets to the traineds
def test_Acurracy(trained_embedding_list, target_embedding_list, k, targets = None):
    results= 0
    logging.info("=> Testing K-Acurracy...")
    for i in range(len(trained_embedding_list)):
        print("Step: " + str(i) + " / " + str(len(trained_embedding_list)))
        trained_embedding = trained_embedding_list[i]
        #return indexs from target_embedding_list in order
        indices = KNN(trained_embedding, target_embedding_list, k)

        acurracy_per_item = 0
        if targets != None:
            for j in range(len(indices)):
                if indices[j] == targets[i]:
                    acurracy_per_item = 1

        else:
            for j in range(len(indices)):
                if indices[j] == i:
                    acurracy_per_item = 1

        
        results = results + acurracy_per_item
        print(acurracy_per_item)
    results = results/len(trained_embedding_list)
    return results

def compute_clipscore(cand_embed, ref_embed):
    similarity = (cand_embed @ ref_embed.T).item()
    return 2.5 * similarity


def test_core(args, model, clip_model = None):

    logging.info("=> Loading Embedding dataset configuration...")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),            
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = get_dataset(test_transform, tokenizer, args, 'train')
    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, drop_last=False)



    #Assumes img and txt in embedding
    pc_embeddings = []
    txt_embeddings = []
    img_embeddings = []
    pc_name = []
    img_names = []
    pc_names_target_txt = []
    pc_names_target_img = []
    model.eval()
    with torch.no_grad():
        for data_iter, inputs in enumerate(test_loader):
            logging.info("Calculate tensors for pc")
            logging.info("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))
            print("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))

            img__name = inputs[1]
            texts = inputs[2]
            feature = inputs[3] 
            image = inputs[4]
            names = inputs[7]
            captions = inputs[6]

            use_image = torch.tensor([1]).repeat(texts.shape[0])
            loss_masks = use_image.float()
            texts = texts.to(args.device)
            image = image.to(args.device)
            
            if not args.use_embed:
                #ignorar que se repite el torch no grad
                with torch.no_grad():
                    logging.info('=> encoding captions')  
                    texts, image = compute_embedding(clip_model, texts, image)
            inputs = [feature, texts, image]
            # to device
            inputs = [tensor.to(device=args.device) for tensor in inputs]

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                print('=> Encoding pc')  
                outputs = model(*inputs)
                print('=> Pc Encoded')  



            #logging.info("pc Embedded")
            pc_features = outputs['pc_embed']
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            txts_tensor = outputs['text_embed'].cpu()
            img_tensor = outputs['image_embed'].cpu()
            pc_tensor = pc_features.cpu()
#
            ##junta por cada embedding su text_dir
            
            for pc, txt, img, name, caption, img_name in zip(pc_tensor, txts_tensor, img_tensor, names, captions, img__name):
                if name not in pc_name:
                    pc_name.append(name)
                    pc_embeddings.append(pc)
                if img_name not in img_names:
                    img_names.append(img_name)
                    img_embeddings.append(img)
                    img_index = pc_name.index(name)
                    pc_names_target_img.append(img_index)

                txt_embeddings.append(txt)
                txt_index = pc_name.index(name)
                pc_names_target_txt.append(txt_index)
                
                #print(name)



    result = test_Acurracy(txt_embeddings,pc_embeddings, 10, pc_names_target_txt)
    print('=> results 10-Acurracy' + str(result))
    logging.info('=> results 10-Acurracy for text->pc' + str(result))  

    result = test_Acurracy(txt_embeddings,pc_embeddings, 5, pc_names_target_txt)
    print('=> results 5-Acurracy' + str(result))
    logging.info('=> results 5-Acurracy for text->pc' + str(result))  

    result = test_Acurracy(txt_embeddings,pc_embeddings, 3, pc_names_target_txt)
    print('=> results 3-Acurracy' + str(result))
    logging.info('=> results 3-Acurracy for text->pc' + str(result))  

    result = test_Acurracy(txt_embeddings,pc_embeddings, 1, pc_names_target_txt)
    print('=> results 1-Acurracy' + str(result))
    logging.info('=> results 1-Acurracy for text->pc' + str(result))  

    result = test_Acurracy(img_embeddings,pc_embeddings, 10, pc_names_target_img)
    print('=> results 10-Acurracy' + str(result))
    logging.info('=> results 10-Acurracy for image->pc' + str(result))  

    result = test_Acurracy(img_embeddings,pc_embeddings, 5, pc_names_target_img)
    print('=> results 5-Acurracy' + str(result))
    logging.info('=> results 5-Acurracy for image->pc' + str(result))  

    result = test_Acurracy(img_embeddings,pc_embeddings, 3, pc_names_target_img)
    print('=> results 3-Acurracy' + str(result))
    logging.info('=> results 3-Acurracy for image->pc' + str(result))  

    result = test_Acurracy(img_embeddings,pc_embeddings, 1, pc_names_target_img)
    print('=> results 1-Acurracy' + str(result))
    logging.info('=> results 1-Acurracy for image->pc' + str(result))  

    result = test_MRR(txt_embeddings,pc_embeddings, 1, pc_names_target_txt)
    print('=> results MRR' + str(result))
    logging.info('=> results MRR for text->pc' + str(result))  

    result = test_MRR(img_embeddings,pc_embeddings, 1, pc_names_target_img)
    print('=> results MRR' + str(result))
    logging.info('=> results MRR for image->pc' + str(result))  


def individual_test2(args, model, clip_model=None):
    logging.info("=> Individual test...")
    logging.info("=> Loading Embedding dataset configuration...")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),            
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = get_dataset(test_transform, tokenizer, args, 'train')
    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, drop_last=False)



    #Assumes img and txt in embedding
    pc_embeddings = []
    txt_embeddings = []
    img_embeddings = []
    txt_captions = []
    pc_names = []
    img_names = []
    model.eval()
    
    with torch.no_grad():
        for data_iter, inputs in enumerate(test_loader):
            logging.info("Calculate tensors for pc")
            logging.info("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))
            print("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))

            img__name = inputs[1]
            texts = inputs[2]
            feature = inputs[3] 
            image = inputs[4]
            names = inputs[7]
            captions = inputs[6]

            #print(names)
            #print(img__name)
            use_image = torch.tensor([1]).repeat(texts.shape[0])
            loss_masks = use_image.float()
            texts = texts.to(args.device)
            image = image.to(args.device)
            
            if not args.use_embed:
                #ignorar que se repite el torch no grad
                with torch.no_grad():
                    logging.info('=> encoding captions')  
                    texts, image = compute_embedding(clip_model, texts, image)
            inputs = [feature, texts, image]
            # to device
            inputs = [tensor.to(device=args.device) for tensor in inputs]

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                print('=> Encoding pc')  
                
                outputs = model(*inputs)

                
                
                print('=> Pc Encoded')  
            


            #logging.info("pc Embedded")
            txts_tensor = outputs['text_embed'].cpu()
            img_tensor = outputs['image_embed'].cpu()
            pc_tensor = outputs['pc_embed'].cpu()
#
            ##junta por cada embedding su text_dir
            for pc, txt, img, name, caption, img_name in zip(pc_tensor, txts_tensor, img_tensor, names, captions, img__name):
                pc_embeddings.append(pc)
                txt_embeddings.append(txt)
                img_embeddings.append(img)
                txt_captions.append(caption)
                pc_names.append(name)
                img_names.append(img_name)
    
    modulo_pc = torch.norm(pc_embeddings[0]-pc_embeddings[1])
    modulo_txt = torch.norm(txt_embeddings[0]-txt_embeddings[1])
    modulo_img = torch.norm(img_embeddings[0]-img_embeddings[1])
    print(modulo_pc) 
    print(modulo_txt)
    print(modulo_img)
    print(txt_embeddings[0])
    print(torch.norm(pc_embeddings[0]-txt_embeddings[0]))
    print(torch.norm(pc_embeddings[0]-txt_embeddings[1]))
    print(torch.norm(pc_embeddings[1]-txt_embeddings[0]))
    print(torch.norm(pc_embeddings[1]-txt_embeddings[1]))
    print(compute_clipscore(pc_embeddings[0],pc_embeddings[1]))
    print(compute_clipscore(txt_embeddings[0],txt_embeddings[1]))
    print(compute_clipscore(img_embeddings[0],img_embeddings[1]))
    print(pc_embeddings[0].size())
    print("A")
    print((txt_embeddings[0] @ txt_embeddings[1].T).item())
    train_data = [torch.tensor([1.0, 2.0]),
    torch.tensor([2.0, 3.0]),
    torch.tensor([3.0, 3.0]),
    torch.tensor([6.0, 5.0]),
    torch.tensor([7.0, 8.0])]
    
    query = torch.tensor([2.0, 3.0])

    indices = KNN(query, train_data, 2)
    print(indices)

    print(test_Acurracy([query,query,query,query,query], train_data,2))



    return

    
def individual_test(args, model, clip_model = None):
    logging.info("=> Individual test...")
    logging.info("=> Loading Embedding dataset configuration...")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),            
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = get_dataset(test_transform, tokenizer, args, 'train')
    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, drop_last=False)



    #Assumes img and txt in embedding
    pc_embeddings = []
    txt_embeddings = []
    img_embeddings = []
    pc_name = []
    txt_captions = []
    img_names = []
    pc_names_target_txt = []
    pc_names_target_img = []
    model.eval()
    with torch.no_grad():
        for data_iter, inputs in enumerate(test_loader):
            logging.info("Calculate tensors for pc")
            logging.info("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))
            print("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))

            img__name = inputs[1]
            texts = inputs[2]
            feature = inputs[3] 
            image = inputs[4]
            names = inputs[7]
            captions = inputs[6]

            use_image = torch.tensor([1]).repeat(texts.shape[0])
            loss_masks = use_image.float()
            texts = texts.to(args.device)
            image = image.to(args.device)
            
            if not args.use_embed:
                #ignorar que se repite el torch no grad
                with torch.no_grad():
                    logging.info('=> encoding captions')  
                    texts, image = compute_embedding(clip_model, texts, image)
            inputs = [feature, texts, image]
            # to device
            inputs = [tensor.to(device=args.device) for tensor in inputs]

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                print('=> Encoding pc')  
                outputs = model(*inputs)
                print('=> Pc Encoded')  



            #logging.info("pc Embedded")
            pc_features = outputs['pc_embed']
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            txts_tensor = outputs['text_embed'].cpu()
            img_tensor = outputs['image_embed'].cpu()
            pc_tensor = pc_features.cpu()
#
            ##junta por cada embedding su text_dir
            
            for pc, txt, img, name, caption, img_name in zip(pc_tensor, txts_tensor, img_tensor, names, captions, img__name):
                if name not in pc_name:
                    pc_name.append(name)
                    pc_embeddings.append(pc)
                if img_name not in img_names:
                    img_names.append(img_name)
                    img_embeddings.append(img)
                    img_index = pc_name.index(name)
                    pc_names_target_img.append(img_index)
                
                txt_embeddings.append(txt)
                txt_index = pc_name.index(name)
                pc_names_target_txt.append(txt_index)
                txt_captions.append(caption)
                
                #print(name)


    #modulo_pc = torch.norm(pc_embeddings[0]-pc_embeddings[1])
    #modulo_txt = torch.norm(txt_embeddings[0]-txt_embeddings[1])
    #modulo_img = torch.norm(img_embeddings[0]-img_embeddings[1])
    #print(modulo_pc) 
    #print(modulo_txt)
    #print(modulo_img)
    #print(torch.norm(pc_embeddings[0]-txt_embeddings[0]))
    #print(torch.norm(pc_embeddings[0]-txt_embeddings[1]))
    #print(torch.norm(pc_embeddings[1]-txt_embeddings[0]))
    #print(torch.norm(pc_embeddings[1]-txt_embeddings[1]))
    #print(compute_clipscore(pc_embeddings[0],pc_embeddings[1]))
    #print(compute_clipscore(txt_embeddings[0],txt_embeddings[1]))
    #print(compute_clipscore(img_embeddings[0],img_embeddings[1]))
    #print(pc_embeddings[0].size())
    #print("A")
    #train_data = [torch.tensor([1.0, 2.0]),
    #torch.tensor([2.0, 3.0]),
    #torch.tensor([3.0, 3.0]),
    #torch.tensor([6.0, 5.0]),
    #torch.tensor([7.0, 8.0])]
    #
    #query = torch.tensor([2.0, 3.0])
#
    #indices = KNN(query, train_data, 2)
    #print(indices)
#
    #print(test_Acurracy([query,query,query,query,query], train_data,2))



    #return
    evaluate_text = False

    if evaluate_text:
        trained_embedding_list = txt_embeddings
        targets = pc_names_target_txt
    else:
        trained_embedding_list = img_embeddings
        targets = pc_names_target_img
    target_embedding_list = pc_embeddings
    
    k = 10


    unordered_results_by_acc = []
    results= 0
    logging.info("=> Testing K-Acurracy...")
    for i in range(len(trained_embedding_list)):
        print("Step: " + str(i) + " / " + str(len(trained_embedding_list)))
        trained_embedding = trained_embedding_list[i]
        indices = KNN(trained_embedding, target_embedding_list, k)

        acurracy_per_item = 0
        if targets != None:
            for j in range(len(indices)):
                if indices[j] == targets[i]:
                    acurracy_per_item = 1
        
        temp_array = []
        temp_array.append(acurracy_per_item)
        temp_array.append(pc_name[targets[i]])
        if evaluate_text:
            temp_array.append(txt_captions[i])
        else:
            temp_array.append(img_names[i])
        
        for j in range(len(indices)):
            temp_array.append(pc_name[indices[j]])
            #if evaluate_text:
            #    temp_array.append(txt_captions[pc_to_text[indices[j]]])
            #else: 
            #    temp_array.append(img_names[pc_to_img[indices[j]]])
            #temp_array.append(img_names[indices[j]])

        unordered_results_by_acc.append(temp_array)


        results = results + acurracy_per_item
        print(acurracy_per_item)
    results = results/len(trained_embedding_list)

    print('=> results acurracy' + str(results))
    logging.info('=> results acurracy' + str(results))
    
    
    


    #unordered mrr, real obj name, real txt, real img, top predicted obj name, top predicted txt, top predicted img
    #unordered_results_by_mrr = []
    #results= 0
    #logging.info("=> Testing MRR...")
    #for i in range(len(trained_embedding_list)):
    #    print("Step: " + str(i) + " / " + str(len(trained_embedding_list)))
    #    trained_embedding = trained_embedding_list[i]
    #    indices = KNN(trained_embedding, target_embedding_list, len(target_embedding_list))
#
    #    RR = 0
    #    if targets != None:
    #        for j in range(len(indices)):
    #            if targets[indices[j]] == targets[i]:
    #                RR = 1/ (j+1)
    #                break
    #    else:
    #        for j in range(len(indices)):
    #            if indices[j] == i:
    #                RR = 1/ (j+1)
#
    #    result_per_i = RR
    #    results = results + result_per_i
    #    print(result_per_i)
    #    
    #    unordered_results_by_mrr.append( [ result_per_i, targets[i], txt_captions[i], img_names[i], targets[indices[j]], txt_captions[indices[j]], img_names[indices[j]] , targets[indices[0]], txt_captions[indices[0]], img_names[indices[0]]] )
    #
    #results = results/len(trained_embedding_list)
#
    

    logging.info('=> results' + '\n')  
    #ordered_result = sorted(unordered_results_by_mrr, key=lambda x: x[0], reverse=True)
    ordered_result = sorted(unordered_results_by_acc, key=lambda x: x[0], reverse=True)
    for result in ordered_result:
        logging.info('\n')
        logging.info('\n')
        logging.info('\n')
        logging.info(result[0])
        logging.info(result[1])
        logging.info(result[2])
        result.pop(0)
        result.pop(0)
        result.pop(0)
        logging.info('\n')
        logging.info('\n')
        for j in range(k):
            logging.info(result[j])
            #logging.info(result[j*2+1])
            #logging.info(result[j*3+3])
        logging.info('\n')
        logging.info('\n')
        logging.info('\n')


    #for result in ordered_result:
    #    logging.info('\n')
    #    logging.info('\n')
    #    logging.info('\n')
    #    logging.info(result[0])
    #    logging.info(result[1])
    #    logging.info(result[2])
    #    logging.info(result[3])
    #    logging.info(result[4])
    #    logging.info(result[5])
    #    logging.info(result[6])
    #    logging.info(result[7])
    #    logging.info(result[8])
    #    logging.info(result[9])
    #    logging.info('\n')
    #    logging.info('\n')
    #    logging.info('\n')



def real_individual_test(args, model, clip_model = None):
    logging.info("=> Real Individual test...")
    logging.info("=> Loading Embedding dataset configuration...")
    #real_individual_test_text = "The exterior surface is decorated with repeating motifs. Each motif consists of a large, rounded shape outlined in dark greenish-black. Within the rounded shape is a red-brown rectangle, possibly representing a stylized element. The sides of each rectangle have thin, dark greenish-black vertical lines separating each rounded shape."
    #real_individual_test_text = "It is decorated with geometric patterns in brown, red, and black."
    real_individual_test_text = "has geometric patterns around the upper portion of its body. These patterns consist of angular shapes, lines, and step-like designs. There is also a band of geometric designs around the middle. The patterns appear to be painted or incised onto the surface of the jar"

    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),            
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = get_dataset(test_transform, tokenizer, args, 'train')
    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, drop_last=False)



    #Assumes img and txt in embedding
    pc_embeddings = []
    txt_embeddings = []
    img_embeddings = []
    pc_name = []
    txt_captions = []
    img_names = []
    pc_names_target_txt = []
    pc_names_target_img = []
    model.eval()
    with torch.no_grad():
        for data_iter, inputs in enumerate(test_loader):
            logging.info("Calculate tensors for pc")
            logging.info("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))
            print("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))

            img__name = inputs[1]
            texts = inputs[2]
            feature = inputs[3] 
            image = inputs[4]
            names = inputs[7]
            captions = inputs[6]

            use_image = torch.tensor([1]).repeat(texts.shape[0])
            loss_masks = use_image.float()
            texts = texts.to(args.device)
            image = image.to(args.device)
            
            if not args.use_embed:
                #ignorar que se repite el torch no grad
                with torch.no_grad():
                    logging.info('=> encoding captions')  
                    texts, image = compute_embedding(clip_model, texts, image)
            inputs = [feature, texts, image]
            # to device
            inputs = [tensor.to(device=args.device) for tensor in inputs]

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                print('=> Encoding pc')  
                outputs = model(*inputs)
                print('=> Pc Encoded')  



            #logging.info("pc Embedded")
            pc_features = outputs['pc_embed']
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            txts_tensor = outputs['text_embed'].cpu()
            img_tensor = outputs['image_embed'].cpu()
            pc_tensor = pc_features.cpu()
#
            ##junta por cada embedding su text_dir
            
            for pc, txt, img, name, caption, img_name in zip(pc_tensor, txts_tensor, img_tensor, names, captions, img__name):
                if name not in pc_name:
                    pc_name.append(name)
                    pc_embeddings.append(pc)

                if img_name not in img_names:
                    img_names.append(img_name)
                    img_embeddings.append(img)
                    img_index = pc_name.index(name)
                    pc_names_target_img.append(img_index)
                
                txt_embeddings.append(txt)
                txt_index = pc_name.index(name)
                pc_names_target_txt.append(txt_index)
                txt_captions.append(caption)
                
                #print(name)



    #return
    evaluate_text = True

    if evaluate_text:
        trained_embedding_list = txt_embeddings
        targets = pc_names_target_txt
    else:
        trained_embedding_list = img_embeddings
        targets = pc_names_target_img
    target_embedding_list = pc_embeddings
    
    k = 10


    unordered_results_by_acc = []
    results= 0

    logging.info("=> Loading Clip...")
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.pretrained) 
    #clip_model.to(args.device)
    logging.info("=> Clip Loaded...")
    logging.info("=> Importing tokenizer...")
    

    #Tokenizador que trunca en 77
    def tokenizer_handmade(tokenizer, text, max_context_length = 77):
        context_length = max_context_length -2 
        texts = [text]
        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        all_tokens = []
        for text in texts:
            chunk = []
            encoded  = tokenizer.encode(text)
            #encoded = [sot_token] + encoded + [eot_token]
            #for i in range(0, len(encoded), context_length):
            for i in range(0, 1, context_length):
                chunk =  [sot_token] + encoded[i : i + context_length] + [eot_token]
                all_tokens.append(chunk)
        #all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), max_context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            tokens = tokens[:max_context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result
    
    with torch.no_grad():
        logging.info('=> encoding captions')  
        tokens1 = []
        tokens1 = tokenizer_handmade(tokenizer, real_individual_test_text)
        text_features1 = clip_model.encode_text(tokens1)
        text_outputs1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
        text_outputs1 = text_outputs1.mean(dim=0)
        

    logging.info("=> Testing K-Acurracy...")
    trained_embedding = text_outputs1
    indices = KNN(trained_embedding, target_embedding_list, k)
    logging.info('=> results' + '\n') 

    
    for j in range(len(indices)):
        logging.info(pc_name[indices[j]])
    
    

    logging.info('=> end' + '\n')  
   


  


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def PCA_analisis(args, model, clip_model=None):
    logging.info("=> Individual test...")
    logging.info("=> Loading Embedding dataset configuration...")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    #test_transform = transforms.Compose([
    #    transforms.Resize(256),            
    #    transforms.CenterCrop(224), 
    #    transforms.ToTensor(),
    #    normalize
    #])

    test_transform = transforms.Compose([
        transforms.Resize(256),            
        transforms.CenterCrop(224), 
        transforms.ToTensor()
        
    ])

    test_dataset = get_dataset(test_transform, tokenizer, args, 'train')
    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, drop_last=False)



    #Assumes img and txt in embedding
    pc_embeddings = []
    txt_embeddings = []
    img_embeddings = []
    pc_name = []
    img_names = []
    pc_names_target_txt = []
    pc_names_target_img = []
    model.eval()
    
    with torch.no_grad():
        for data_iter, inputs in enumerate(test_loader):
            logging.info("Calculate tensors for pc")
            logging.info("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))
            print("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))

            img__name = inputs[1]
            texts = inputs[2]
            feature = inputs[3] 
            image = inputs[4]
            names = inputs[7]
            captions = inputs[6]

            #print(names)
            #print(img__name)
            use_image = torch.tensor([1]).repeat(texts.shape[0])
            loss_masks = use_image.float()
            texts = texts.to(args.device)
            image = image.to(args.device)
            
            if not args.use_embed:
                #ignorar que se repite el torch no grad
                with torch.no_grad():
                    logging.info('=> encoding captions')  
                    texts, image = compute_embedding(clip_model, texts, image)
            inputs = [feature, texts, image]
            # to device
            inputs = [tensor.to(device=args.device) for tensor in inputs]

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                print('=> Encoding pc')  
                
                outputs = model(*inputs)

                
                
                print('=> Pc Encoded')  
            


            #logging.info("pc Embedded")
            pc_features = outputs['pc_embed']
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            txts_tensor = outputs['text_embed'].cpu()
            img_tensor = outputs['image_embed'].cpu()
            pc_tensor = pc_features.cpu()
#
            ##junta por cada embedding su text_dir
            
            for pc, txt, img, name, caption, img_name in zip(pc_tensor, txts_tensor, img_tensor, names, captions, img__name):
                if name not in pc_name:
                    pc_name.append(name)
                    pc_embeddings.append(pc)
                if img_name not in img_names:
                    img_names.append(img_name)
                    img_embeddings.append(img)
                    img_index = pc_name.index(name)
                    pc_names_target_img.append(img_index)

                txt_embeddings.append(txt)
                txt_index = pc_name.index(name)
                pc_names_target_txt.append(txt_index)
    
    

    #grupos = [pc_embeddings, img_embeddings, txt_embeddings]  # Lista de listas
    #nombres_grupos = ['Objetos 3D', 'Imagenes', 'Textos']

    grupos = [img_embeddings, txt_embeddings]  # Lista de listas
    nombres_grupos = ['Imagenes', 'Textos']

    todos_los_tensores = []
    etiquetas = []

    for i, grupo in enumerate(grupos):
        for tensor in grupo:
            todos_los_tensores.append(tensor)
            etiquetas.append(i)

    # Convertir a tensor y luego a numpy
    tensor_total = torch.stack(todos_los_tensores).numpy()
    etiquetas = np.array(etiquetas)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(tensor_total)

    # Graficar con distintos colores
    plt.figure(figsize=(8, 6))
    colores = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']  # Puedes añadir más

    for i, nombre in enumerate(nombres_grupos):
        idx = etiquetas == i
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], alpha=0.7, label=nombre, color=colores[i])

    #plt.xlabel('PC1')
    #plt.ylabel('PC2')
    #plt.title('Proyección PCA por grupo')
    plt.grid(True)
    plt.legend()
    plt.savefig('pca_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

   

    return




def memory_test(args, model, clip_model = None):
    logging.info("=> memory test...")
    logging.info("=> Loading Embedding dataset configuration...")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),            
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = get_dataset(test_transform, tokenizer, args, 'train')
    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, drop_last=False)



    
    model.eval()
    Total_time = 0
    with torch.no_grad():
        for data_iter, inputs in enumerate(test_loader):
            logging.info("Calculate tensors for pc")
            logging.info("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))
            print("Local Step: " + str(data_iter) + "/" + str(len(test_loader)))

           
            texts = inputs[2]
            feature = inputs[3] 
            image = inputs[4]
            

            #print(names)
            #print(img__name)
            use_image = torch.tensor([1]).repeat(texts.shape[0])
            loss_masks = use_image.float()
            texts = texts.to(args.device)
            image = image.to(args.device)
            
            if not args.use_embed:
                #ignorar que se repite el torch no grad
                with torch.no_grad():
                    logging.info('=> encoding captions')  
                    texts, image = compute_embedding(clip_model, texts, image)

            inputs = [feature, texts, image]
            # to device
            inputs = [tensor.to(device=args.device) for tensor in inputs]

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                print('=> Encoding pc')  
                start = time.perf_counter()
                outputs = model(*inputs)
                end = time.perf_counter()
                print('=> Pc Encoded')  
            Total_time = Total_time + (end - start)
            print('=> Time: '  + str(end - start))  


            #logging.info("pc Embedded")
            txts_tensor = outputs['text_embed'].cpu()
            img_tensor = outputs['image_embed'].cpu()
            pc_tensor = outputs['pc_embed'].cpu()
#

    
    allocated = torch.cuda.memory_allocated() / 1024**2  # en MB
    reserved = torch.cuda.memory_reserved() / 1024**2    # en MB

    print(f"Memoria asignada: {allocated:.2f} MB")
    print(f"Memoria reservada: {reserved:.2f} MB")

    #only use a batch size of 1
    Total_time = Total_time/len(test_loader)
    logging.info(f"Memoria asignada: {allocated:.2f} MB")
    logging.info(f"Memoria reservada: {reserved:.2f} MB")
    logging.info("=> Mean Time: " + str(Total_time))

    return



def test_zeroshot_3d_core(test_loader, validate_dataset_name, model, clip_model, tokenizer, args=None, test_data=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f') 
    top3 = AverageMeter('Acc@3', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top3, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    with open(os.path.join("./data", 'labels.json')) as f:
        labels = json.load(f)[validate_dataset_name]

    with torch.no_grad():
        logging.info('=> encoding captions')               
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).to(device=args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top3 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)

        for i, (pc, target, target_name, rgb) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.to(device=args.device, non_blocking=True)
            rgb = rgb.to(device=args.device, non_blocking=True)
            feature = torch.cat((pc, rgb),dim=-1)
            target = target.to(device=args.device, non_blocking=True)

            # encode pc
            pc_features = utils.get_model(model).encode_pc(feature)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_pc = pc_features.float() @ text_features.float().t()

            # measure accuracy and record loss
            (acc1, acc3, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 3, 5))
            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc3, acc5 = utils.scaled_all_reduce([acc1, acc3, acc5])
            top1.update(acc1.item(), pc.size(0))
            top3.update(acc3.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top3_accurate = correct[:3].float().sum(0, keepdim=True).squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top3_accurate[idx].item():
                    per_class_correct_top3[name] += 1
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        top3_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            top3_accuracy_per_class[name] = per_class_correct_top3[name] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        top3_accuracy_per_class = collections.OrderedDict(top3_accuracy_per_class)
        top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        logging.info(','.join(top1_accuracy_per_class.keys()))
        logging.info(','.join([str(value) for value in top1_accuracy_per_class.values()]))
        logging.info(','.join([str(value) for value in top3_accuracy_per_class.values()]))        
        logging.info(','.join([str(value) for value in top5_accuracy_per_class.values()]))
    progress.synchronize()
    logging.info('0-shot * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Acc@5 {top5.avg:.3f}')
    return {'acc1': top1.avg, 'acc3': top3.avg, 'acc5': top5.avg}

def test_zeroshot_3d(args, model, clip_model):
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    logging.info('loaded checkpoint {}'.format(args.ckpt_path))
    sd = checkpoint['module']
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    tokenizer = SimpleTokenizer()

    test_dataset = utils.get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )
    test_lvis_dataset = utils.get_dataset(None, tokenizer, args, 'val_lvis')
    test_lvis_loader = torch.utils.data.DataLoader(
        test_lvis_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )

    test_dataset_scanonjnn = utils.get_dataset(None, tokenizer, args, 'val_scanobjnn')
    test_loader_scanonjnn = torch.utils.data.DataLoader(
        test_dataset_scanonjnn, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )

    results_mnet = test_zeroshot_3d_core(test_loader, args.validate_dataset_name, model, clip_model, tokenizer, args, 'modelnet')
    results_lvis = test_zeroshot_3d_core(test_lvis_loader, args.validate_dataset_name_lvis, model, clip_model, tokenizer, args, 'lvis')
    results_scanobjnn = test_zeroshot_3d_core(test_loader_scanonjnn, args.validate_dataset_name_scanobjnn, model, clip_model, tokenizer, args, 'scanobjnn')
    return results_mnet, results_lvis, results_scanobjnn


def test_zeroshot_3d_core_memoria(test_loader, model, clip_model, args=None, test_data=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f') 
    top3 = AverageMeter('Acc@3', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top3, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    #with open(os.path.join("./data", 'templates.json')) as f:
    #    templates = json.load(f)[args.validate_dataset_prompt]

    #with open(os.path.join("./data", 'labels.json')) as f:
    #    labels = json.load(f)[validate_dataset_name]

    with torch.no_grad():
        logging.info('=> encoding captions')           
        text_features = []
        for i, inputs in enumerate(test_loader):
            print("Local Step: " + str(i) + "/" + str(len(test_loader)))
            texts = inputs[2]
            texts = texts.to(args.device)
            texts, __ = compute_embedding_for_dataset(clip_model, texts, None)
            #texts = [t.format(l) for t in templates]
            #texts = tokenizer(texts).to(device=args.device, non_blocking=True)
            #if len(texts.shape) < 2:
            #    texts = texts[None, ...]
            #class_embeddings = clip_model.encode_text(texts)
            #class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            #class_embeddings = class_embeddings.mean(dim=0)
            #class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(texts)
        text_features = torch.cat(text_features, dim=0)
       
        end = time.time()
        #per_class_stats = collections.defaultdict(int)
        #per_class_correct_top1 = collections.defaultdict(int)
        #per_class_correct_top3 = collections.defaultdict(int)
        #per_class_correct_top5 = collections.defaultdict(int)
        print('=> Calculating the acurracies')    
        logging.info('=> Calculating the acurracies')    
        for i, inputs in enumerate(test_loader):
            #for name in target_name:
            #    per_class_stats[name] += 1
            print("Local Step: " + str(i) + "/" + str(len(test_loader)))
            target = inputs[5]
            pc = inputs[3] 
            rgb = inputs[4]

            pc_features = pc.to(device=args.device, non_blocking=True)
            rgb = rgb.to(device=args.device, non_blocking=True)
            #feature = torch.cat((pc, rgb),dim=-1)
            #nose porque le pusieron target, pero es el texto
            target = target.to(device=args.device, non_blocking=True)
            #target, __ = compute_embedding_for_dataset(clip_model, target, None)
            #target = clip_model.encode_text(target)
            #target = target / target.norm(dim=-1, keepdim=True)
            #target = target.mean(dim=0)

            # encode pc
            pc_features = utils.get_model(model).encode_pc(pc_features)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            #[item1, similarity [0.2,0.2,0.6,...],
            # item2, similarity [0.1,0.8,0.1,...]]
            logits_per_pc = pc_features.float() @ text_features.float().t()
            #logits_per_pc = pc_features @ text_features.t()
            #logits_per_pc.shape()
            # measure accuracy and record loss
            #Acurracies y si esta correcto
            print(logits_per_pc.size())
            print(target.size())
            (acc1, acc3, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 3, 5))
            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc3, acc5 = utils.scaled_all_reduce([acc1, acc3, acc5])
            top1.update(acc1.item(), pc.size(0))
            top3.update(acc3.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            del rgb, pc_features, inputs, target
            torch.cuda.empty_cache()
            #top1_accurate = correct[:1].squeeze()
            #top3_accurate = correct[:3].float().sum(0, keepdim=True).squeeze()
            #top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            #for idx, name in enumerate(target_name):
            #    if top1_accurate[idx].item():
            #        per_class_correct_top1[name] += 1
            #    if top3_accurate[idx].item():
            #        per_class_correct_top3[name] += 1
            #    if top5_accurate[idx].item():
            #        per_class_correct_top5[name] += 1

            #if i % args.print_freq == 0:
            #    progress.display(i)
        del text_features
        torch.cuda.empty_cache()
        #top1_accuracy_per_class = {}
        #top3_accuracy_per_class = {}
        #top5_accuracy_per_class = {}
        #for name in per_class_stats.keys():
        #    top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
        #    top3_accuracy_per_class[name] = per_class_correct_top3[name] / per_class_stats[name]
        #    top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        #top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        #top3_accuracy_per_class = collections.OrderedDict(top3_accuracy_per_class)
        #top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        #logging.info(','.join(top1_accuracy_per_class.keys()))
        #logging.info(','.join([str(value) for value in top1_accuracy_per_class.values()]))
        #logging.info(','.join([str(value) for value in top3_accuracy_per_class.values()]))        
        #logging.info(','.join([str(value) for value in top5_accuracy_per_class.values()]))
    progress.synchronize()
    print("Top1: " + str(top1.avg))
    print("Top1: " + str(top3.avg))
    print("Top1: " + str(top5.avg))
    logging.info('0-shot * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Acc@5 {top5.avg:.3f}')
    
    return {'acc1': top1.avg, 'acc3': top3.avg, 'acc5': top5.avg}


def test_zeroshot_memoria(args, model, clip_model):
    #checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    #logging.info('loaded checkpoint {}'.format(args.ckpt_path))
    #sd = checkpoint['module']
    #if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
    #    sd = {k[len('module.'):]: v for k, v in sd.items()}
    #model.load_state_dict(sd)
    #Hay que cambiar aqui los valores
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])

    tokenizer = SimpleTokenizer()
    print(args.validate_dataset_name)
    test_dataset = utils.get_dataset(test_transform, tokenizer, args, "val")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )    

    results_test = test_zeroshot_3d_core_memoria(test_loader, args.validate_dataset_name, model, clip_model, tokenizer, args, 'modelnet')
    return results_test


def val_loss_calculation(args,  val_loader, model, clip_model, criterion):
    model.eval()
    clip_model.eval()
    

    with torch.no_grad():

        local_sample_count = 0
        local_running_loss = 0
        
        for data_iter, inputs in enumerate(val_loader):

            #step = epoch * iters_per_epoch + optim_iter # global training iteration
            print("Local Step: " + str(data_iter) + "/" + str(len(val_loader)))

            texts = inputs[2]
            feature = inputs[3] 
            image = inputs[4]
            batch_size = image.shape[0]
            use_image = torch.tensor([1]).repeat(texts.shape[0])
            loss_masks = use_image.float()
            texts = texts.to(args.device)
            image = image.to(args.device)
            print("EMPEZANDO A ENCODEAR")
            if not args.use_embed:
                #ignorar que se repite el torch no grad
                with torch.no_grad():
                    logging.info('=> encoding captions')  
                    texts, image = compute_embedding_for_dataset(clip_model, texts, image, image_exist=True)
            #else:
            #    texts = torch.cat(texts, dim=0)  # Shape: [N, D]
            #    image = torch.cat(image, dim=0)
            print("TERMINO DE ENCODEAR")
            inputs = [feature, texts, image]
            # to device
            inputs = [tensor.to(device=args.device) for tensor in inputs]

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                
                outputs = model(*inputs)
                loss_dict = criterion(outputs, loss_masks)
                loss = loss_dict['loss']
                #loss /= args.update_freq
                print(image.shape[0])
                print(loss.item())
                #Loss es la formula completa por batch
                #Ojo: por alguna razon la loss que se usa es realmente la del paper de
                # ulip2 no la de uni3d del paper
                local_running_loss += loss.item()  # Total loss, no el promedio
                local_sample_count += batch_size

        total_loss_tensor = torch.tensor(local_running_loss, device=args.device)
        total_count_tensor = torch.tensor(local_sample_count, device=args.device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count_tensor, op=dist.ReduceOp.SUM)

        epoch_loss = total_loss_tensor.item() / total_count_tensor.item()
        print('Val Loss: {:.4f}'.format(epoch_loss))
        print(f"Memory Allocated in Validation: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        print(f"Memory Reserved in Validation: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
        print('Val Loss wrong: {:.4f}'.format(total_loss_tensor.item()))
        logging.info('Val Loss wrong: {:.4f}'.format(total_loss_tensor.item()))   
        return epoch_loss
            




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        logging.info('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


if __name__ == '__main__':
    main(sys.argv[1:])