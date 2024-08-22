import argparse
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb


def get_now_time():
    return datetime.now().strftime('%Y-%m-%d_%H:%M')


def initialize_wandb(args: argparse.Namespace, **kwargs):
    if not args.do_inference and args.wandb:
        if args.distributed:
            accelerator = kwargs.get('accelerator')
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=args,
                init_kwargs={
                    'wandb': {
                        'entity': args.wandb_entity,
                        'name': f'{args.wandb_run_name}-{get_now_time()}'
                    }
                })
        else:
            wandb.init(project=args.wandb_project,
                       entity=args.wandb_entity,
                       config=args,
                       name=f'{args.wandb_run_name}-{get_now_time()}')
    else:  # inference only
        os.environ['WANDB_MODE'] = 'disabled'


def create_run_dir(checkpoint_dir: str, is_main_process: bool = None) -> str:
    # traverse the checkpoint_dir to find the largest run_dir and create a new one `run_{index}`
    dirs = os.listdir(checkpoint_dir)
    max_run_index = 0
    for d in dirs:
        if d.startswith('run_'):
            index = int(d.split('_')[1])
            max_run_index = max(max_run_index, index)

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if is_main_process is None:
        run_index = max_run_index + 1
    else:
        if is_main_process:
            run_index = max_run_index
        elif local_rank == 1:
            run_index = max_run_index + 1
        else:
            run_index = max_run_index

    run_dir = os.path.join(checkpoint_dir, f'run_{run_index}')

    return run_dir


def create_logger(log_dir: Path,
                  log_file_name: str,
                  logger_name: str = None) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        filename=log_dir / log_file_name,
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(logger_name or __name__)

    return logger


def set_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def scheduled_sampling(epoch, total_epochs, start_p=0.0, end_p=0.9):
    p = start_p + (end_p - start_p) * epoch / total_epochs

    return np.random.rand() < p


def scheduled_sampling_exp(epoch, total_epochs, start_p=0.0, end_p=0.9, base=2):
    ratio = epoch / total_epochs

    p = start_p + (end_p - start_p) * (1 - base**(-ratio * 10))
    return np.random.rand() < p
