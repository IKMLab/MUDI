import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb


class BaseTrainer:
    r"""The base trainer for the model

    Args:
        config (dict): The configuration of the model
        logger (logging.Logger): The logger
        model (CoherenceGraphModel): The model
        train_data (DataLoader): The training data
        valid_data (DataLoader): The validation data
    """

    def __init__(self, config: dict, logger: logging.Logger, model: nn.Module,
                 train_data: DataLoader, valid_data: DataLoader):
        self.device = config.device
        self.distributed = config.distributed
        self.logger = logger
        self.model = model

        self.train_data = train_data
        self.valid_data = valid_data

        self.dataset_name = config.dataset_name
        self.checkpoint = config.checkpoint

        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.learner = config.learner  # Optimizer name
        self.optimizer = self._build_optimizer()  # Optimizer instance
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, 0.95)

        self.endure_times = config.endure_times

        self.wandb_tracker = wandb

        if config.distributed:
            self.accelerator = config.accelerator
            self.model, self.optimizer, self.scheduler, \
                self.train_data, self.valid_data = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, self.train_data,
                self.valid_data)

            self.wandb_tracker = self.accelerator

    @property
    def model_name(self) -> str:
        r"""Get the model name

        Returns:
            str: The model name
        """

        raise NotImplementedError

    def _build_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.lr``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: The optimizer
        """

        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.lr)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if learner.lower() == 'adam':
            optimizer = torch.optim.Adam(params,
                                         lr=learning_rate,
                                         weight_decay=weight_decay)
        elif learner.lower() == 'adamw':
            optimizer = torch.optim.AdamW(params,
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(params,
                                        lr=learning_rate,
                                        weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(params,
                                            lr=learning_rate,
                                            weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(params,
                                            lr=learning_rate,
                                            weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(params, lr=learning_rate)

        return optimizer

    def train(self) -> tuple[str, int]:
        raise NotImplementedError

    def training_step(self, data_loader: DataLoader):
        raise NotImplementedError

    def evaluate_step(self, data_loader: DataLoader):
        raise NotImplementedError

    def predict(self, data_loader: DataLoader):
        raise NotImplementedError
