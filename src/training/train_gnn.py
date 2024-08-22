import argparse
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from easydict import EasyDict
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
)

sys.path.append('.')

from trainer import get_trainer_class

from src.base.base_dataset import BaseGraphDataset
from src.data.collator import DialogueGraphDataCollator
from src.data.dataset import (
    ConvAI2ForDialogueGraphEncodingDataset,
    RccDataset,
)
from src.data.processor import (
    ConvAI2DataProcessor,
    DialogueGraphDataProcessMode,
)
from src.models.dgat_conv_finetuning import (
    DGatForCoherenceAwareDialogueEncoding,
    get_layer_class,
)
from src.models.dgat_conv_pretraining import (
    DGatConvPretraining,
    DGatForPreTraining,
)
from src.utils.args_utils import parse_gnn_args
from src.utils.constants import ModelTrainMode
from src.utils.data_utils import (
    convert_to_dict,
    find_and_load_model,
    save_yaml,
)
from src.utils.model_configs import (
    ModelArguments,
    TrainingConfigs,
)
from src.utils.utils import (
    create_logger,
    create_run_dir,
    initialize_wandb,
    set_seed,
)


def prepare_dataset(
        args: argparse.Namespace, train_mode: ModelTrainMode
) -> tuple[BaseGraphDataset, BaseGraphDataset]:
    if train_mode == ModelTrainMode.PRETRAINING:
        train_convai2_dataset = RccDataset(
            root=args.data_dir,
            raw_file_name=args.train_data_name,
            processed_file_dir=args.processed_train_data_dir,
            processed_file_name=args.processed_train_data_name)
        valid_convai2_dataset = RccDataset(
            root=args.data_dir,
            raw_file_name=args.valid_data_name,
            processed_file_dir=args.processed_valid_data_dir,
            processed_file_name=args.processed_valid_data_name)
    elif train_mode == ModelTrainMode.FINETUNEING:
        if args.pretrained_utterance_encoder == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif args.pretrained_utterance_encoder == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif args.pretrained_utterance_encoder == 'none':
            tokenizer = None

        processor = ConvAI2DataProcessor(k_hop=args.k_hop,
                                         mode=DialogueGraphDataProcessMode(
                                             args.process_mode))

        train_convai2_dataset = ConvAI2ForDialogueGraphEncodingDataset(
            root=args.data_dir,
            raw_file_name=args.train_data_name,
            processed_file_dir=args.processed_train_data_dir,
            processed_file_name=args.processed_train_data_name,
            transform=processor.transform,
            pre_transform=processor.pre_transform,
            pre_filter=processor.pre_filter,
            utterance_tokenizer=tokenizer,
            reverse_edge=args.reverse_edge,
            directed=args.directed)

        valid_convai2_dataset = ConvAI2ForDialogueGraphEncodingDataset(
            root=args.data_dir,
            raw_file_name=args.valid_data_name,
            processed_file_dir=args.processed_valid_data_dir,
            processed_file_name=args.processed_valid_data_name,
            transform=processor.transform,
            pre_transform=processor.pre_transform,
            pre_filter=processor.pre_filter,
            utterance_tokenizer=tokenizer,
            reverse_edge=args.reverse_edge,
            directed=args.directed)

    return train_convai2_dataset, valid_convai2_dataset


def build_dataloader(
    batch_size: int,
    num_workers: int,
    train_mode: ModelTrainMode,
    training_dataset: BaseGraphDataset,
    validation_dataset: BaseGraphDataset,
    testing_dataset: BaseGraphDataset,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_collator = DialogueGraphDataCollator(train_mode=train_mode)

    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              collate_fn=data_collator,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=validation_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              collate_fn=data_collator,
                              pin_memory=True)
    test_loader = DataLoader(dataset=testing_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=data_collator,
                             pin_memory=True)

    return train_loader, valid_loader, test_loader


def build_model(args: ModelArguments,
                train_mode: ModelTrainMode) -> DGatConvPretraining:
    if train_mode == ModelTrainMode.PRETRAINING:
        return DGatForPreTraining(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            edge_dim=args.edge_dim,
            encoder_layer_class=get_layer_class(args.layer_type),
            num_encoder_layers=args.num_layers,
            # Below are the parameters for GNN Layer class (collecting to kwargs)
            heads=args.num_heads,
            add_self_loops=args.add_self_loops)
    elif train_mode == ModelTrainMode.FINETUNEING:
        return DGatForCoherenceAwareDialogueEncoding(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            edge_dim=args.edge_dim,
            encoder_layer_instance=get_layer_class(args.layer_type),
            num_encoder_layers=args.num_layers,
            utterance_encoder_class=args.pretrained_utterance_encoder,
            pretrained_weights_path=args.pretrained_weights_path,
            # Below are the parameters for GNN Layer class (collecting to kwargs)
            heads=args.num_heads,
            add_self_loops=args.add_self_loops)


def main(args: argparse.Namespace):
    args = EasyDict(vars(args))

    if args.distributed:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            log_with='wandb' if not args.do_inference or args.wandb else None)
        initialize_wandb(args, accelerator=accelerator)
        device = accelerator.device
    else:
        initialize_wandb(args)
        device = torch.device('cuda' if not args.cpu else 'cpu')

    set_seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.do_inference:
        run_dir = Path(args.ckpt_dir)
        log_file_name = 'infer.log'
    else:
        run_dir = Path(
            create_run_dir(args.ckpt_dir,
                           is_main_process=accelerator.is_main_process
                           if args.distributed else None))
        log_file_name = 'log'

    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir (log's location): {run_dir}")

    logger = create_logger(log_dir=run_dir,
                           log_file_name=log_file_name,
                           logger_name=f'train_gnn.{args.train_mode}')

    logger.info('========== args ==========')
    for k, v in args.items():
        logger.info(f'{k}: {v}')
    print(args)
    save_yaml(run_dir / 'config.yaml', convert_to_dict(args))

    train_mode = ModelTrainMode(args.train_mode)

    # Load dataset
    train_dataset, valid_dataset = prepare_dataset(args, train_mode)

    logger.info(f'Number of dataset: \
        train_set: {len(train_dataset)}, \
        valid_set: {len(valid_dataset)}')

    # Build dataloader
    train_loader, valid_loader, test_loader = build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        training_dataset=train_dataset,
        validation_dataset=valid_dataset,
        testing_dataset=valid_dataset,
        train_mode=train_mode)

    # Build model
    model_arguments = ModelArguments(
        input_dim=train_dataset.num_node_features,
        hidden_dim=args.embedding_dim,
        output_dim=args.embedding_dim,
        edge_dim=train_dataset.num_edge_features,
        layer_type=args.layer_type,
        num_layers=args.num_layers,
        # Only for Fine-tuning
        pretrained_utterance_encoder=args.pretrained_utterance_encoder,
        pretrained_weights_path=args.pretrained_model_path,
        # Below are the parameters for GNN Layer class (collecting to kwargs)
        num_heads=args.num_heads,
        add_self_loops=True)
    model = build_model(args=model_arguments, train_mode=train_mode)
    model = model.to(device)
    logger.info(f'Building Coherence Graph model ({train_mode}).')

    # Build trainer
    training_config = TrainingConfigs(
        device=device,
        distributed=args.distributed,
        dataset_name='ConvAI2'
        if train_mode == ModelTrainMode.FINETUNEING else 'RCC',
        checkpoint=run_dir,
        learner=args.optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        endure_times=args.endure_times,
        # Below are the external parameters for loss function
        coh_rel_cls_weight=args.coh_rel_cls_weight,
        link_prediction_weight=args.link_prediction_weight,
        next_resp_type_direct_weight=args.next_resp_type_direct_weight,
        next_resp_type_seq_weight=args.next_resp_type_seq_weight)
    if args.distributed:
        training_config.accelerator = accelerator

    trainer_cls = get_trainer_class('dgat', train_mode)
    trainer = trainer_cls(config=training_config,
                          logger=logger,
                          model=model,
                          train_data=train_loader,
                          valid_data=valid_loader)

    if not args.do_inference:
        model_path, best_epoch = trainer.train()
        logger.info(f'Best epoch: {best_epoch}')
        logger.info(f'Loading the best model from {model_path}')

        if args.distributed:
            accelerator.end_training()
    else:
        logger.info('Doing inference only.')
        logger.info(f'Loading model from {run_dir}')
        model = find_and_load_model(model, run_dir)
        model = model.to(device)


if __name__ == '__main__':
    main(parse_gnn_args())
