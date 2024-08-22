import argparse
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from easydict import EasyDict
from torch.utils.data import DataLoader
from transformers import (
    BartTokenizer,
    BertTokenizer,
    RobertaTokenizer,
)

sys.path.append('.')

from trainer import get_trainer_class

from base.base_dataset import BaseGraphDataset
from data.collator import PersonalizedDialogueGenerationDataCollator
from data.dataset import (
    ConvAI2ForDialogueGraphEncodingDataset,
    ConvAI2ForPersonalizedDialogueGenerationDataset,
)
from data.processor import ConvAI2DataProcessor, DialogueGraphDataProcessMode
from models.modeling_generator import PersonalizedDialogueGenerator
from utils.args_utils import parse_generator_args
from utils.constants import (
    NEW_SPEICAL_TOKENS,
    NEW_SPEICAL_TOKENS_MAP,
    ModelTrainMode,
)
from utils.data_utils import (
    convert_to_dict,
    find_and_load_model,
    load_pickle,
    save_yaml,
)

# from utils.mllt.datasets.loader import ClassAwareSampler
from utils.model_configs import (
    ModelArguments,
    TrainingConfigs,
)
from utils.utils import (
    create_logger,
    create_run_dir,
    initialize_wandb,
    set_seed,
)


def prepare_dataset(
    args: argparse.Namespace, generator_tokenizer: BartTokenizer
) -> tuple[ConvAI2ForPersonalizedDialogueGenerationDataset,
           ConvAI2ForPersonalizedDialogueGenerationDataset]:
    if args.pretrained_utterance_encoder == 'bert':
        utterance_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.pretrained_utterance_encoder == 'roberta':
        utterance_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif args.pretrained_utterance_encoder == 'none':
        utterance_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        processor = ConvAI2DataProcessor(k_hop=args.k_hop,
                                         mode=DialogueGraphDataProcessMode(
                                             args.process_mode))

        train_convai2_graph_dataset = ConvAI2ForDialogueGraphEncodingDataset(
            root=args.data_dir,
            raw_file_name=args.train_data_name,
            processed_file_dir=args.processed_train_data_dir,
            processed_file_name=args.processed_train_data_name,
            transform=processor.transform,
            pre_transform=processor.pre_transform,
            pre_filter=processor.pre_filter,
            utterance_tokenizer=utterance_tokenizer,
            reverse_edge=args.reverse_edge,
            directed=args.directed)

        valid_convai2_graph_dataset = ConvAI2ForDialogueGraphEncodingDataset(
            root=args.data_dir,
            raw_file_name=args.valid_data_name,
            processed_file_dir=args.processed_valid_data_dir,
            processed_file_name=args.processed_valid_data_name,
            transform=processor.transform,
            pre_transform=processor.pre_transform,
            pre_filter=processor.pre_filter,
            utterance_tokenizer=utterance_tokenizer,
            reverse_edge=args.reverse_edge,
            directed=args.directed)

        train_convai2_dataset = ConvAI2ForPersonalizedDialogueGenerationDataset(
            dataset=load_pickle(train_convai2_graph_dataset.raw_paths[0]),
            graph_dataset=train_convai2_graph_dataset,
            tokenizer=generator_tokenizer,
            nearest_k_turn=5)

        valid_convai2_dataset = ConvAI2ForPersonalizedDialogueGenerationDataset(
            dataset=load_pickle(valid_convai2_graph_dataset.raw_paths[0]),
            graph_dataset=valid_convai2_graph_dataset,
            tokenizer=generator_tokenizer,
            nearest_k_turn=5)

    return train_convai2_dataset, valid_convai2_dataset


def build_dataloader(
    batch_size: int,
    num_workers: int,
    training_dataset: BaseGraphDataset,
    validation_dataset: BaseGraphDataset,
    testing_dataset: BaseGraphDataset,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_collator = PersonalizedDialogueGenerationDataCollator()

    train_loader = DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        # sampler=ClassAwareSampler(
        #     nclass=len(COHERENCE_RELATIONS),
        #     cls_data_list=initializer.get_tagtrain_index_dic(),
        #     gt_labels=tag_train_data.gt_labels,
        #     seed=args.seed,
        # ),
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


def build_model(args: ModelArguments) -> PersonalizedDialogueGenerator:
    return PersonalizedDialogueGenerator.from_pretrained(
        'facebook/bart-large',
        dialogue_encoder_input_dim=args.input_dim,
        dialogue_encoder_hidden_dim=args.hidden_dim,
        dialogue_encoder_output_dim=args.output_dim,
        dialogue_encoder_edge_dim=args.edge_dim,
        dialogue_encoder_layer_type=args.layer_type,
        dialogue_encoder_num_encoder_layers=args.num_layers,
        dialogue_encoder_utterance_encoder_class=args.
        pretrained_utterance_encoder,
        dialogue_encoder_pretrained_weights_path=args.
        pretrained_dialogue_encoder_weights_path,
        dialogue_encoder_pretrained_encoder_weights_path=args.
        pretrained_dialogue_encoder_encoder_weights_path,
        dialogue_encoder_heads=args.num_heads,
        dialogue_encoder_add_self_loops=args.add_self_loops,
        freeze_dialogue_encoder=True)


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
                           logger_name='train_generator')

    logger.info('========== args ==========')
    for k, v in args.items():
        logger.info(f'{k}: {v}')
    print(args)
    save_yaml(run_dir / 'config.yaml', convert_to_dict(args))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    tokenizer.add_tokens(NEW_SPEICAL_TOKENS)
    logger.info(
        f'Add new special tokens to tokenizer: {NEW_SPEICAL_TOKENS_MAP}')
    logger.info(f'Vocab size: {len(tokenizer)}')

    # Load dataset
    train_dataset, valid_dataset = prepare_dataset(
        args=args, generator_tokenizer=tokenizer)

    logger.info(f'Number of dataset: \
        train_set: {len(train_dataset)}, \
        valid_set: {len(valid_dataset)}')

    # Build dataloader
    train_loader, valid_loader, test_loader = build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        training_dataset=train_dataset,
        validation_dataset=valid_dataset,
        testing_dataset=valid_dataset)

    # Build model
    model_arguments = ModelArguments(
        input_dim=train_dataset.graph_dataset.num_node_features,
        hidden_dim=args.embedding_dim,
        output_dim=args.embedding_dim,
        edge_dim=train_dataset.graph_dataset.num_edge_features,
        layer_type=args.layer_type,
        num_layers=args.num_layers,
        # Only for Fine-tuning
        pretrained_utterance_encoder=args.pretrained_utterance_encoder,
        pretrained_dialogue_encoder_weights_path=args.
        pretrained_dialogue_encoder_weights_path,
        pretrained_dialogue_encoder_encoder_weights_path=args.
        pretrained_dialogue_encoder_encoder_weights_path,
        # Below are the parameters for GNN Layer class (collecting to kwargs)
        num_heads=args.num_heads,
        add_self_loops=True)
    model = build_model(args=model_arguments)
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))
    # model.add_soft_prompt()
    logger.info('Building Personalized Dialogue Generator model')

    # Build trainer
    training_config = TrainingConfigs(device=device,
                                      distributed=args.distributed,
                                      dataset_name='ConvAI2',
                                      checkpoint=run_dir,
                                      learner=args.optimizer,
                                      batch_size=args.batch_size,
                                      epochs=args.epochs,
                                      lr=args.lr,
                                      weight_decay=args.weight_decay,
                                      endure_times=args.endure_times)
    if args.distributed:
        training_config.accelerator = accelerator

    trainer_cls = get_trainer_class('generator', ModelTrainMode('finetuning'))
    trainer = trainer_cls(config=training_config,
                          logger=logger,
                          model=model,
                          train_data=train_loader,
                          valid_data=valid_loader)

    if not args.do_inference:
        model.config.save_pretrained(run_dir / 'model')
        tokenizer.save_pretrained(run_dir / 'tokenizer')
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
    main(parse_generator_args())
