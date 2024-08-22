import argparse
import os

from src.data.processor import DialogueGraphDataProcessMode as ProcessMode


def parse_gnn_args():
    parser = argparse.ArgumentParser(description='Train Coherence Graph Model')

    # Dataset
    parser.add_argument('--data_dir',
                        type=str,
                        default='dataset/ConvAI2/',
                        help='path to the dataset directory')
    parser.add_argument('--processed_train_data_dir',
                        type=str,
                        default=None,
                        help='path to the processed training data directory')
    parser.add_argument('--processed_valid_data_dir',
                        type=str,
                        default=None,
                        help='path to the processed validation data directory')
    parser.add_argument(
        '--train_data_name',
        type=str,
        default='train_self_original_coherence.pkl',
        help='training data name under the data_dir',
    )
    parser.add_argument(
        '--valid_data_name',
        type=str,
        default='valid_self_original_coherence.pkl',
        help='validation data name under the data_dir',
    )
    parser.add_argument(
        '--processed_train_data_name',
        type=str,
        default=None,
        help='processed training data name under the processed_train_data_dir')
    parser.add_argument(
        '--processed_valid_data_name',
        type=str,
        default=None,
        help='processed validation data name under the processed_valid_data_dir'
    )
    parser.add_argument('--ckpt_dir',
                        type=str,
                        default='checkpoints/gnn/',
                        help='checkpoint directory')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # Wandb
    parser.add_argument('--wandb',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='use wandb or not')
    parser.add_argument('--wandb_entity', type=str, help='wandb entity')
    parser.add_argument('--wandb_project', type=str, help='wandb project')
    parser.add_argument('--wandb_run_name',
                        '-a',
                        type=str,
                        default=None,
                        help='wandb run name')

    # Data Preprcess
    parser.add_argument('--process_mode',
                        choices=[m.value for m in ProcessMode],
                        default='single_filter',
                        help='label (coherence relations) preprocess mode')
    parser.add_argument('--k_hop',
                        type=int,
                        default=3,
                        help='keep how many k-hop neighbors')
    parser.add_argument('--reverse_edge',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='reverse edge or not')
    parser.add_argument('--directed',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='directed graph or not')

    # Training
    parser.add_argument('--train_mode',
                        type=str,
                        choices=['finetuning', 'pretraining'],
                        default='finetuning',
                        help='model training mode')
    parser.add_argument('--do_inference',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='do inference only')
    parser.add_argument(
        '--cpu',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='use cpu or not, if True, then use cpu, else use gpu (cuda)')
    parser.add_argument('--distributed',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='use distributed training or not')
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--pretrained_utterance_encoder',
                        choices=['none', 'bert', 'roberta'],
                        default='none',
                        help='pretrained model for utterance/persona encoder')
    parser.add_argument('--layer_type',
                        choices=['GAT', 'GATv2', 'DialogueGAT'],
                        default='DialogueGAT',
                        help='type of GNN encoder layers')
    parser.add_argument('--num_layers',
                        type=int,
                        default=2,
                        help='number of GNN encoder layers')
    parser.add_argument(
        '--num_heads',
        type=int,
        default=4,
        help='number of attention heads if using GAT like layers')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=512,
                        help='embedding dimension of GNN layers')
    parser.add_argument('--batch_size',
                        type=int,
                        default=512,
                        help='batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01,
                        help='weight decay')
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'adamw', 'sgd', 'adagrad', 'rmsprop', 'sparse_adam'],
        default='adamw',
        help='optimizer')
    parser.add_argument(
        '--coh_rel_cls_weight',
        type=float,
        default=1.0,
        help='loss weight for coherence relations classification')
    parser.add_argument('--link_prediction_weight',
                        type=float,
                        default=1.0,
                        help='loss weight for link prediction')
    parser.add_argument('--next_resp_type_direct_weight',
                        type=float,
                        default=1.0,
                        help='loss weight for next response type prediction')
    parser.add_argument('--next_resp_type_seq_weight',
                        type=float,
                        default=1.0,
                        help='loss weight for next response type prediction')

    # Validation
    parser.add_argument(
        '--endure_times',
        type=int,
        default=10,
        help='the maximum endure epochs of loss increasing on validation')

    args = parser.parse_args()

    if 'processed_train_data_dir' not in args:
        args.processed_train_data_dir = os.path.join(args.data_dir,
                                                     'processed_train')

    if 'processed_valid_data_dir' not in args:
        args.processed_valid_data_dir = os.path.join(args.data_dir,
                                                     'processed_valid')

    return args


def parse_generator_args():
    parser = argparse.ArgumentParser(
        description='Train Personalized Dialogue Generator')

    # Dataset
    parser.add_argument('--data_dir',
                        type=str,
                        default='dataset/ConvAI2/',
                        help='path to the dataset directory')
    parser.add_argument('--processed_train_data_dir',
                        type=str,
                        default=None,
                        help='path to the processed training data directory')
    parser.add_argument('--processed_valid_data_dir',
                        type=str,
                        default=None,
                        help='path to the processed validation data directory')
    parser.add_argument(
        '--train_data_name',
        type=str,
        default='train_self_original_coherence.pkl',
        help='training data name under the data_dir',
    )
    parser.add_argument(
        '--valid_data_name',
        type=str,
        default='valid_self_original_coherence.pkl',
        help='validation data name under the data_dir',
    )
    parser.add_argument(
        '--processed_train_data_name',
        type=str,
        default=None,
        help='processed training data name under the processed_train_data_dir')
    parser.add_argument(
        '--processed_valid_data_name',
        type=str,
        default=None,
        help='processed validation data name under the processed_valid_data_dir'
    )
    parser.add_argument('--ckpt_dir',
                        type=str,
                        default='ckpts/generator/ConvAI2/',
                        help='checkpoint directory')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # Wandb
    parser.add_argument('--wandb',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='use wandb or not')
    parser.add_argument('--wandb_entity', type=str, help='wandb entity')
    parser.add_argument('--wandb_project', type=str, help='wandb project')
    parser.add_argument('--wandb_run_name',
                        '-a',
                        type=str,
                        default=None,
                        help='wandb run name')

    # Data Preprcess
    parser.add_argument('--process_mode',
                        choices=[m.value for m in ProcessMode],
                        default='single_filter',
                        help='label (coherence relations) preprocess mode')
    parser.add_argument('--k_hop',
                        type=int,
                        default=3,
                        help='keep how many k-hop neighbors')
    parser.add_argument('--reverse_edge',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='reverse edge or not')
    parser.add_argument('--directed',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='directed graph or not')

    # Training
    parser.add_argument('--train_mode',
                        type=str,
                        choices=['finetuning', 'pretraining'],
                        default='finetuning',
                        help='model training mode')
    parser.add_argument('--do_inference',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='do inference only')
    parser.add_argument(
        '--cpu',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='use cpu or not, if True, then use cpu, else use gpu (cuda)')
    parser.add_argument('--distributed',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='use distributed training or not')
    parser.add_argument('--pretrained_dialogue_encoder_weights_path',
                        type=str,
                        default=None)
    parser.add_argument('--pretrained_dialogue_encoder_encoder_weights_path',
                        type=str,
                        default=None)
    parser.add_argument('--pretrained_utterance_encoder',
                        choices=['none', 'bert', 'roberta'],
                        default='none',
                        help='pretrained model for utterance/persona encoder')
    parser.add_argument('--layer_type',
                        choices=['GAT', 'GATv2', 'RGAT', 'DialogueGAT'],
                        default='DialogueGAT',
                        help='type of GNN encoder layers')
    parser.add_argument('--num_layers',
                        type=int,
                        default=2,
                        help='number of GNN encoder layers')
    parser.add_argument(
        '--num_heads',
        type=int,
        default=4,
        help='number of attention heads if using GAT like layers')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=512,
                        help='embedding dimension of GNN layers')
    parser.add_argument('--batch_size',
                        type=int,
                        default=512,
                        help='batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01,
                        help='weight decay')
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'adamw', 'sgd', 'adagrad', 'rmsprop', 'sparse_adam'],
        default='adamw',
        help='optimizer')
    parser.add_argument(
        '--coh_rel_cls_weight',
        type=float,
        default=1.0,
        help='loss weight for coherence relations classification')
    parser.add_argument('--link_prediction_weight',
                        type=float,
                        default=1.0,
                        help='loss weight for link prediction')
    parser.add_argument('--next_resp_type_direct_weight',
                        type=float,
                        default=1.0,
                        help='loss weight for next response type prediction')
    parser.add_argument('--next_resp_type_seq_weight',
                        type=float,
                        default=1.0,
                        help='loss weight for next response type prediction')

    # Validation
    parser.add_argument(
        '--endure_times',
        type=int,
        default=10,
        help='the maximum endure epochs of loss increasing on validation')

    args = parser.parse_args()

    if 'processed_train_data_dir' not in args:
        args.processed_train_data_dir = os.path.join(args.data_dir,
                                                     'processed_train')

    if 'processed_valid_data_dir' not in args:
        args.processed_valid_data_dir = os.path.join(args.data_dir,
                                                     'processed_valid')

    return args
