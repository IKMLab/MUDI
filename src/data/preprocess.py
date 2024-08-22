import argparse
import copy
import sys
from typing import Literal

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append('.')
from src.utils.data_utils import load_json, load_jsonl, save_pickle


def load_data(data_path: str) -> list:
    if data_path.endswith('.json'):
        return load_json(data_path)
    elif data_path.endswith('.jsonl'):
        return load_jsonl(data_path)
    else:
        raise ValueError('Invalid data path')


def augment_data(dataset: list) -> list:
    augmented_data = []
    for d in dataset:
        augmented_data.append({
            'global_index': d['global_index'],
            'persona': d['persona'],
            'whole_dialogue': d['whole_dialogue'],
            'coherence': d['coherence'],
            'is_true_response': True,
        })

        cand_response = copy.deepcopy(d['dialogue'][-1]['cand_response'][:-1])
        cand_response = np.random.choice(cand_response, 1, replace=False)

        for cand_resp in cand_response:

            whole_dialogue = []
            for c in d['whole_dialogue']:
                whole_dialogue.append(c)

            whole_dialogue[-1] = cand_resp

            item = {
                'global_index': d['global_index'],
                'persona': d['persona'],
                'whole_dialogue': whole_dialogue,
                'coherence': d['coherence'],
                'is_true_response': False,
            }

            augmented_data.append(copy.deepcopy(item))

    return augmented_data


def encode_sentence(
        dataset: list, dataset_name: Literal['convai2', 'daily_dialog',
                                             'rcc']) -> list:
    sentence_embedder = SentenceTransformer(
        'all-mpnet-base-v2',
        device='cuda' if torch.cuda.is_available() else 'cpu')

    encoded_dataset = dataset.copy()
    total_len = len(dataset)
    for i, item in tqdm(enumerate(dataset),
                        desc='Generating sentence embeddings',
                        total=total_len):
        if dataset_name == 'convai2':
            personas = item['persona']
            dialogues = item['whole_dialogue']

            encoded_dataset[i]['encoded_persona'] = sentence_embedder.encode(
                personas)
            encoded_dataset[i]['encoded_dialogue'] = sentence_embedder.encode(
                dialogues)
        elif dataset_name == 'daily_dialog':
            dialogues = item['dialog']
            encoded_dataset[i]['encoded_dialogue'] = sentence_embedder.encode(
                dialogues)
        elif dataset_name == 'rcc':
            dialogues = item['dialogue']
            encoded_dataset[i]['encoded_dialogue'] = sentence_embedder.encode(
                dialogues)

    return encoded_dataset


def main(args: argparse.Namespace):
    dataset = load_data(args.input_file_path)

    if args.dataset == 'convai2' and args.augment:
        print('Augmenting the dataset...')
        dataset = augment_data(dataset)

    encoded_dataset = encode_sentence(dataset, args.dataset)

    save_pickle(args.output_file_path, encoded_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the dataset.')
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        choices=['convai2', 'rcc', 'daily_dialog'],
                        help='Dataset name.',
                        required=True)
    parser.add_argument('-i',
                        '--input_file_path',
                        type=str,
                        help='Path to the input file.',
                        required=True)
    parser.add_argument(
        '-o',
        '--output_file_path',
        type=str,
        help=
        'Path to the save dataset after preprocess. Only pickle file is allowed.',
        required=True)
    parser.add_argument('--augment',
                        action=argparse.BooleanOptionalAction,
                        help='Augment the dataset. Only for ConvAI2 dataset.',
                        default=True,
                        required=False)

    parser.parse_args()

    main(parser.parse_args())
