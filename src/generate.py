import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BartTokenizer,
    BertTokenizer,
    RobertaTokenizer,
)

sys.path.append('.')

from data.collator import PersonalizedDialogueGenerationDataCollator
from data.dataset import (
    ConvAI2ForDialogueGraphEncodingDataset,
    ConvAI2ForPersonalizedDialogueGenerationDataset,
)
from data.processor import ConvAI2DataProcessor, DialogueGraphDataProcessMode
from data.processor import DialogueGraphDataProcessMode as ProcessMode
from models.modeling_generator import PersonalizedDialogueGenerator
from utils.data_utils import load_pickle
from utils.model_configs import PersonalizedDialogueGeneratorConfig


def prepare_dataset(
    args: argparse.Namespace, generator_tokenizer: BartTokenizer
) -> tuple[ConvAI2ForPersonalizedDialogueGenerationDataset,
           ConvAI2ForDialogueGraphEncodingDataset]:
    if args.pretrained_utterance_encoder == 'bert':
        utterance_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.pretrained_utterance_encoder == 'roberta':
        utterance_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif args.pretrained_utterance_encoder == 'none':
        utterance_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        processor = ConvAI2DataProcessor(k_hop=args.k_hop,
                                         mode=DialogueGraphDataProcessMode(
                                             args.process_mode))

        convai2_graph_dataset = ConvAI2ForDialogueGraphEncodingDataset(
            root=args.data_dir,
            raw_file_name=args.data_name,
            processed_file_dir=args.processed_data_dir,
            processed_file_name=args.processed_data_name,
            transform=processor.transform,
            pre_transform=processor.pre_transform,
            pre_filter=processor.pre_filter,
            utterance_tokenizer=utterance_tokenizer,
            reverse_edge=args.reverse_edge,
            directed=args.directed)

        convai2_dataset = ConvAI2ForPersonalizedDialogueGenerationDataset(
            dataset=load_pickle(convai2_graph_dataset.raw_paths[0]),
            graph_dataset=convai2_graph_dataset,
            tokenizer=generator_tokenizer,
            nearest_k_turn=3)

    return convai2_dataset, convai2_graph_dataset


def main(args):
    device = 'cuda'
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name_or_path)

    model = PersonalizedDialogueGenerator.from_pretrained(
        args.model_name_or_path,
        coherence_attn_strategy=args.coherence_attn_strategy,
        graph_encoder_strategy=args.graph_encoder_strategy)
    model = model.to(device)

    model_config = PersonalizedDialogueGeneratorConfig.from_pretrained(
        args.model_name_or_path)

    # Load dataset
    testing_dataset, testing_graph_dataset = prepare_dataset(
        args=args, generator_tokenizer=tokenizer)

    # Build dataloader
    test_loader = DataLoader(
        dataset=testing_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=PersonalizedDialogueGenerationDataCollator(),
        pin_memory=True)

    raw_dataset = load_pickle(testing_graph_dataset.raw_paths[0])

    prediction = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader),
                                     total=len(test_loader)):
            batch['dialogue_encoder_input'] = {
                k: v.to(device)
                for k, v in batch['dialogue_encoder_input'].items()
            }
            batch['generator_input'] = {
                k: v.to(device)
                for k, v in batch['generator_input'].items()
            }

            input_seq = tokenizer.batch_decode(
                batch['generator_input']['input_ids'], skip_special_tokens=True)
            # ground_truth = tokenizer.batch_decode(
            #     batch['generator_input']['decoder_input_ids'],
            #     skip_special_tokens=True)

            del batch['generator_input']['decoder_input_ids']
            del batch['generator_input']['decoder_attention_mask']

            decoder_inputs = tokenizer(
                ['<response>'] * len(batch['generator_input']['input_ids']),
                return_tensors='pt',
                padding=False,
                add_special_tokens=False).to(device)

            outputs = model.generate(
                **batch,
                decoder_input_ids=decoder_inputs.input_ids,
                decoder_attention_mask=decoder_inputs.attention_mask,
                max_new_tokens=512,
                do_sample=True,
                early_stopping=False,
                num_beams=model_config.task_specific_params['generation']
                ['num_beams']
                if 'generation' in model_config.task_specific_params else 10,
                temperature=0.9,
                top_p=0.7,
                repetition_penalty=1.6,
                length_penalty=-1.0,
                min_length=20,
                decoder_start_token_id=0,
                tau=args.tau,
                top_k_relations=args.top_k_relations,
            )
            output_seq = tokenizer.batch_decode(outputs,
                                                skip_special_tokens=True)[0]
            output_seq = output_seq.replace('<response>', '').strip()

            dialogue = raw_dataset[batch_idx]['whole_dialogue']
            ground_truth = dialogue[-1]

            print(f'Input: {input_seq}')
            print(f'Prediction: {output_seq}')
            print(f'Ground-truth: {ground_truth}')
            print('-' * 30 + '\n')

            prediction.append({
                'global_index':
                raw_dataset[batch_idx]['global_index'],
                'input_sequence':
                input_seq[0].strip(),
                'persona':
                raw_dataset[batch_idx]['persona'],
                'context':
                dialogue[:-2],
                'query':
                dialogue[-2],
                'ground_truth_response':
                ground_truth,
                'predicted_response':
                output_seq,
            })

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
            f'{args.output_dir}/{args.model_name_or_path.split("/")[-2]}.json',
            'w') as f:
        json.dump(prediction, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Personalized dialogue generation test')

    parser.add_argument('-m', '--model_name_or_path', type=str, required=True)
    parser.add_argument('-t',
                        '--tokenizer_name_or_path',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='output directory',
                        required=True)

    parser.add_argument('--data_dir',
                        type=str,
                        default='dataset/ConvAI2/',
                        help='path to the dataset directory')
    parser.add_argument('--processed_data_dir', type=str, default=None)
    parser.add_argument(
        '--data_name',
        type=str,
        default='valid_self_original_coherence.pkl',
    )
    parser.add_argument('--processed_data_name', type=str, default=None)
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of workers')
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

    parser.add_argument('--batch_size', type=int, default=4, help='batch size')

    parser.add_argument('--pretrained_utterance_encoder',
                        choices=['none', 'bert', 'roberta'],
                        default='none',
                        help='pretrained model for utterance/persona encoder')

    # For personalized dialogue generation
    parser.add_argument('--tau', type=float, default=0.2, required=False)
    parser.add_argument('--top_k_relations',
                        type=int,
                        default=3,
                        required=False)

    # Ablaition Study
    parser.add_argument('--coherence_attn_strategy',
                        type=str,
                        choices=['SP', 'Emb', 'SP+Emb'],
                        default='SP+Emb',
                        help='coherence attention strategy')
    parser.add_argument('--graph_encoder_strategy',
                        type=str,
                        choices=['Attn', 'Add', 'C', 'P', 'Random', 'None'],
                        default='Attn',
                        help='graph encoder strategy')

    args = parser.parse_args()
    main(args)
