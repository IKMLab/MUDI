import argparse
import json
import time

import numpy as np
from build_dataset import parse_convai2_data
from transformers import pipeline


def main(args):
    sample_size = args.sample_size
    start_index = args.start_index
    output_file_path = args.output
    input_file_path = args.input
    model_name = args.model

    persona, query, response, cand = parse_convai2_data(input_file_path)
    dataset = []
    for i in range(len(query)):
        dialogues_string = []
        diaglogues = []

        for j, (q, r) in enumerate(zip(query[i], response[i])):
            diaglogues.append(q)
            diaglogues.append(r)
            dialogues_string.append(f'User1:{q.capitalize()}')
            dialogues_string.append(f'User2:{r.capitalize()}')

        coherence = []
        for j, utterance in enumerate(diaglogues):
            for k, other_utterance in enumerate(diaglogues[j + 1:]):
                coherence.append({
                    'from': utterance,
                    'to': other_utterance,
                    'from_index': j,
                    'to_index': j + k + 1,
                    'labels': []
                })

        dataset.append({
            'global_index': i,
            'personas': persona[i],
            'dialogues': diaglogues,
            'coherence': coherence
        })

    dataset = dataset[start_index:start_index + sample_size]

    classifier = pipeline(model=model_name)
    coherence_labels = [
        'Comment', 'Clarification-Question', 'Elaboration', 'Acknowledgment',
        'Continuation', 'Explanation', 'Conditional', 'QA', 'Alternation',
        'Question-Elaboration', 'Result', 'Background', 'Narration',
        'Correction', 'Parallel', 'Contrast', 'Topic Shift'
    ]
    top_n = args.top_n

    for i, data in enumerate(dataset):
        coherence = data['coherence']
        for j, item in enumerate(coherence):
            input_data = 'Sentence1:' + item['from'] + ' Sentence2:' + item['to']

            print(i + start_index, j, '=' * 30)
            print(input_data)
            print('->')

            response = classifier(
                input_data,
                candidate_labels=coherence_labels,
                multi_label=True,
                hypothesis_template=
                'The coherence relations between the Sentence1 and Sentence2 is: {}.'
            )
            top_n_prob_indices = np.argsort(response['scores'])[::-1][:top_n]
            top_n_labels = [response['labels'][i] for i in top_n_prob_indices]
            top_n_probs = [response['scores'][i] for i in top_n_prob_indices]

            print(list(zip(top_n_labels, top_n_probs)))
            dataset[i]['coherence'][j]['labels'] = top_n_labels
            dataset[i]['coherence'][j]['label_probs'] = top_n_probs

            print(top_n_labels, top_n_probs)

            time.sleep(0.5)

    with open(output_file_path, 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        choices=['facebook/bart-large-mnli', 'roberta-large-mnli'],
        required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--sample-size',
                        type=int,
                        default=1000,
                        help='Number of samples to use.')
    parser.add_argument('--start-index',
                        type=int,
                        default=0,
                        help='Index of the first sample to use.')
    parser.add_argument('-n',
                        '--top-n',
                        type=int,
                        default=3,
                        help='Preserve top n predictions.')
    args = parser.parse_args()

    main(args)
