import argparse
import json
import os
import time

import openai
from build_dataset import parse_convai2_data
from dotenv import load_dotenv
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.task_prompt import (
    CHERENCE_ANNOTATION_SEPARATED_PROMPT_v3,
    COHERENCE_ANNOTATION_SEPARATED_PROMPT_FEW_SHOTS_EXAMPLE_v2,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

load_dotenv()

openai.OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletion_with_backoff(**kwargs) -> openai.ChatCompletion:
    return client.chat.completions.create(**kwargs)


def call_gpt_api(prompt: str, model='gpt-3.5-turbo', max_tokens=1024) -> str:
    messages = [{
        'role': 'system',
        'content': SYSTEM_PROMPT
    }, {
        'role': 'user',
        'content': prompt
    }]

    completion = chatcompletion_with_backoff(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=messages,
    )

    res = completion.choices[0].message.content

    return res


def main(args):
    model = args.model
    sample_size = args.sample_size
    start_index = args.start_index
    output_file_path = args.output
    input_file_path = args.input

    persona, query, response, cand = parse_convai2_data(input_file_path)
    dataset = []
    for i in range(len(query)):
        diaglogues = []
        for j, (q, r) in enumerate(zip(query[i], response[i])):
            diaglogues.append(q)
            diaglogues.append(r)

        coherences = []
        for j, source_utterance in enumerate(diaglogues):
            for k, target_utterance in enumerate(diaglogues[j + 1:]):
                coherences.append({
                    'from': source_utterance,
                    'to': target_utterance,
                    'from_index': j,
                    'to_index': j + k + 1,
                    'labels': [],
                    'labeler': ''
                })

        dataset.append({
            'global_index': i,
            'personas': persona[i],
            'dialogues': diaglogues,
            'coherence': coherences,
        })

    print('Size of the dataset (row): ', len(dataset))
    if sample_size > 0:
        end_index = start_index + sample_size
    else:
        end_index = len(dataset)

    dataset = dataset[start_index:end_index]
    print('Start index: ', start_index)
    print('End index: ', end_index)
    print('Size of the sample: ', len(dataset))

    prompt_template = CHERENCE_ANNOTATION_SEPARATED_PROMPT_v3
    demonstration = COHERENCE_ANNOTATION_SEPARATED_PROMPT_FEW_SHOTS_EXAMPLE_v2
    try:
        for i, conv in enumerate(dataset):
            coherences = conv['coherence']
            global_index = conv['global_index']

            for j, item in enumerate(coherences):
                input_data = 'Sentence 1: ' + item[
                    'from'] + '\n' + 'Sentence 2: ' + item['to']

                assert global_index == i + start_index
                print(global_index, j, '=' * 30)
                print(input_data)
                print('->')

                prompt = prompt_template.format(demonstration=demonstration,
                                                input=input_data)
                response = call_gpt_api(prompt, model, max_tokens=4096)
                dataset[i]['coherence'][j]['labels'] = response.split(', ')
                dataset[i]['coherence'][j]['labeler'] = model

                print(response)
                print(response.split(', '))
                print()

                time.sleep(0.5)
    except ValueError as e:
        print(e)
    finally:
        with open(output_file_path, 'w') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)

        print(f'Annotations are saved to {output_file_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        choices=[
            'gpt-4o',
            'gpt-4',  # GPT4, 8k
            'gpt-4-turbo',
            'gpt-4-0125-preview',  # new model: GPT4-turbo, low cost, 128k
            'gpt-3.5-turbo',  # legacy, 4k
            'gpt-3.5-turbo-16k',  # legacy, 16k
            'gpt-3.5-turbo-0125',  # new model: GPT3.5-turbo, low cost, 16k
        ])
    parser.add_argument('--sample-size',
                        type=int,
                        default=-1,
                        help='Number of samples to use.')
    parser.add_argument('--start-index',
                        type=int,
                        default=0,
                        help='Index of the first sample to use.')
    args = parser.parse_args()

    main(args)
