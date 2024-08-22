import argparse
import json
from functools import partial
from typing import Callable, Union

from build_dataset import parse_convai2_data
from vllm import LLM, SamplingParams


def prepare_model(model_name_or_path: str, quantization: str, max_num_seqs=256):
    quantization = None if quantization == 'none' else quantization
    return LLM(model=model_name_or_path,
               tensor_parallel_size=2,
               max_num_seqs=max_num_seqs,
               quantization=quantization)


def generate(input_text: Union[str, list[str]], llm: LLM,
             get_prompt_func: Callable, args: argparse.Namespace) -> list[str]:
    r"""Generate predictions for the given input text / list of input text.

    Args:
        input_text (Union[str, list[str]]): input text / list of input text
        model_name_or_path (str): model name or path
        get_prompt_func (Callable): function to generate a prompt
        args (argparse.Namespace): arguments for generation

    Returns:
        list[str]: predictions for the given input text / list of input text
    """

    sampling_params = SamplingParams(max_tokens=args.max_tokens,
                                     temperature=args.temperature,
                                     top_k=args.top_k,
                                     top_p=args.top_p,
                                     frequency_penalty=0.2)

    # llm = LLM(model=model_name_or_path, tensor_parallel_size=4, max_num_seqs=1)
    if isinstance(input_text, str):
        outputs = llm.generate(get_prompt_func(input_text=input_text),
                               sampling_params)
    else:
        input_prompts = [get_prompt_func(input_text=p) for p in input_text]
        outputs = llm.generate(input_prompts, sampling_params)

    predictions = [output.outputs[0].text for output in outputs]
    return predictions


def get_prompt(input_text: str, type: str) -> str:
    r"""Generate a prompt for the given strategy.

    Args:
        input_text (str): input text

    Returns:
        str: prompt
    """

    if type == 'query':
        return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                You are provided with a single query from a dialogue. Your task is to analyze the query and identify the key phrases that are crucial for determining how to respond. 
                Focus on identifying meaningful phrases that represent specific concerns, topics, or objects mentioned in the query that are relevant to formulating an appropriate response. 
                Avoid extracting single words unless they stand alone as a significant part of the query.
                Please provide the identified key phrases as a list, separated by a comma and a space. Your output should be concise and only include phrases that directly influence the response formulation.

                First give you an example to help you understand the task:
                [Query]
                Are you afraid of spiders then? Spiderman is my fav comic book.
                [Keywords]
                afraid of spiders, spiderman, comic book
                [Query]
                I just got done watching a horror movie.
                [Keywords]
                watching, horror movie
                [Query]
                That's great! I think it's really important to take care of your body. By the way, have you tried those new vegan restaurants downtown?
                [Keywords]
                take care of your body, new vegan restaurants downtown
                [Query]
                My dad was always busy working at home depot.
                [Keywords]
                busy working at home depot

                Then give you the dialogues of the task:
                [Query]
                {input_text}
                [Keywords]
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            '''
    elif type == 'persona':
        return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                You are provided with a description of a persona. Your task is to analyze the description and identify key phrases that represent important persona features. 
                These key phrases should encapsulate specific characteristics, behaviors, or preferences mentioned in the description that are crucial to understanding the persona.
                Focus on extracting meaningful phrases that convey significant aspects of the persona. Avoid extracting single words unless they stand alone as a significant trait or preference.
                Please provide the identified key phrases as a list, separated by a comma and a space. Your output should be concise and only include phrases that directly describe important features of the persona.

                First give you an example to help you understand the task:
                [Query]
                I read twenty books a year.
                [Keywords]
                read books, twenty books a year
                [Query]
                I'm a stunt double as my second job.
                [Keywords]
                stunt double, second job
                [Query]
                I only eat kosher.
                [Keywords]
                only eat kosher
                [Query]
                I was raised in a single parent household.
                [Keywords]
                single parent household

                Then give you the dialogues of the task:
                [Query]
                {input_text}
                [Keywords]
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            '''


def main(args: argparse.Namespace):
    persona, query, response, cand = parse_convai2_data(args.data)
    dataset = []
    for i in range(len(query)):
        dialogues_string = []
        diaglogues = []

        for j, (q, r) in enumerate(zip(query[i], response[i])):
            diaglogues.append(q)
            diaglogues.append(r)
            dialogues_string.append(f'User1:{q.capitalize()}')
            dialogues_string.append(f'User2:{r.capitalize()}')

        dataset.append({
            'global_index': i,
            'personas': persona[i],
            'dialogues': diaglogues
        })

    start_index = args.start_index
    sample_size = args.sample_size

    print('Size of the dataset (row): ', len(dataset))
    if sample_size > 0:
        end_index = start_index + sample_size
    else:
        end_index = len(dataset)

    dataset = dataset[start_index:end_index]
    print('Start index: ', start_index)
    print('End index: ', end_index)
    print('Size of the sample: ', len(dataset))
    print('Extraction Type: ', args.type)

    max_num_seqs = 256
    llm = prepare_model(args.model,
                        args.quantization,
                        max_num_seqs=max_num_seqs)

    try:
        for i, data in enumerate(dataset):
            global_index = data['global_index']

            target_item = data['dialogues'] if args.type == 'query' else data[
                'personas']
            batch_input_data = [item for item in target_item]
            response = generate(input_text=batch_input_data,
                                llm=llm,
                                get_prompt_func=partial(get_prompt,
                                                        type=args.type),
                                args=args)

            keywords = []
            for j, (utterance,
                    resp) in enumerate(zip(batch_input_data, response)):
                assert global_index == i + start_index
                print(global_index, j, '=' * 30)
                print(utterance)
                print('->')

                keywords.append({
                    'index': j,
                    'sentence': utterance,
                    'labels': resp.split(', '),
                    'labeler': args.model
                })

                print(resp)
                print(resp.split(', '))
                print()

            dataset[i]['keyword'] = keywords
    finally:
        output_paths = args.output.split('/')
        output_path = '/'.join(output_paths[:-1]) + '/' + str(
            start_index) + '-' + str(end_index) + '_' + output_paths[-1]
        with open(output_path, 'w', encoding='UTF-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=True)

        print(f'Annotations are saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    parser.add_argument('--type',
                        type=str,
                        default='query',
                        choices=['query', 'persona'])

    # Data parameters
    parser.add_argument('--start-index',
                        type=int,
                        default=0,
                        help='Index of the first sample to use.')
    parser.add_argument('--sample-size', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)

    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--top_p', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=2048)

    parser.add_argument('-q',
                        '--quantization',
                        choices=[
                            'aqlm', 'awq', 'fp8', 'gptq', 'squeezellm',
                            'gptq_marlin', 'marlin', 'none'
                        ],
                        default='none')

    args = parser.parse_args()

    main(args)
