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

    if isinstance(input_text, str):
        outputs = llm.generate(get_prompt_func(input_text=input_text),
                               sampling_params)
    else:
        input_prompts = [get_prompt_func(input_text=p) for p in input_text]
        outputs = llm.generate(input_prompts, sampling_params)

    predictions = [output.outputs[0].text for output in outputs]
    return predictions


def get_prompt(input_text: str, model_name: str) -> str:
    r"""Generate a prompt for the given strategy.

    Args:
        input_text (str): input text

    Returns:
        str: prompt
    """

    if 'Llama-3' in model_name:
        return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            You are provided with two sentences from a dialogue. Your task is to analyze the coherence relationship between these two sentences, specifically focusing on how Sentence 2 responds to or relates to Sentence 1. Assign appropriate coherence labels to these relationships.
            The coherence labels you may use are:
            - **Acknowledgment**: Sentence 2 acknowledges or confirms what was said in Sentence 1.
            - **Elaboration**: Sentence 2 provides more details or expands on Sentence 1.
            - **Explanation**: Sentence 2 explains or provides a reason for Sentence 1.
            - **QA**: Sentence 2 answers a question posed in Sentence 1.
            - **Comment**: Sentence 2 adds a comment to Sentence 1.
            - **Clarification-Question**: Sentence 2 asks for clarification about Sentence 1.
            - **Continuation**: Sentence 2 continues the idea or topic introduced in Sentence 1.
            - **Conditional**: Sentence 2 provides a condition related to Sentence 1.
            - **Alternation**: Sentence 2 presents an alternative to Sentence 1.
            - **Question-Elaboration**: Sentence 2 elaborates on a question posed in Sentence 1.
            - **Result**: Sentence 2 states a result or outcome of Sentence 1.
            - **Background**: Sentence 2 provides background information related to Sentence 1.
            - **Narration**: Sentence 2 narrates a sequence of events related to Sentence 1.
            - **Correction**: Sentence 2 corrects information in Sentence 1.
            - **Parallel**: Sentence 2 presents parallel or similar information to Sentence 1.
            - **Contrast**: Sentence 2 contrasts with Sentence 1.
            - **Topic Shift**: Sentence 2 shifts to a different topic from Sentence 1.

            A relationship can have multiple labels if it serves multiple functions or has multiple characteristics. However, you should assign no more than four labels to any relationship.
            If the two sentences are incoherent or do not have any relationship (difficult to assign a coherent relationship), return "None".

            Ensure that your labeling accurately reflects how Sentence 2 responds to or relates to Sentence 1.
            For each coherence labels separated by commas. Please refer to the following examples for the output format of the coherence labels.
            Please do not provide any rationale or explanation for your labels. Only provide the labels.
            Please do not always provide the same first label for each pair of sentences. Ensure that your labeling accurately reflects how Sentence 2 responds to or relates to Sentence 1.

            First give you an example to help you understand the task:
            [Dialogues]
            Sentence 1: Hello what are doing today?
            Sentence 2: I am good , i just got off work and tired, i have two jobs.
            [Coherence Labels]
            QA, Explanation
            [Dialogues]
            Sentence 1: Hello what are doing today?
            Sentence 2: I just got done watching a horror movie.
            [Coherence Labels]
            Topic Shift
            [Dialogues]
            Sentence 1: Oh that is a good profession must make alot of money.
            Sentence 2: It is a comfortable living. I like spending time with family.
            [Coherence Labels]
            Comment, Elaboration
            [Dialogues]
            Sentence 1: yes but not right now . i am a red head like my mom and dad
            Sentence 2: because i want to know if you are telling the truth while we are talking here .
            [Coherence Labels]
            Conditional, Clarification-Question
            [Dialogues]
            Sentence 1: I work in the movies as well.
            Sentence 2: Yes it is neat, I stunt double, it is so much fun and hard work.
            [Coherence Labels]
            Acknowledgment, Elaboration
            [Dialogues]
            Sentence 1: wow! I do love a good horror movie. Loving this cooler weather.
            Sentence 2: My dad was always busy working at home depot.
            [Coherence Labels]
            Background
            [Dialogue]
            Sentence 1: I've been trying to eat healthier lately.
            Sentence 2: That's great! I think it's really important to take care of your body. By the way, have you tried those new vegan restaurants downtown?
            [Coherence Labels]
            Acknowledgment, Elaboration, Topic Shift
            [Dialogue]
            Sentence 1: I've been feeling really stressed out at work lately.
            Sentence 2: I'm sorry to hear that. Have you considered talking to a therapist? Sometimes, just having someone to talk to can make a big difference. By the way, do you want to join us for a movie night this weekend?
            [Coherence Labels]
            Acknowledgment, Clarification-Question, Elaboration, Topic Shift

            Then give you the dialogues of the task:
            [Dialogues]
            {input_text}
            [Coherence Labels]
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''

    elif 'Llama-2' in model_name:
        return f'''[INST]<<SYS>>
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            <</SYS>>
            You are provided with two sentences from a dialogue. Your task is to analyze the coherence relationship between these two sentences, specifically focusing on how Sentence 2 responds to or relates to Sentence 1. Assign appropriate coherence labels to these relationships.
            The coherence labels you may use are:

            - **Acknowledgment**: Sentence 2 acknowledges or confirms what was said in Sentence 1.
            - **Elaboration**: Sentence 2 provides more details or expands on Sentence 1.
            - **Explanation**: Sentence 2 explains or provides a reason for Sentence 1.
            - **QA**: Sentence 2 answers a question posed in Sentence 1.
            - **Comment**: Sentence 2 adds a comment to Sentence 1.
            - **Clarification-Question**: Sentence 2 asks for clarification about Sentence 1.
            - **Continuation**: Sentence 2 continues the idea or topic introduced in Sentence 1.
            - **Conditional**: Sentence 2 provides a condition related to Sentence 1.
            - **Alternation**: Sentence 2 presents an alternative to Sentence 1.
            - **Question-Elaboration**: Sentence 2 elaborates on a question posed in Sentence 1.
            - **Result**: Sentence 2 states a result or outcome of Sentence 1.
            - **Background**: Sentence 2 provides background information related to Sentence 1.
            - **Narration**: Sentence 2 narrates a sequence of events related to Sentence 1.
            - **Correction**: Sentence 2 corrects information in Sentence 1.
            - **Parallel**: Sentence 2 presents parallel or similar information to Sentence 1.
            - **Contrast**: Sentence 2 contrasts with Sentence 1.
            - **Topic Shift**: Sentence 2 shifts to a different topic from Sentence 1.

            A relationship can have multiple labels if it serves multiple functions or has multiple characteristics. However, you should assign no more than four labels to any relationship.
            If the two sentences are incoherent or do not have any relationship (difficult to assign a coherent relationship), return "None".

            Ensure that your labeling accurately reflects how Sentence 2 responds to or relates to Sentence 1.
            For each coherence labels separated by commas. Please refer to the following examples for the output format of the coherence labels.
            Please do not provide any rationale or explanation for your labels. Only provide the labels.

            First give you an example to help you understand the task:
            [Dialogues]
            Sentence 1: Hello what are doing today?
            Sentence 2: I am good , i just got off work and tired, i have two jobs.
            [Coherence Labels]
            QA, Explanation
            [Dialogues]
            Sentence 1: Hello what are doing today?
            Sentence 2: I just got done watching a horror movie.
            [Coherence Labels]
            Topic Shift
            [Dialogues]
            Sentence 1: Oh that is a good profession must make alot of money.
            Sentence 2: It is a comfortable living. I like spending time with family.
            [Coherence Labels]
            Comment, Elaboration

            Then give you the dialogue history of the task, Just generate the response without explanations:
            [Dialogues]
            {input_text}
            [Coherence Labels]
            [/INST]
        '''

    return f'''[INST]You are provided with two sentences from a dialogue. Your task is to analyze the coherence relationship between these two sentences, specifically focusing on how Sentence 2 responds to or relates to Sentence 1. Assign appropriate coherence labels to these relationships.
        The coherence labels you may use are:
        - **Comment**: Sentence 2 adds a comment to Sentence 1.
        - **Clarification-Question**: Sentence 2 asks for clarification about Sentence 1.
        - **Elaboration**: Sentence 2 provides more details or expands on Sentence 1.
        - **Acknowledgment**: Sentence 2 acknowledges or confirms what was said in Sentence 1.
        - **Continuation**: Sentence 2 continues the idea or topic introduced in Sentence 1.
        - **Explanation**: Sentence 2 explains or provides a reason for Sentence 1.
        - **Conditional**: Sentence 2 provides a condition related to Sentence 1.
        - **QA**: Sentence 2 answers a question posed in Sentence 1.
        - **Alternation**: Sentence 2 presents an alternative to Sentence 1.
        - **Question-Elaboration**: Sentence 2 elaborates on a question posed in Sentence 1.
        - **Result**: Sentence 2 states a result or outcome of Sentence 1.
        - **Background**: Sentence 2 provides background information related to Sentence 1.
        - **Narration**: Sentence 2 narrates a sequence of events related to Sentence 1.
        - **Correction**: Sentence 2 corrects information in Sentence 1.
        - **Parallel**: Sentence 2 presents parallel or similar information to Sentence 1.
        - **Contrast**: Sentence 2 contrasts with Sentence 1.
        - **Topic Shift**: Sentence 2 shifts to a different topic from Sentence 1.

        A relationship can have multiple labels if it serves multiple functions or has multiple characteristics. However, you should assign no more than four labels to any relationship.
        If the two sentences are incoherent or do not have any relationship (difficult to assign a coherent relationship), return "None".

        Ensure that your labeling accurately reflects how Sentence 2 responds to or relates to Sentence 1.
        For each coherence labels separated by commas. Please refer to the following examples for the output format of the coherence labels.
        Please do not provide any rationale or explanation for your labels. Only provide the labels.

        First give you an example to help you understand the task:
        [Dialogues]
        Sentence 1: Hello what are doing today?
        Sentence 2: I am good , i just got off work and tired, i have two jobs.
        [Coherence Labels]
        QA, Explanation
        [Dialogues]
        Sentence 1: Hello what are doing today?
        Sentence 2: I just got done watching a horror movie.
        [Coherence Labels]
        Topic Shift
        [Dialogues]
        Sentence 1: Oh that is a good profession must make alot of money.
        Sentence 2: It is a comfortable living. I like spending time with family.
        [Coherence Labels]
        Comment, Elaboration

        Then give you the dialogues of the task, Just generate the response without explanations, and follow the response format "label1, label2, label3, ...":
        [Dialogues]
        {input_text}
        [Coherence Labels]
        [/INST]
    '''


def main(args: argparse.Namespace):
    persona, query, response, cand = parse_convai2_data(args.input_file_path)
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
        for j, cur_utterance in enumerate(diaglogues):
            for k, other_utterance in enumerate(diaglogues[j + 1:]):
                coherence.append({
                    'from': cur_utterance,
                    'to': other_utterance,
                    'from_index': j,
                    'to_index': j + k + 1,
                    'labels': []
                })

        dataset.append({
            'global_index': i,
            'personas': persona[i],
            'dialogues': diaglogues,
            'coherence': coherence,
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

    max_num_seqs = 256
    llm = prepare_model(args.model,
                        args.quantization,
                        max_num_seqs=max_num_seqs)

    try:
        for i, data in enumerate(dataset):
            coherence = data['coherence']
            global_index = data['global_index']

            batch_input_data = []
            for j, item in enumerate(coherence):
                input_data = 'Sentence 1: ' + item[
                    'from'] + '\n' + 'Sentence 2: ' + item['to']
                batch_input_data.append(input_data)

            response = generate(input_text=batch_input_data,
                                llm=llm,
                                get_prompt_func=partial(get_prompt,
                                                        model_name=args.model),
                                args=args)

            for j, (item, resp) in enumerate(zip(coherence, response)):
                input_data = 'Sentence 1: ' + item[
                    'from'] + '\n' + 'Sentence 2: ' + item['to']
                assert global_index == i + start_index
                print(global_index, j, '=' * 30)
                print(input_data)
                print('->')

                dataset[i]['coherence'][j]['labels'] = resp.split(', ')
                dataset[i]['coherence'][j]['labeler'] = args.model
                print(resp)
                print(resp.split(', '))
                print()

    finally:
        output_paths = args.output_file_path.split('/')
        output_path = '/'.join(output_paths[:-1]) + '/' + str(
            start_index) + '-' + str(end_index) + '_' + output_paths[-1]
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)

        print(f'Annotations are saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-i', '--input_file_path', type=str, required=True)
    parser.add_argument('-o', '--output_file_path', type=str, required=True)

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
