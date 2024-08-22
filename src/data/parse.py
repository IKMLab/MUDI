import argparse
import json


def parse_convai2_data(data_file: str):
    with open(data_file, encoding='utf8') as f:
        persona = []
        query = []
        response = []
        cand = []
        is_persona = False
        tmp_persona = []
        tmp_query = []
        tmp_response = []
        tmp_cand = []
        first = True
        cnt = 0
        sum_u = 0
        for line in f:
            cnt += 1
            line = line.strip()
            if 'your persona: ' in line:
                if not is_persona and not first:
                    query.append(tmp_query)
                    response.append(tmp_response)
                    cand.append(tmp_cand)
                    sum_u += len(tmp_query)
                    tmp_query = []
                    tmp_response = []
                    tmp_cand = []
                first = False
                is_persona = True
                line = line.split(': ', maxsplit=1)[1]
                tmp_persona.append(line)
            else:
                if is_persona:
                    persona.append(tmp_persona)
                    is_persona = False
                    tmp_persona = []

                line = line[line.find(' ') + 1:]
                tmp_query.append(line.split('\t')[0])
                tmp_response.append(line.split('\t')[1])
                tmp_cand.append(line.split('\t')[3].split('|'))

        query.append(tmp_query)
        response.append(tmp_response)
        cand.append(tmp_cand)
        sum_u += len(tmp_query)

        assert len(query) == len(response) == len(persona) == len(cand)

    print(f'{data_file} has {len(query)} dialog and {sum_u} query')

    return persona, query, response, cand


def parse_rcc_data(data_file: str):
    conversations = []
    with open(data_file, encoding='utf-8') as f:
        for line in f:
            utterances = line.split('\t')
            utterances = [u.strip().replace('\n', '') for u in utterances]

            conversations.append(utterances)

    print(f'{data_file} has {len(conversations)} conversations')

    return conversations


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse dataset to json format.')
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        choices=['convai2', 'rcc'],
                        help='Dataset name (convai2 or rcc).',
                        required=True)
    parser.add_argument(
        '-i',
        '--input_file_path',
        type=str,
        help='Path to the input file. Only txt file is allowed.',
        required=True)
    parser.add_argument(
        '-o',
        '--output_file_path',
        type=str,
        help='Path to the save file. Only json file is allowed.',
        required=True)
    args = parser.parse_args()

    input_file_path = args.input_file_path
    output_file_path = args.output_file_path

    assert input_file_path.endswith('.txt'), 'Input file should be a txt file.'
    assert output_file_path.endswith(
        '.json'), 'Output file should be a json file.'

    formatted_data = []
    if args.dataset == 'convai2':
        persona, query, response, cand = parse_convai2_data(input_file_path)

        for i, (p, q, r,
                cand_r) in enumerate(zip(persona, query, response, cand)):
            dialogue = []
            history_q = []
            for q_item, r_item, cand_item in zip(q, r, cand_r):
                history_q.append(q_item)
                dialogue.append({
                    'query': history_q.copy(),
                    'response': r_item,
                    'cand_response': cand_item
                })
                history_q.append(r_item)

            whole_dialogue = dialogue[-1]['query'] + [dialogue[-1]['response']]
            formatted_data.append({
                'global_index': i,
                'persona': p,
                'whole_dialogue': whole_dialogue,
                'dialogue': dialogue,
            })
    elif args.dataset == 'rcc':
        conversations = parse_rcc_data(input_file_path)
        formatted_data = [{'dialogue': c} for c in conversations]

    with open(output_file_path, 'w') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)
