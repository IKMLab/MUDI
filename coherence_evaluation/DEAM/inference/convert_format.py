import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model_type',
                    type=str,
                    choices=['lmedr', 'paa', 'gpt', 'gold', 'mudi'])
parser.add_argument('-i', '--input_file_path', type=str, help='Input file path')
args = parser.parse_args()

if args.model_type == 'lmedr':
    df = pd.read_csv(args.input_file_path, sep='\t')
    df['query'] = df['dialogue'].apply(
        lambda x: [i.replace('User:', '').replace('Assistant:', '') for i in x])
    df['persona'] = df['persona'].apply(lambda x: ' '.join(x))
    df['response'] = df['predict_response']
elif args.model_type == 'paa':
    df = pd.read_json(args.input_file_path)
    df['response'] = df['pred']
    df['persona'] = df['persona'].apply(lambda x: ' '.join(x))
elif args.model_type == 'gpt':
    df = pd.read_json(args.input_file_path)
    df['query'] = df['dialogue'].apply(
        lambda x: [i.replace('User:', '').replace('Assistant:', '') for i in x])
    df['response'] = df['predict_response']
    df['persona'] = df['persona'].apply(lambda x: ' '.join(x))
elif args.model_type == 'gold':
    df = pd.read_json(args.input_file_path)
    df['query'] = df['dialogue'].apply(
        lambda x: [i.replace('User:', '').replace('Assistant:', '') for i in x])
    df['response'] = df['ground_truth_response']
    df['persona'] = df['persona'].apply(lambda x: ' '.join(x))
elif args.model_type == 'mudi':
    df = pd.read_json(args.input_file_path)
    query = df['query'].tolist()
    context = df['context'].tolist()
    convs = []
    for c, q in zip(context, query):
        convs.append(c + [q])

    df['query'] = pd.Series(convs)
    df['response'] = df['predicted_response']
    df['persona'] = df['persona'].apply(lambda x: ' '.join(x))

single_turn_data = []
multi_turn_data = []
current_persona = ''
dialogue_history = []
for i, row in df.iterrows():
    persona = row['persona']
    query = row['query']
    response = row['response']

    if persona != current_persona:
        current_persona = persona
        dialogue_history = []

    if args.model_type == 'paa' or isinstance(query, list):
        dialogue_history = query
        query = query[-1]
    else:
        dialogue_history.append(query)

    dialogue_history.append(response)

    single_turn_data.append('</UTT>'.join([query, response, '0']))
    multi_turn_data.append('</UTT>'.join(['</UTT>'.join(dialogue_history),
                                          '0']))

print(
    f'Save DEAM formatted results to {args.input_file_path.split(".")[0]}-single_turn.txt'
)
with open(f'{args.input_file_path.split(".json")[0]}-single_turn.txt',
          'w') as f:
    f.writelines('\n'.join(single_turn_data))

print(
    f'Save DEAM formatted results to {args.input_file_path.split(".")[0]}-multi_turn.txt'
)
with open(f'{args.input_file_path.split(".json")[0]}-multi_turn.txt', 'w') as f:
    f.writelines('\n'.join(multi_turn_data))
