import argparse
import json

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i',
                    '--input',
                    type=str,
                    help='The input file',
                    action='append',
                    nargs='+')
parser.add_argument('-o', '--output', type=str, help='The output file')
args = parser.parse_args()

print(args)
input_file_paths = args.input
output_file_path = args.output

merged_data = None
for input_file_path in [p for arg in input_file_paths for p in arg]:
    if merged_data is None:
        merged_data = pd.read_json(input_file_path)
    else:
        merged_data = pd.concat(
            [merged_data, pd.read_json(input_file_path)], axis=0)

merged_data.to_json(output_file_path,
                    orient='records',
                    indent=4,
                    force_ascii=False)

with open(output_file_path) as f:
    merged_data = json.load(f)

empty_count = 0
for i, data in enumerate(merged_data):
    for j, item in enumerate(data['keyword']):
        labels = [label.strip() for label in item['labels']]

        processed_labels = []
        if 'None' in labels or 'none' in labels or 'no keywords (query is unclear)' in labels \
            or 'no keywords' in labels:
            processed_labels = []
            empty_count += 1
        else:
            for label in labels:
                processed_labels.append(label)

        print(i, j, processed_labels)
        merged_data[i]['keyword'][j]['labels'] = processed_labels

with open(output_file_path, 'w') as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=True)

print(f'Merged data is saved to {output_file_path}')
print(f'empty count: {empty_count}')
