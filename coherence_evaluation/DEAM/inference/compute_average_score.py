import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input_file_path',
    type=str,
    action='append',
    nargs='+',
    help='Path to the file containing the scores. You can add multiple paths.')
args = parser.parse_args()

path_list = [n for arg in args.input_file_path for n in arg]
for data_path in path_list:
    scores = []
    with open(data_path) as f:
        lines = f.readlines()

        for line in lines:
            score = line.split('</UTT>')[1]
            scores.append(float(score))

    print(f'{data_path.split("/")[-1]} average score (DEAM): {np.mean(scores)}')
    print(f'{data_path.split("/")[-1]} std: {np.std(scores)}')
    print('-' * 30)
