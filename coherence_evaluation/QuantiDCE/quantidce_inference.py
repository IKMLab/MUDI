import pandas as pd
import util.main_utils as main_utils
import util.opt as opt
from tqdm import tqdm

if __name__ == '__main__':
    args = opt.parse_coherence_eval_opt()

    model = main_utils.get_model(args)
    data = pd.read_json(args.input_file_path)

    with open(args.output_file_path, 'w') as f:
        single_turn_scores = []
        multi_turn_scores = []
        for i, dialogue in tqdm(data.iterrows(), total=len(data)):
            context = dialogue['context']
            query = dialogue['query']
            response = dialogue[
                'predicted_response'] if 'predicted_response' in dialogue else dialogue[
                    'ground_truth_response']

            assert isinstance(context, list), 'context should be a list'
            assert isinstance(query, str), 'query should be a string'
            assert isinstance(response, str), 'response should be a string'

            single_turn_score = model.get_score([query], response)
            single_turn_score = round(single_turn_score * 4 + 1, 2)
            single_turn_scores.append(single_turn_score)
            print('query:', query, file=f)
            print('response:', response, file=f)
            print('(single-turn) QuantiDCE score:', single_turn_score, file=f)
            print('-' * 10, file=f)

            history = context + [query]
            multi_turn_score = model.get_score(history, response)
            multi_turn_score = round(multi_turn_score * 4 + 1, 2)
            multi_turn_scores.append(multi_turn_score)
            print('context:', history, file=f)
            print('response:', response, file=f)
            print('(multi-turn) QuantiDCE score:', multi_turn_score, file=f)
            print('-' * 10, file=f)

        print('average single-turn score:',
              sum(single_turn_scores) / len(single_turn_scores),
              file=f)
        print('average multi-turn score:',
              sum(multi_turn_scores) / len(multi_turn_scores),
              file=f)
