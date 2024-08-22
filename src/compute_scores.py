import argparse

import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.metrics import (
    BARTScorer,
    ConsistencyScorer,
    ConsistencyScorerV2,
    compute_entropy_and_distinct,
    compute_unique_sentence_ratio,
)

BATCH_SIZE = 64
ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
DIVERSITY_NGRAMS = [1, 2, 3]
NLI_SCORES = {0: 1, 1: 0, 2: -1}


def main(args: argparse.Namespace):
    data = pd.read_json(args.input_file_path)

    reference_sentences = data['ground_truth_response'].tolist()
    prediction_sentences = data['predicted_response'].tolist()

    print('Total number of samples:', len(reference_sentences))

    # =============================================
    #  Diversity: USR, USN, Entropy, Distinct
    # ==============================================
    print('Computing diversity metrics: USR, USN, Entropy, Distinct...')

    unique_sentence_ratio, unique_sentence_num = compute_unique_sentence_ratio(
        prediction_sentences)

    entr_scores, distinct_scores = compute_entropy_and_distinct(
        prediction_sentences, n_grams=DIVERSITY_NGRAMS)
    entr_mean = np.mean(list(entr_scores.values()))

    # =====================================
    #  Text-quality: BLEU, ROUGE
    # =====================================
    print('Computing text-quality metrics: BLEU, ROUGE...')

    bleu_scorer = evaluate.load('bleu')
    google_bleu_scorer = evaluate.load('google_bleu')
    rouge_scorer = evaluate.load('rouge')

    bleu_1 = bleu_scorer.compute(references=reference_sentences,
                                 predictions=prediction_sentences,
                                 max_order=1,
                                 smooth=True)['bleu']
    bleu_2 = bleu_scorer.compute(references=reference_sentences,
                                 predictions=prediction_sentences,
                                 max_order=2,
                                 smooth=True)['bleu']
    bleu_3 = bleu_scorer.compute(references=reference_sentences,
                                 predictions=prediction_sentences,
                                 max_order=3,
                                 smooth=True)['bleu']
    bleu_4 = bleu_scorer.compute(references=reference_sentences,
                                 predictions=prediction_sentences,
                                 max_order=4,
                                 smooth=True)['bleu']

    google_bleu = google_bleu_scorer.compute(
        references=[[ref] for ref in reference_sentences],
        predictions=prediction_sentences,
    )['google_bleu']

    rouge = rouge_scorer.compute(references=reference_sentences,
                                 predictions=prediction_sentences,
                                 rouge_types=ROUGE_TYPES,
                                 use_stemmer=True)

    # =====================================
    #  Feature-coverage: BERTScore
    # =====================================
    print('Computing feature-coverage metrics: BERTScore...')

    bert_scorer = evaluate.load('bertscore')

    bert_score = bert_scorer.compute(references=reference_sentences,
                                     predictions=prediction_sentences,
                                     lang='en',
                                     batch_size=BATCH_SIZE)
    bert_score_p = np.mean(bert_score['precision'])
    bert_score_r = np.mean(bert_score['recall'])
    bert_score_f1 = np.mean(bert_score['f1'])

    # consist_scorer = ConsistencyScorer(device='cuda',
    #                                    checkpoint='zayn1111/deberta-v3-dnli')
    consist_scorer = ConsistencyScorerV2(device='cuda',
                                         checkpoint='consistent_model')

    keywords = pd.read_json('dataset/ConvAI2/valid_self_original_keyword.json')
    keywords['query'] = keywords['dialogues'].str[-1]

    filter_keywords = []
    if len(prediction_sentences) < len(keywords):
        for gt_resp in reference_sentences:
            f = keywords[keywords['query'].str == gt_resp.lower()]
            filter_keywords.append(f.iloc[0])

        keywords = pd.DataFrame(filter_keywords)

    qk_consist_sum = []
    pk_consist_sum = []
    for i, row in tqdm(keywords.iterrows(), total=len(keywords)):
        query_keywords = row['query_keyword']
        persona_keyword = row['persona_keyword']
        pred_response = prediction_sentences[i]

        # For each persona keyword, compute the consistency score
        for qk in query_keywords:
            for k in qk['labels']:
                qk_consist_score = consist_scorer.compute(pred_response, k)
                index = np.argmax(list(qk_consist_score.values()))
                qk_consist_sum.append(NLI_SCORES[index])
                if index == 0:
                    qk_consist_sum.append(1)
                else:
                    qk_consist_sum.append(0)

        # Only consider the query keyword
        for k in persona_keyword[-2]['labels']:
            pk_consist_score = consist_scorer.compute(pred_response, k)
            index = np.argmax(list(pk_consist_score.values()))
            pk_consist_sum.append(NLI_SCORES[index])
            if index == 0:
                pk_consist_sum.append(1)
            else:
                pk_consist_sum.append(0)

    qk_consist_score = np.mean(qk_consist_sum)
    pk_consist_score = np.mean(pk_consist_sum)

    # =====================================
    #  Factuality & Faithfulness: BARTScore
    # =====================================
    print('Computing factuality & faithfulness metrics: BARTScore...')

    bart_scorer = BARTScorer(device='cuda',
                             checkpoint='facebook/bart-large-cnn')
    # Load the pre-trained model from original BARTScore Paper
    bart_scorer.load('bart_score.pth')

    context = data['context'].tolist()
    query = data['query'].tolist()

    dialogue = [' '.join(c + [q]) for c, q in zip(context, query)]

    bart_score_sum = 0
    for i in tqdm(range(0, len(dialogue), BATCH_SIZE)):
        bart_score = bart_scorer.score(
            srcs=dialogue[i:i + BATCH_SIZE],
            tgts=prediction_sentences[i:i + BATCH_SIZE],
            batch_size=BATCH_SIZE,
        )
        bart_score_sum += np.sum(bart_score)

    bart_score_dialogue = bart_score_sum / len(dialogue)

    bart_score_sum = 0
    for i in tqdm(range(0, len(query), BATCH_SIZE)):
        bart_score = bart_scorer.score(
            srcs=query[i:i + BATCH_SIZE],
            tgts=prediction_sentences[i:i + BATCH_SIZE],
            batch_size=BATCH_SIZE,
        )
        bart_score_sum += np.sum(bart_score)

    bart_score_query = bart_score_sum / len(query)

    # =====================================
    #  Consistency: Persona-Consistency (C.Score)
    # =====================================
    print('Computing consistency metrics: Persona-Consistency (C.Score)...')

    if consist_scorer is None:
        # consist_scorer = ConsistencyScorer(
        #     device='cuda', checkpoint='zayn1111/deberta-v3-dnli')
        consist_scorer = ConsistencyScorerV2(
            device='cuda', checkpoint='consistent_model')

    persona_sentences = data['persona'].tolist()
    persona_sentences_combined = [' '.join(p) for p in persona_sentences]
    assert len(persona_sentences) == len(prediction_sentences)

    consist_score_sum = []
    count = {0: 0, 1: 0, 2: 0}
    for i in tqdm(range(len(persona_sentences)), total=len(persona_sentences)):
        for p in persona_sentences[i]:
            consist_score = consist_scorer.compute(prediction_sentences[i], p)

            index = np.argwhere(
                np.array(list(consist_score.values())) >= 0.5).tolist()
            if len(index) == 0:
                continue

            index = index[0][0]
            consist_score_sum.append(NLI_SCORES[index])
            count[index] += 1

    consist_score = np.sum(consist_score_sum)

    consist_score_sum = []
    for i in tqdm(range(0, len(persona_sentences_combined), BATCH_SIZE)):
        srcs = persona_sentences_combined[i:i + BATCH_SIZE]
        tgts = prediction_sentences[i:i + BATCH_SIZE]

        hypo_ref = np.array(bart_scorer.score(srcs, tgts,
                                              batch_size=BATCH_SIZE))
        ref_hypo = np.array(bart_scorer.score(tgts, srcs,
                                              batch_size=BATCH_SIZE))
        avg_f = 0.5 * (ref_hypo + hypo_ref)
        consist_score_sum += avg_f.tolist()

    consist_score_bart = np.mean(consist_score_sum)

    # =====================================
    # Results
    # =====================================
    print(f'USR: {unique_sentence_ratio}, USN: {unique_sentence_num}')
    print(f'ENTR (average): {entr_mean}')
    for gram_size in DIVERSITY_NGRAMS:
        print(f'Ent-{gram_size}: {entr_scores[gram_size]}')
    for gram_size in DIVERSITY_NGRAMS:
        print(f'Dist-{gram_size}: {distinct_scores[gram_size]}')

    print(f'BLEU-1: {bleu_1}')
    print(f'BLEU-2: {bleu_2}')
    print(f'BLEU-3: {bleu_3}')
    print(f'BLEU-4: {bleu_4}')
    print(f'Google-BLEU: {google_bleu}')
    for rouge_type in ROUGE_TYPES:
        print(f'{rouge_type}: {rouge[rouge_type]}')

    print(f'BERT score (precision): {bert_score_p}')
    print(f'BERT score (recall): {bert_score_r}')
    print(f'BERT score (f1): {bert_score_f1}')

    print(f'BART score (context+query): {bart_score_dialogue}')
    print(f'BART score (query): {bart_score_query}')

    print(f'Persona-Consistency score (NLI): {consist_score}')
    print(f'Persona-Consistency score (BART): {consist_score_bart}')

    print(f'Persona-Consistency score (Keyword): {pk_consist_score}')
    print(f'Query-Consistency score (Keyword): {qk_consist_score}')

    result_df = pd.DataFrame({
        'USR': unique_sentence_ratio,
        'USN': unique_sentence_num,
        'ENTR': entr_mean,
        **{
            f'Ent-{gram_size}': entr_scores[gram_size]
            for gram_size in DIVERSITY_NGRAMS
        },
        **{
            f'Dist-{gram_size}': distinct_scores[gram_size]
            for gram_size in DIVERSITY_NGRAMS
        },
        'BLEU-1': [bleu_1],
        'BLEU-2': [bleu_2],
        'BLEU-3': [bleu_3],
        'BLEU-4': [bleu_4],
        'Google-BLEU': [google_bleu],
        **{
            rouge_type.upper(): rouge[rouge_type]
            for rouge_type in ROUGE_TYPES
        },
        'BERTScore-P': bert_score_p,
        'BERTScore-R': bert_score_r,
        'BERTScore-F1': bert_score_f1,
        'BARTScore-Dialogue': bart_score_dialogue,
        'BARTScore-Query': bart_score_query,
        'Persona-Consistency': consist_score,
        'Persona-Consistency-BART': consist_score_bart,
        'Persona-Consistency-Keyword': pk_consist_score,
        'Query-Consistency-Keyword': qk_consist_score,
    })

    result_df.to_csv(args.output_file_path)
    print(f'Saved the results to {args.output_file_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the performance of the model')
    parser.add_argument('-i',
                        '--input_file_path',
                        type=str,
                        help='Path to the data for evaluation.',
                        required=True)
    parser.add_argument('-o',
                        '--output_file_path',
                        type=str,
                        help='Path where evaluation outputs are saved.',
                        required=True)
    args = parser.parse_args()

    main(args)
