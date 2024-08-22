from collections import Counter

import numpy as np
from nltk import ngrams
from nltk.probability import FreqDist


def distinct_n_sentence_level(sentence: str, n: int) -> float:
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences: list[str], n: int) -> float:
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(
        distinct_n_sentence_level(sentence, n)
        for sentence in sentences) / len(sentences)


def compute_entropy_and_distinct(corpus: list[str],
                                 n_grams: list[int] = [1, 2,
                                                       3]) -> tuple[dict, dict]:
    """
    coprus:
     :list[str]
    return:
        none
    prints:
        d-1,2,3; ENTR 1,2,3
    """

    entr_scores = {}
    distinct_scores = {}
    for n_gram in n_grams:
        all_grams = []
        for sent in corpus:
            all_grams += list(ngrams(sent.split(), n_gram))
        fdist = FreqDist(Counter(all_grams))
        entr = 0
        for x in fdist.keys():
            p_x = fdist.freq(x)
            entr += p_x * np.log2(p_x)
        entr_scores[n_gram] = -entr
        distinct_scores[n_gram] = distinct_n_corpus_level(corpus, n_gram)
    return entr_scores, distinct_scores


def compute_unique_sentence_ratio(sentences: list[str]) -> tuple[float, int]:
    unique_seq = set(sentences)

    return len(unique_seq) / len(sentences), len(unique_seq)
