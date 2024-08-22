from enum import Enum

COHERENCE_RELATIONS = [
    'Comment', 'Clarification-Question', 'Elaboration', 'Acknowledgment',
    'Continuation', 'Explanation', 'Conditional', 'QA', 'Alternation',
    'Question-Elaboration', 'Result', 'Background', 'Narration', 'Correction',
    'Parallel', 'Contrast', 'Topic Shift'
]

COHREL2ID = {label: idx for idx, label in enumerate(COHERENCE_RELATIONS)}
ID2COHREL = {idx: label for idx, label in enumerate(COHERENCE_RELATIONS)}

NEW_SPEICAL_TOKENS_MAP = {
    'persona': '<persona>',
    'query': '<query>',
    'response': '<response>',
    **{
        label.lower(): f'<{label.lower()}>'
        for label in COHERENCE_RELATIONS
    }
}
NEW_SPEICAL_TOKENS = list(NEW_SPEICAL_TOKENS_MAP.values())


class ModelTrainMode(Enum):
    r"""Model training mode.
    PRETRAINING: Pretrain the encoder.
    FINETUNEING: Finetune the model with the pretrained encoder.
    """

    PRETRAINING = 'pretraining'
    FINETUNEING = 'finetuning'


SAMPLE_PER_CLASS = {
    'mixtral_3-hop_filter_topicshift': [
        340489, 132192, 273274, 130994, 31382, 58218, 2226, 34016, 3021, 32900,
        3006, 38368, 17330, 425, 3967, 71722, 402232
    ],
    'mixtral_3-hop_filter_comment_topicshift': [
        335561, 132192, 273274, 130994, 31382, 58218, 2226, 34016, 3021, 32900,
        3006, 38368, 17330, 425, 3967, 71722, 402232
    ],
    'llama3_3-hop_filter_topicshift': [
        85280, 57000, 244365, 181665, 13325, 30840, 3399, 190385, 5602, 1412,
        369, 5969, 101, 2456, 9847, 64737, 116203
    ]
}
