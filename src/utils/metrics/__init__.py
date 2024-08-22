from typing import Union

from transformers.modeling_outputs import ModelOutput

from .bart_score import BARTScorer
from .bleu import compute_bleu
from .classification import compute_accuracy, compute_f1
from .consistency import ConsistencyScorer, ConsistencyScorerV2
from .diversity import (
    compute_entropy_and_distinct,
    compute_unique_sentence_ratio,
)
from .perplexity import Perplexity
from .ranking import compute_hits_at_k, compute_mrr, compute_roc_auc
from .rouge import rouge, rouge_l_sentence_level, rouge_l_summary_level, rouge_n


class MetricsAccumulator:

    def __init__(self, total_steps: int, output_class: ModelOutput = None):
        self.metrics = {}
        self.count = 0
        self.total_steps = total_steps
        self.output_class = output_class

    def update(self, **metrics):
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key] += value.item() if hasattr(value,
                                                             'item') else value
            else:
                self.metrics[key] = value.item() if hasattr(value,
                                                            'item') else value
        self.count += 1

    def averages(self) -> Union[dict[str, float], ModelOutput]:
        assert self.count == self.total_steps, 'Not all losses are updated'

        metrics_averages = {
            key: value / self.count
            for key, value in self.metrics.items()
        }

        if self.output_class:
            return self.output_class(**metrics_averages)

        return metrics_averages

    def reset(self):
        self.metrics = {}
        self.count = 0
