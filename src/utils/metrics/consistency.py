import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
)


class ConsistencyScorer:

    def __init__(self,
                 device: str = 'cuda',
                 checkpoint: str = 'zayn1111/deberta-v3-dnli'):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(
            checkpoint, use_fast=False, model_max_length=512)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            checkpoint).to(device)

        self.device = device

    def compute(self, premise: str, hypothesis: str) -> dict[str, float]:
        input_encodings = self.tokenizer(premise,
                                         hypothesis,
                                         truncation=True,
                                         return_tensors='pt')

        output = self.model(input_encodings['input_ids'].to(self.device))
        prediction = torch.softmax(output['logits'][0], -1).tolist()

        label_names = ['entailment', 'neutral', 'contradiction']
        prediction = {name: pred for pred, name in zip(prediction, label_names)}
        return prediction


class ConsistencyScorerV2:

    def __init__(self,
                 device: str = 'cuda',
                 checkpoint: str = 'consistency_model/'):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint).to(device)

        self.device = device

    def compute(self, premise: str, hypothesis: str) -> dict[str, float]:
        input_encodings = self.tokenizer(premise,
                                         hypothesis,
                                         truncation=True,
                                         return_tensors='pt')

        output = self.model(input_encodings['input_ids'].to(self.device))
        prediction = torch.nn.functional.softmax(output[0].detach(),
                                                 dim=1).tolist()[0]
        prediction.reverse()

        label_names = ['entailment', 'neutral', 'contradiction']
        prediction = {name: pred for pred, name in zip(prediction, label_names)}
        return prediction
