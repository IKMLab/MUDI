"""callback to calculate perplexity as an evaluation metric."""
import logging
import math
import statistics
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer


class Perplexity:
    """
    Calculate perplexity as defined in https://huggingface.co/docs/transformers/en/perplexity.
    This is a custom variant that doesn't re-tokenize the input or re-load the model.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        stride: int = 512,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.name = "perplexity"

    def _feature_names(self) -> list[str]:
        return ["references"]

    def compute(
        self,
        references: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        Compute perplexity in a fixed length sliding window across the sequence.
        """
        assert references is not None, "Missing parameter: references"

        references_tokenized = self.tokenizer(references,
                                              return_tensors="pt",
                                              padding=True,
                                              truncation=True)
        input_ids: Tensor = references_tokenized["input_ids"]  # type: ignore
        input_ids = input_ids.to(self.device)

        sequence_length = input_ids.size(1)

        losses = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, sequence_length, self.stride)):
            end_loc = min(begin_loc + self.max_seq_len, sequence_length)
            trg_len = end_loc - prev_end_loc
            input_ids_slice = input_ids[:, begin_loc:end_loc]
            labels_slice = input_ids_slice.clone()
            labels_slice[:, :-trg_len] = -100

            with torch.no_grad():
                outputs: CausalLMOutput = self.model(input_ids=input_ids_slice,
                                                     labels=labels_slice)

            losses.append(outputs.loss)

            prev_end_loc = end_loc
            if end_loc == sequence_length:
                break

        perplexity = torch.exp(torch.stack(losses).mean()).item()

        return {
            "score": perplexity,
        }


class BartTokenLevelPerplexityScorer:

    def __init__(self, model, num_test_chars=None, device='cuda'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            'facebook/bart-large')  # bart-large is the same
        self.text = self.load_text(num_test_chars)

    # Load the text. num_test_chars: 10K chars = 2246 tokens (use None for all)
    def load_text(self, num_test_chars):
        logging.getLogger('datasets').setLevel(
            logging.ERROR)  # Reusing dataset wikitext,...
        articles = [
            a['text']
            for a in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        ]
        text = ' '.join(articles)[:num_test_chars]
        return text

    def compute(self,
                 seq_len=None,
                 num_test_chars=None,
                 batch_size=8,
                 mlm_prob=0.15):
        # Tokenize.  verbose=False elminates message 'token sequences too long for model'
        tok_ids = self.tokenizer(self.text,
                                 add_special_tokens=False,
                                 verbose=False).input_ids

        # Split into tokenized sequences all of the same length and discard any short samples at the end
        if seq_len is None:
            seq_len = self.tokenizer.model_max_length
        samples = [c for c in chunk(tok_ids, seq_len) if len(c) == seq_len]
        print(f'Loaded {len(samples):,} samples of length {len(samples[0]):,} tokens')

        # Add bos and eos tokens and create the decoder_input_ids
        # mask_token_id = 50264
        bos = self.tokenizer.bos_token_id  # = 0
        eos = self.tokenizer.eos_token_id  # = 2
        dst = self.model.config.decoder_start_token_id  # = 2 (same as eos token id)
        input_ids = [[bos] + sample + [eos] for sample in samples]
        decoder_ids = [[dst] + iids[:-1]
                       for iids in input_ids]  # shift_tokens_right

        # Put this all into a dataset and create the loader
        # The collator will take care of randomly masking the input_id tokens and creating the
        # 'labels' keys with -100 for any non-masked token
        dataset = EvalDataset(input_ids, decoder_ids)
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                   mlm_probability=mlm_prob)
        dataloader = DataLoader(dataset,
                                collate_fn=collator,
                                batch_size=batch_size)

        # Run evaluation
        print('Testing')
        self.model.eval()
        losses = []
        for step, batch in enumerate(tqdm(dataloader, ncols=100,
                                          disable=False)):
            with torch.no_grad():
                torch.set_printoptions(threshold=10000, linewidth=150)
                decoder_ids = batch['decoder_input_ids'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                     labels=labels,
                                     decoder_input_ids=decoder_ids)
            losses.append(outputs.loss.item())
        try:
            perplexity = math.exp(statistics.mean(losses))
        except OverflowError:
            perplexity = float('inf')
        return perplexity


# iterator to split a list into n segments
def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Container for model data
class EvalDataset(Dataset):

    def __init__(self, input_ids, decoder_input_ids):
        assert len(input_ids) == len(decoder_input_ids)
        self.input_ids = input_ids
        self.decoder_input_ids = decoder_input_ids

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'decoder_input_ids': self.decoder_input_ids[index]
        }

    def __len__(self):
        return len(self.input_ids)
