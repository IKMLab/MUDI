from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput, Seq2SeqLMOutput


@dataclass
class DGatForPreTrainingOutput(ModelOutput):
    path_logits: Optional[torch.FloatTensor] = None
    turn_logits: Optional[torch.FloatTensor] = None
    adj_recon_logits: Optional[torch.FloatTensor] = None

    path_labels: Optional[torch.LongTensor] = None
    turn_labels: Optional[torch.LongTensor] = None
    adj_recon_labels: Optional[torch.LongTensor] = None


@dataclass
class DGatForCoherenceAwareDialogueEncodingOutput(ModelOutput):
    coh_rel_logits: Optional[torch.FloatTensor] = None
    coh_rel_preds: Optional[torch.FloatTensor] = None
    coh_rel_labels: Optional[torch.LongTensor] = None

    link_prediction_logits: Optional[torch.FloatTensor] = None
    link_prediction_labels: Optional[torch.LongTensor] = None

    next_resp_type_direct_logits: Optional[torch.FloatTensor] = None
    next_resp_type_direct_preds: Optional[torch.FloatTensor] = None
    next_resp_type_direct_labels: Optional[torch.LongTensor] = None
    next_resp_type_seq_logits: Optional[torch.FloatTensor] = None
    next_resp_type_seq_preds: Optional[torch.FloatTensor] = None
    next_resp_type_seq_labels: Optional[torch.LongTensor] = None

    last_resp_type_direct_preds: Optional[torch.LongTensor] = None
    last_resp_type_seq_preds: Optional[torch.LongTensor] = None
    last_resp_type_labels: Optional[torch.LongTensor] = None

    context_node_embeddings: Optional[torch.FloatTensor] = None
    context_graph_embeddings: Optional[torch.FloatTensor] = None

    persona_node_embeddings: Optional[torch.FloatTensor] = None
    persona_graph_embeddings: Optional[torch.FloatTensor] = None

    personalized_node_embeddings: Optional[torch.FloatTensor] = None
    personalized_graph_embeddings: Optional[torch.FloatTensor] = None


@dataclass
class PersonalizedDialogueGeneratorOutput(Seq2SeqLMOutput):
    dial_encoder_outputs: Optional[
        DGatForCoherenceAwareDialogueEncodingOutput] = None
    cls_loss: Optional[torch.FloatTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None
    cls_labels: Optional[torch.LongTensor] = None


@dataclass
class BaseTrainerOutput:
    total_loss: float


@dataclass
class DGatForPreTrainingTrainerOutput(BaseTrainerOutput):
    path_loss: float
    turn_loss: float
    adj_recon_loss: float


@dataclass
class DGatForCoherenceAwareDialogueEncodingTrainerOutput(BaseTrainerOutput):
    coh_rel_cls_loss: float
    link_prediction_loss: float
    next_resp_type_direct_loss: float
    next_resp_type_seq_loss: float

    coh_rel_accuracy: float = 0.0
    coh_rel_f1: float = 0.0
    coh_rel_top_at_k: float = 0.0
    coh_rel_mrr: float = 0.0

    link_pred_roc_auc: float = 0.0

    next_resp_type_direct_f1: float = 0.0
    next_resp_type_direct_top_at_k: float = 0.0
    next_resp_type_seq_f1: float = 0.0
    next_resp_type_seq_top_at_k: float = 0.0


@dataclass
class PersonalizedDialogueGeneratorTrainerOutput(BaseTrainerOutput):
    nll_loss: float
    cls_loss: float = 0.0
    multiple_choice_accuracy: float = 0.0
    multiple_choice_f1: float = 0.0


@dataclass
class CoherenceAwarePersonalizedDialogueGenerationSeq2SeqTrainerOutput(
        PersonalizedDialogueGeneratorTrainerOutput):
    dial_encoder_coh_rel_cls_loss: float = 0.0
    dial_encoder_link_prediction_loss: float = 0.0
    dial_encoder_next_resp_type_direct_loss: float = 0.0
    dial_encoder_next_resp_type_seq_loss: float = 0.0

    dial_encoder_coh_rel_accuracy: float = 0.0
    dial_encoder_coh_rel_f1: float = 0.0
    dial_encoder_coh_rel_top_at_k: float = 0.0
    dial_encoder_link_pred_roc_auc: float = 0.0
    dial_encoder_next_resp_type_direct_f1: float = 0.0
    dial_encoder_next_resp_type_direct_top_at_k: float = 0.0
    dial_encoder_next_resp_type_seq_f1: float = 0.0
    dial_encoder_next_resp_type_seq_top_at_k: float = 0.0
