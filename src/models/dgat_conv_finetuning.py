from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import negative_sampling

from src.models.dgat_conv_pretraining import DGatConvPretraining
from src.modules.classifiers import (
    CoherenceRelationsClassifier,
    NextResponseTypePredictor,
)
from src.modules.encoder import (
    PersonaGraphEncoder,
    UtteranceEncoder,
)
from src.modules.layers import DialogueGATConv
from src.utils.constants import SAMPLE_PER_CLASS
from src.utils.losses import compute_loss_with_class_balanced
from src.utils.model_outputs import (
    DGatForCoherenceAwareDialogueEncodingOutput,
    DGatForCoherenceAwareDialogueEncodingTrainerOutput,
)

GNN_LAYER_CLASSES = {
    'GAT': GATConv,
    'GATv2': GATv2Conv,
    'DialogueGAT': DialogueGATConv
}


def get_layer_class(layer_type: str):
    return GNN_LAYER_CLASSES[layer_type]


class DGatForCoherenceAwareDialogueEncoding(DGatConvPretraining):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 edge_dim: int,
                 encoder_layer_instance: MessagePassing,
                 num_encoder_layers: int,
                 utterance_encoder_class: Literal['none', 'bert', 'roberta'],
                 pretrained_weights_path: Optional[str] = None,
                 **kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, edge_dim,
                         encoder_layer_instance, num_encoder_layers, **kwargs)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.num_edge_labels = 17
        node_embedding_dim = output_dim
        print(f'Input dimension: {input_dim}')  # 768
        print(f'Hidden dimension: {hidden_dim}')  # 512
        print(f'Output dimension: {output_dim}')  # 512

        self.initialize_encoder(pretrained_weights_path)

        # 768 -> 512 (-> 512)
        self.persona_encoder = PersonaGraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            layer_class=GATv2Conv,
            num_layers=num_encoder_layers,
            **kwargs)

        if utterance_encoder_class != 'none':
            self.utterance_encoder = UtteranceEncoder(
                pretrained_model_name=utterance_encoder_class, freeze=True)

        # Node-level task
        # 512 + 512 -> 512
        self.personalized_node_aggregator = MultiHeadAttentionBasedAggregator(
            node_embedding_dim=node_embedding_dim,
            persona_embedding_dim=node_embedding_dim,
            num_heads=4,
            concat=True)
        self.cosine_similarity = CosineSimilarity(temperature=0.5)

        # Edge-level task
        # 512 + 512 -> 17 (multi-label classification)
        self.coherence_relations_classifier = CoherenceRelationsClassifier(
            hidden_dim=node_embedding_dim, num_edge_labels=self.num_edge_labels)

        self.next_response_type_predictor = NextResponseTypePredictor(
            node_embedding_dim=node_embedding_dim,
            hidden_dim=node_embedding_dim,
            num_edge_labels=self.num_edge_labels,
            contains_edge_attr=False)

    def forward(self,
                order_ids: torch.LongTensor,
                turn_ids: torch.LongTensor,
                role_ids: torch.LongTensor,
                x: torch.FloatTensor,
                y: torch.FloatTensor,
                edge_index: torch.LongTensor,
                edge_attr: torch.FloatTensor,
                utterance_input_ids: Optional[torch.LongTensor] = None,
                utterance_attention_mask: Optional[torch.LongTensor] = None,
                persona_input_ids: Optional[torch.LongTensor] = None,
                persona_attention_mask: Optional[torch.LongTensor] = None,
                y_res_type: Optional[torch.FloatTensor] = None,
                res_type_edge_index: Optional[torch.LongTensor] = None,
                x_user_personas: Optional[torch.FloatTensor] = None,
                x_partner_personas: Optional[torch.FloatTensor] = None,
                user_personas_edge_index: Optional[torch.LongTensor] = None,
                partner_personas_edge_index: Optional[torch.LongTensor] = None,
                batch: Optional[torch.LongTensor] = None,
                user_personas_batch: Optional[torch.LongTensor] = None,
                partner_personas_batch: Optional[torch.LongTensor] = None,
                num_user_personas: Optional[torch.LongTensor] = None,
                num_partner_personas: Optional[torch.LongTensor] = None,
                use_resp_type_prediction_in_training: Optional[bool] = False,
                **kwargs) -> DGatForCoherenceAwareDialogueEncodingOutput:
        if getattr(self, 'utterance_encoder', None):
            x = self.utterance_encoder(utterance_input_ids,
                                       utterance_attention_mask).squeeze()
            x_user_personas = self.utterance_encoder(
                persona_input_ids, persona_attention_mask).squeeze()

        # 1.1: Encode the persona
        user_persona_node_embeddings, user_persona_graph_embeddings, user_personas_batch = self.encode_persona_embeddings(
            x_personas=x_user_personas,
            personas_edge_index=user_personas_edge_index,
            num_personas=num_user_personas,
            personas_batch=user_personas_batch)
        partner_persona_node_embeddings, partner_persona_graph_embeddings, partner_personas_batch = self.encode_persona_embeddings(
            x_personas=x_partner_personas,
            personas_edge_index=partner_personas_edge_index,
            num_personas=num_partner_personas or num_user_personas,
            ref_tensor=(user_persona_node_embeddings,
                        user_persona_graph_embeddings))

        persona_embeddings = self.stack_persona_embeddings(
            user_persona_node_embeddings=user_persona_node_embeddings,
            partner_persona_node_embeddings=partner_persona_node_embeddings,
            user_personas_batch=user_personas_batch,
            partner_personas_batch=partner_personas_batch,
            global_batch=batch,
            speaker_ids=role_ids)

        # 1.2: Encode the coherenec-aware dialogue graph
        if self.training:
            if use_resp_type_prediction_in_training:
                dialogue_embeddings, encoded_edge_attr = self.encoder(
                    x, edge_index, order_ids, turn_ids,
                    torch.rand_like(edge_attr,
                                    dtype=torch.float,
                                    device=x.device))
            else:
                dialogue_embeddings, encoded_edge_attr = self.encoder(
                    x, edge_index, order_ids, turn_ids, edge_attr)
        else:
            dialogue_embeddings, encoded_edge_attr = self.encoder(
                x, edge_index, order_ids, turn_ids,
                torch.rand_like(edge_attr, dtype=torch.float, device=x.device))

        # 2: Aggregate the context node embeddings with the persona embeddings
        personalized_node_embeddings = self.personalized_node_aggregator(
            dialogue_embeddings, persona_embeddings)

        # 3: Downstream tasks
        # 3.1: Edge-level task: Next Response Type Prediction
        batch_x_resp_type = [
            dialogue_embeddings[batch == i] for i in batch.unique(sorted=True)
        ]
        batch_y_resp_type = []
        g_start_idx = 0
        for g in batch_x_resp_type:
            batch_y_resp_type.append(
                self.encoder.edge_emb(y_res_type[g_start_idx:g_start_idx +
                                                 len(g)].float()))
            g_start_idx += len(g)

        next_resp_type_logits = self.next_response_type_predictor(
            batch_node_embeddings=batch_x_resp_type,
            batch_edge_attr=batch_y_resp_type,
            device=x.device)

        accumulation_index = 0
        last_resp_type_direct_preds = []
        last_resp_type_seq_preds = []
        last_resp_type_labels = []
        for g_size in [len(i) for i in batch_y_resp_type]:
            accumulation_index += g_size

            last_resp_type_direct_preds.append(
                torch.sigmoid(next_resp_type_logits[0][accumulation_index -
                                                       1]).tolist())
            last_resp_type_seq_preds.append(
                torch.sigmoid(next_resp_type_logits[1][accumulation_index -
                                                       1]).tolist())
            last_resp_type_labels.append(y_res_type[accumulation_index -
                                                    1].tolist())

        # 3.2: Edge-level task: Coherence Relations Classification (multi-label)
        if not self.training:
            # In validation and test, use the predicted response type instead of the ground truth
            next_resp_type_pred_label_probs = torch.sigmoid(
                next_resp_type_logits[1])
            top_k_next_resp_pred_labels = torch.zeros_like(
                next_resp_type_pred_label_probs)

            for i, probs in enumerate(next_resp_type_pred_label_probs):
                top_k_indices = torch.topk(probs, k=2).indices
                top_k_next_resp_pred_labels[i, top_k_indices] = 1

            encoded_top_k_pred_labels = self.encoder.edge_emb(
                top_k_next_resp_pred_labels)
            max_node_id = torch.max(edge_index.max(),
                                    res_type_edge_index.max()).item() + 1
            edge_hash = edge_index[0] * max_node_id + edge_index[1]
            resp_type_hash = res_type_edge_index[
                0] * max_node_id + res_type_edge_index[1]
            _, edge_pos, resp_type_pos = np.intersect1d(
                edge_hash.cpu().numpy(),
                resp_type_hash.cpu().numpy(),
                return_indices=True)
            encoded_edge_attr[edge_pos] = encoded_top_k_pred_labels[
                resp_type_pos]

        coh_rel_logits = self.coherence_relations_classifier(
            dialogue_embeddings, edge_index, encoded_edge_attr)

        # 3.3: Edge-level task: Link Prediction
        neg_edge_index = negative_sampling(
            edge_index, num_nodes=dialogue_embeddings.size(0))
        augmented_edge_index = torch.cat([edge_index, neg_edge_index], dim=1)

        pos_link_pred_labels = torch.ones(edge_index.size(1),
                                          device=edge_index.device)
        neg_link_pred_labels = torch.zeros(neg_edge_index.size(1),
                                           device=neg_edge_index.device)
        link_pred_labels = torch.cat(
            [pos_link_pred_labels, neg_link_pred_labels], dim=0)

        link_pred_logits = (dialogue_embeddings[augmented_edge_index[0]] *
                            dialogue_embeddings[augmented_edge_index[1]]).sum(
                                dim=-1)

        return DGatForCoherenceAwareDialogueEncodingOutput(
            coh_rel_logits=coh_rel_logits,
            coh_rel_preds=torch.sigmoid(coh_rel_logits),
            coh_rel_labels=y,
            link_prediction_logits=link_pred_logits,
            link_prediction_labels=link_pred_labels,
            next_resp_type_direct_logits=next_resp_type_logits[0],
            next_resp_type_direct_preds=torch.sigmoid(next_resp_type_logits[0]),
            next_resp_type_direct_labels=y_res_type.float(),
            next_resp_type_seq_logits=next_resp_type_logits[1],
            next_resp_type_seq_preds=torch.sigmoid(next_resp_type_logits[1]),
            next_resp_type_seq_labels=y_res_type.float(),
            last_resp_type_direct_preds=torch.tensor(
                last_resp_type_direct_preds, device=x.device),
            last_resp_type_seq_preds=torch.tensor(last_resp_type_seq_preds,
                                                  device=x.device),
            last_resp_type_labels=torch.tensor(last_resp_type_labels,
                                               device=x.device),
            context_node_embeddings=dialogue_embeddings,
            context_graph_embeddings=global_mean_pool(dialogue_embeddings,
                                                      batch),
            persona_node_embeddings=persona_embeddings,
            persona_graph_embeddings=user_persona_graph_embeddings,
            personalized_node_embeddings=personalized_node_embeddings,
            personalized_graph_embeddings=global_mean_pool(
                personalized_node_embeddings, batch))

    def stack_persona_embeddings(self,
                                 user_persona_node_embeddings: torch.Tensor,
                                 partner_persona_node_embeddings: torch.Tensor,
                                 user_personas_batch: torch.Tensor,
                                 partner_personas_batch: torch.Tensor,
                                 global_batch: list[int],
                                 speaker_ids: list[int]) -> list[torch.Tensor]:
        persona_embeddings = []
        for batch_id, speaker_id in zip(global_batch, speaker_ids):
            speaker_embeddings = partner_persona_node_embeddings[
                partner_personas_batch ==
                batch_id] if speaker_id == 0 else user_persona_node_embeddings[
                    user_personas_batch == batch_id]
            persona_embeddings.append(speaker_embeddings)

        return persona_embeddings

    def encode_persona_embeddings(
        self,
        x_personas: Union[torch.Tensor, None],
        personas_edge_index: Union[torch.Tensor, None],
        num_personas: int,
        personas_batch: Optional[torch.Tensor] = None,
        ref_tensor: Optional[tuple[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x_personas.device if x_personas is not None else ref_tensor[
            0].device
        if personas_batch is None:
            batch = [
                j for i in range(len(num_personas))
                for j in [i] * num_personas[i]
            ]
            personas_batch = torch.tensor(batch, device=device)

        if x_personas is not None and personas_edge_index is not None:
            persona_node_embeddings, persona_graph_embeddings = self.persona_encoder(
                x_personas, personas_edge_index, personas_batch)
        else:
            # if no persona features, use zero vector for embeddings
            if ref_tensor is not None:
                persona_node_embeddings = torch.zeros_like(ref_tensor[0],
                                                           device=device)
                persona_graph_embeddings = torch.zeros_like(ref_tensor[1],
                                                            device=device)
            else:
                persona_node_embeddings = torch.zeros(len(personas_batch),
                                                      self.output_dim,
                                                      device=device)
                persona_graph_embeddings = torch.zeros(len(num_personas),
                                                       self.output_dim,
                                                       device=device)

        return persona_node_embeddings, persona_graph_embeddings, personas_batch

    def compute_loss(
            self, output: DGatForCoherenceAwareDialogueEncodingOutput,
            weight: dict[dict, float],
            **kwargs) -> DGatForCoherenceAwareDialogueEncodingTrainerOutput:
        coh_rel_cls_loss = compute_loss_with_class_balanced(
            labels=output.coh_rel_labels,
            logits=output.coh_rel_logits,
            samples_per_cls=SAMPLE_PER_CLASS['llama3_3-hop_filter_topicshift'],
            no_of_classes=self.num_edge_labels,
            loss_type='focal',
            beta=0.99999,
            gamma=2.0,
            is_onehot=True)

        link_prediction_loss = F.binary_cross_entropy_with_logits(
            output.link_prediction_logits, output.link_prediction_labels)

        next_resp_type_direct_loss = F.binary_cross_entropy_with_logits(
            output.next_resp_type_direct_logits,
            output.next_resp_type_direct_labels)

        next_resp_type_seq_loss = F.binary_cross_entropy_with_logits(
            output.next_resp_type_seq_logits, output.next_resp_type_seq_labels)

        total_loss = weight['coh_rel_cls'] * coh_rel_cls_loss + \
            weight['link_prediction'] * link_prediction_loss + \
            weight['next_resp_type_direct'] * next_resp_type_direct_loss + \
            weight['next_resp_type_seq'] * next_resp_type_seq_loss

        return DGatForCoherenceAwareDialogueEncodingTrainerOutput(
            total_loss=total_loss,
            coh_rel_cls_loss=coh_rel_cls_loss,
            link_prediction_loss=link_prediction_loss,
            next_resp_type_direct_loss=next_resp_type_direct_loss,
            next_resp_type_seq_loss=next_resp_type_seq_loss,
        )


class MultiHeadAttentionBasedAggregator(nn.Module):

    def __init__(self,
                 node_embedding_dim: int,
                 persona_embedding_dim: int,
                 num_heads: int,
                 concat: Optional[bool] = True):
        super().__init__()

        self.num_heads = num_heads
        self.dim_per_head = node_embedding_dim // num_heads
        self.concat = concat

        self.query = nn.Linear(node_embedding_dim,
                               self.dim_per_head * num_heads)
        self.key = nn.Linear(persona_embedding_dim,
                             self.dim_per_head * num_heads)
        self.value = nn.Linear(persona_embedding_dim,
                               self.dim_per_head * num_heads)

        if concat:
            self.output = nn.Linear(
                node_embedding_dim + num_heads * self.dim_per_head,
                node_embedding_dim)
        else:
            self.output = nn.Linear(node_embedding_dim, node_embedding_dim)

    def forward(self, node_embeddings: torch.Tensor,
                persona_embeddings_list: list[torch.Tensor]) -> torch.Tensor:
        personalized_embeddings = []
        for node_emb, persona_embs in zip(node_embeddings,
                                          persona_embeddings_list):
            queries = self.query(node_emb).view(1, self.num_heads,
                                                self.dim_per_head)
            keys = self.key(persona_embs).view(-1, self.num_heads,
                                               self.dim_per_head)
            values = self.value(persona_embs).view(-1, self.num_heads,
                                                   self.dim_per_head)

            attention_scores = torch.einsum('bhd,khd->bhk', [queries, keys])
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_output = torch.einsum(
                'bhk,khd->bhd', [attention_weights, values]).squeeze(0)

            attention_output = attention_output.view(-1)

            if self.concat:
                combined_emb = torch.cat([node_emb, attention_output], dim=0)
            else:
                combined_emb = 0.5 * (node_emb + attention_output)

            projected_emb = self.output(combined_emb)
            personalized_embeddings.append(projected_emb)

        personalized_node_embeddings = torch.stack(personalized_embeddings)
        return personalized_node_embeddings


class CosineSimilarity(nn.Module):

    def __init__(self, temperature: float):
        super().__init__()

        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, h_1: torch.Tensor, h_2: torch.Tensor) -> torch.Tensor:
        return self.cos_sim(h_1, h_2) / self.temperature
