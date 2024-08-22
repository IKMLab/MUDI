from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from transformers.modeling_outputs import ModelOutput

from src.modules.classifiers import (
    InnerProductDecoder,
    ShortestPathPredictor,
    TurnClassifier,
)
from src.modules.encoder import DGATConvEncoder
from src.utils.model_outputs import (
    DGatForPreTrainingOutput,
    DGatForPreTrainingTrainerOutput,
)


class DGatConvPretraining(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 edge_dim: int, encoder_layer_class: MessagePassing,
                 num_encoder_layers: int, **kwargs):
        super().__init__()

        self.encoder = DGATConvEncoder(input_dim=input_dim,
                                       hidden_dim=hidden_dim,
                                       output_dim=output_dim,
                                       edge_dim=edge_dim,
                                       layer_class=encoder_layer_class,
                                       num_layers=num_encoder_layers,
                                       **kwargs)

    def forward(self,
                x: torch.FloatTensor,
                edge_index: torch.LongTensor,
                order_ids: Optional[torch.LongTensor] = None,
                turn_ids: Optional[torch.LongTensor] = None,
                edge_attr: Optional[torch.LongTensor] = None) -> torch.Tensor:
        return self.encoder(x=x,
                            edge_index=edge_index,
                            order_ids=order_ids,
                            turn_ids=turn_ids,
                            edge_attr=edge_attr)

    def compute_loss(self,
                     output: ModelOutput,
                     weight: Optional[dict[str, float]] = None,
                     **kwargs) -> ModelOutput:
        raise NotImplementedError

    def initialize_encoder(self, pretrained_weights_path: Union[str,
                                                                None]) -> None:
        if pretrained_weights_path is not None:
            print('Use pretrained weights for Encoder.')
            state_dict = torch.load(pretrained_weights_path)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict = {
                k.removeprefix('encoder.'): v
                for k, v in state_dict.items() if k.startswith('encoder.')
            }
            self.encoder.load_state_dict(state_dict, strict=False)
        else:
            print('No pre-trained weights for Encoder.')


class DGatForPreTraining(DGatConvPretraining):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 edge_dim: int, encoder_layer_class: MessagePassing,
                 num_encoder_layers: int, **kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, edge_dim,
                         encoder_layer_class, num_encoder_layers, **kwargs)

        self.path_predictor = ShortestPathPredictor(hidden_dim=output_dim,
                                                    concat=True)
        self.turn_classifier = TurnClassifier(hidden_dim=output_dim,
                                              concat=True)
        self.adj_reconstructor = InnerProductDecoder()

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor,
                order_ids: Optional[torch.Tensor] = None,
                turn_ids: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                **kwargs) -> DGatForPreTrainingOutput:
        # [n nodes, hidden dim]
        h_x = self.encoder(x, edge_index, order_ids, turn_ids, edge_attr)
        if len(h_x) == 2:
            h_x, _ = h_x

        batched_h_x = [h_x[batch == i] for i in batch.unique(sorted=True)]

        sample_edge_indices = []
        path_labels = []
        turn_labels = []
        for batch_idx, dialogue_embeddings in enumerate(batched_h_x):
            start_index = batch.tolist().index(batch_idx)
            sample_edge_index = self.sample_node_pairs(
                dialogue_embeddings.size(0),
                num_samples=20,
                start_index=start_index)

            sample_edge_indices += sample_edge_index

            for p in sample_edge_index:
                p_i = p[0] - start_index
                p_j = p[1] - start_index
                path_diff = abs(p_i - p_j)

                path_labels.append(path_diff)
                turn_labels.append(float(path_diff == 1))

        sample_edge_indices = torch.tensor([[e[0] for e in sample_edge_indices],
                                            [e[1]
                                             for e in sample_edge_indices]],
                                           dtype=torch.long,
                                           device=x.device)

        path_labels = torch.tensor(path_labels,
                                   dtype=torch.float,
                                   device=x.device).squeeze(0).unsqueeze(1)
        turn_labels = torch.tensor(turn_labels,
                                   dtype=torch.float,
                                   device=x.device).unsqueeze(1)

        # [n nodes, 1]
        path_logits = self.path_predictor(h_x, sample_edge_indices)

        # [n nodes, 1]
        turn_logits = self.turn_classifier(h_x, sample_edge_indices)

        # [n nodes, n nodes]
        # Use forward_all to compute all nodes with all other nodes, so it can be considered as cross-graph learning
        adj_recon_logits = self.adj_reconstructor.forward_all(h_x,
                                                              sigmoid=False)

        return DGatForPreTrainingOutput(
            path_logits=path_logits,
            turn_logits=turn_logits,
            adj_recon_logits=adj_recon_logits,
            path_labels=path_labels,
            turn_labels=turn_labels,
            # If no batch is specified, adj matrix: [num_nodes, num_nodes] (all nodes considered across all graphs in the batch)
            # Otherwise [unique batch size; num_nodes in batch, num_nodes in batch]
            adj_recon_labels=to_dense_adj(edge_index).squeeze(0))

    def sample_node_pairs(self,
                          num_nodes: int,
                          num_samples: int,
                          start_index: Optional[int] = 0):
        num_samples = min(num_samples, num_nodes * (num_nodes - 1) // 2)

        start, end = start_index, start_index + num_nodes
        all_possible_pairs = [(i, j) for i in range(start, end)
                              for j in range(i + 1, end)]
        sampled_pairs = np.random.choice(len(all_possible_pairs),
                                         size=num_samples,
                                         replace=False)

        return [all_possible_pairs[i] for i in sampled_pairs]

    def compute_loss(self,
                     output: DGatForPreTrainingOutput,
                     weight: Optional[dict[str, float]] = None,
                     **kwargs) -> DGatForPreTrainingTrainerOutput:
        path_loss = F.mse_loss(output.path_logits, output.path_labels)
        turn_loss = F.binary_cross_entropy_with_logits(output.turn_logits,
                                                       output.turn_labels)
        adj_recon_loss = F.binary_cross_entropy_with_logits(
            output.adj_recon_logits, output.adj_recon_labels)

        total_loss = path_loss + turn_loss + adj_recon_loss

        return DGatForPreTrainingTrainerOutput(total_loss=total_loss,
                                               path_loss=path_loss,
                                               turn_loss=turn_loss,
                                               adj_recon_loss=adj_recon_loss)
