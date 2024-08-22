from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from transformers import BertModel, RobertaModel

from src.modules.layers import DialogueGATConv


class DGATConvEncoder(torch.nn.Module):
    """Encode the dialogue graph.

    Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        output_dim (int): Output dimension
        edge_dim (int): Edge dimension
        layer_class (MessagePassing): GNN Layer class
        num_layers (int): Number of layers
        edge_updates (bool): Update edge attributes or not
        dropout (float): Dropout rate
        **kwargs: Keyword arguments for layer_class
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 edge_dim: int,
                 layer_class: MessagePassing,
                 num_layers: int,
                 edge_updates: bool = False,
                 dropout: float = 0.009,
                 **kwargs):
        super().__init__()

        n_heads = kwargs.get('heads', 1)
        tmp_out = hidden_dim // n_heads

        self.n_hidden = tmp_out * n_heads
        self.n_heads = n_heads
        self.num_gnn_layers = num_layers
        self.edge_updates = edge_updates and edge_dim > 0
        self.dropout = dropout

        self.node_emb = nn.Linear(input_dim, self.n_hidden)
        if edge_dim > 0:
            self.edge_emb = nn.Linear(edge_dim, self.n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            if self.edge_updates:
                self.emlps.append(
                    nn.Sequential(
                        nn.Linear(3 * self.n_hidden, self.n_hidden),
                        nn.ReLU(),
                        nn.Linear(self.n_hidden, self.n_hidden),
                    ))
            self.convs.append(
                layer_class(self.n_hidden,
                            tmp_out,
                            self.n_heads,
                            concat=True,
                            dropout=self.dropout,
                            add_self_loops=True,
                            edge_dim=self.n_hidden))
            self.batch_norms.append(BatchNorm(self.n_hidden))

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                order_ids: Optional[torch.Tensor] = None,
                turn_ids: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None):
        x = self.node_emb(x)
        if getattr(self, 'edge_emb', None):
            edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            if isinstance(self.convs[i], DialogueGATConv):
                x = (x + F.relu(self.batch_norms[i](self.convs[i](
                    x, edge_index, edge_attr, order_ids, turn_ids)))) / 2
            else:
                x = (x + F.relu(self.batch_norms[i]
                                (self.convs[i](x, edge_index, edge_attr)))) / 2

            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](
                    torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr],
                              dim=-1)) / 2

        return x, edge_attr


class PersonaGraphEncoder(nn.Module):
    """Encode the persona graph.

    Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        output_dim (int): Output dimension
        layer_class (MessagePassing): GNN Layer class
        num_layers (int): Number of layers
        **kwargs: Keyword arguments for layer_class
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 layer_class: MessagePassing, num_layers: int, **kwargs):
        super().__init__()

        num_heads = kwargs.get('heads', 1)
        dim_per_head = hidden_dim // num_heads

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()

        self.convs.append(
            layer_class(input_dim,
                        dim_per_head,
                        heads=num_heads,
                        concat=True,
                        add_self_loops=True,
                        dropout=0.1))
        self.lins.append(Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(
                layer_class(hidden_dim,
                            dim_per_head,
                            heads=num_heads,
                            concat=True,
                            add_self_loops=True,
                            dropout=0.1))
            self.lins.append(Linear(hidden_dim, hidden_dim))

        self.convs.append(
            layer_class(hidden_dim,
                        output_dim,
                        heads=1,
                        concat=False,
                        add_self_loops=True,
                        dropout=0.1))
        self.lins.append(Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.dropout(x, p=0.1, training=self.training)
        for conv, lin in zip(self.convs[:-1], self.lins[:-1]):
            x = conv(x, edge_index) + lin(x)
            x = x.relu()
            x = F.dropout(x, p=0.1, training=self.training)

        x = self.convs[-1](x, edge_index) + self.lins[-1](x)

        # Readout layer to get graph embeddings
        graph_embeddings = self.readout(x, batch)

        return x, graph_embeddings

    def readout(self, x, batch):
        return global_mean_pool(x, batch)


class UtteranceEncoder(nn.Module):
    """Encode the utterance.

    Args:
        pretrained_model_name (Literal['bert', 'roberta']): Pretrained model name
        freeze (Optional[bool]): Freeze the encoder weights or not
    """

    def __init__(self,
                 pretrained_model_name: Literal['bert', 'roberta'],
                 freeze: Optional[bool] = True):
        super().__init__()

        if pretrained_model_name == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        elif pretrained_model_name == 'roberta':
            self.encoder = RobertaModel.from_pretrained('roberta-base')
        else:
            raise ValueError(
                f'Invalid pretrained model name: {pretrained_model_name}')

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        cls_tensor = enc_out[0][:, 0, :]

        return cls_tensor
