import torch
import torch.nn as nn
from torch_geometric.nn.dense import Linear


class ShortestPathPredictor(nn.Module):

    def __init__(self, hidden_dim: int, concat: bool = True):
        super().__init__()

        self.relu = nn.ReLU()
        self.path_linear = nn.Sequential(
            nn.Linear(
                hidden_dim if not concat else hidden_dim * 2,
                hidden_dim,
            ), nn.ReLU(), nn.Linear(
                hidden_dim,
                1,
            ))

        self.concat = concat

    def forward(self, hidden_states: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        if self.concat:
            # [num_edges, hidden_dim + hidden_dim]
            edge_embeddings = torch.cat(
                [hidden_states[edge_index[0]], hidden_states[edge_index[1]]],
                dim=-1,
            )
        else:
            # [num_edges, hidden_dim]
            edge_embeddings = (hidden_states[edge_index[0]] +
                               hidden_states[edge_index[1]])

        # [num_edges, 1]
        logits = self.path_linear(self.relu(edge_embeddings))

        return logits


class TurnClassifier(nn.Module):

    def __init__(self, hidden_dim: int, concat: bool = True):
        super().__init__()

        self.relu = nn.ReLU()
        self.turn_linear = nn.Sequential(
            nn.Linear(
                hidden_dim if not concat else hidden_dim * 2,
                hidden_dim,
            ), nn.ReLU(), nn.Linear(
                hidden_dim,
                1,
            ))

        self.concat = concat

    def forward(self, hidden_states: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        if self.concat:
            # [num_edges, hidden_dim + hidden_dim]
            edge_embeddings = torch.cat(
                [hidden_states[edge_index[0]], hidden_states[edge_index[1]]],
                dim=-1,
            )
        else:
            # [num_edges, hidden_dim]
            edge_embeddings = (hidden_states[edge_index[0]] +
                               hidden_states[edge_index[1]])

        # [num_edges, 1]
        logits = self.turn_linear(self.relu(edge_embeddings))

        return logits


class InnerProductDecoder(nn.Module):

    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        sigmoid: bool = True,
    ) -> torch.Tensor:
        """Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            hidden_states (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """

        value = (hidden_states[edge_index[0]] *
                 hidden_states[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self,
                    hidden_states: torch.Tensor,
                    sigmoid: bool = True) -> torch.Tensor:
        """Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            hidden_states (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """

        adj = torch.matmul(hidden_states, hidden_states.t())
        return torch.sigmoid(adj) if sigmoid else adj


class CoherenceRelationsClassifier(nn.Module):

    def __init__(self, hidden_dim: int, num_edge_labels: int):
        super().__init__()

        self.n_hidden = hidden_dim
        self.mlp = nn.Sequential(Linear(self.n_hidden * 3, self.n_hidden),
                                 nn.ReLU(), nn.Dropout(0.1),
                                 Linear(self.n_hidden, self.n_hidden // 2),
                                 nn.ReLU(), nn.Dropout(0.1),
                                 Linear(self.n_hidden // 2, num_edge_labels))

    def forward(self, node_embeddings, edge_index, edge_attr=None):
        x = node_embeddings[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = x

        return self.mlp(out)


class NextResponseTypePredictor(nn.Module):

    def __init__(self,
                 node_embedding_dim: int,
                 hidden_dim: int,
                 num_edge_labels: int,
                 contains_edge_attr: bool = False):
        super().__init__()

        self.relu = nn.ReLU()

        self.seq_predictor = nn.GRU(
            node_embedding_dim *
            2 if contains_edge_attr else node_embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1)

        self.cls_linear = nn.Linear(hidden_dim, num_edge_labels)

        self.contains_edge_attr = contains_edge_attr

    def forward(self, batch_node_embeddings, batch_edge_attr=None, device=None):
        node_embeddings = torch.tensor(
            [j.tolist() for i in batch_node_embeddings for j in i],
            device=device)
        direct_logits = self.cls_linear(self.relu(node_embeddings))

        if batch_edge_attr is None:
            if self.contains_edge_attr:
                raise ValueError(
                    'Contains edge attributes but edge attributes are not provided.'
                )
            batch_edge_attr = [None] * len(batch_node_embeddings)

        seq_logits = []
        for embeds, attrs in zip(batch_node_embeddings, batch_edge_attr):
            if self.contains_edge_attr:
                seq_input = torch.cat((embeds, attrs), dim=-1)
            else:
                seq_input = embeds

            seq_outs, _ = self.seq_predictor(seq_input.float())

            cls_outs = self.cls_linear(self.relu(seq_outs))
            seq_logits.append(cls_outs)

        seq_logits = torch.tensor([j.tolist() for i in seq_logits for j in i],
                                  device=device)

        return direct_logits, seq_logits
