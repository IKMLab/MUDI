from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


class DialogueGATConv(GATv2Conv):

    def __init__(self,
                 in_channels: Union[int, tuple[int, int]],
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 edge_dim: Optional[int] = None,
                 fill_value: Union[float, Tensor, str] = 'mean',
                 bias: bool = True,
                 share_weights: bool = False,
                 **kwargs):
        super().__init__(in_channels, out_channels, heads, concat,
                         negative_slope, dropout, add_self_loops, edge_dim,
                         fill_value, bias, share_weights, **kwargs)

        self.order_att = Parameter(torch.empty(1, heads, out_channels))
        self.turn_att = Parameter(torch.empty(1, heads, out_channels))

        # initialize the learnable attention weights
        glorot(self.order_att)
        glorot(self.turn_att)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        order_ids: OptTensor = None,
        turn_ids: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            tuple[Tensor, tuple[Tensor, Tensor]],
            tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)

        if order_ids is not None and turn_ids is not None:
            _, order_edge_index = self.add_order_connection(edge_index,
                                                            order_ids,
                                                            max_distance=3)
            _, turn_edge_index = self.add_turn_connection(edge_index, turn_ids)

            order_alpha = self.order_edge_update(x_i=x_r,
                                                 x_j=x_l,
                                                 edge_index=order_edge_index,
                                                 order_ids=order_ids)
            turn_alpha = self.turn_edge_update(x_i=x_r,
                                               x_j=x_l,
                                               edge_index=turn_edge_index,
                                               turn_ids=turn_ids)

            edge_index = torch.cat(
                [edge_index, order_edge_index, turn_edge_index], dim=1)
            alpha = torch.cat([alpha, order_alpha, turn_alpha], dim=0)

        # propagate_type: (x: PairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def add_order_connection(
            self,
            edge_index,
            order_ids,
            max_distance=3) -> tuple[Tensor, Union[Tensor, list]]:
        num_nodes = len(order_ids)

        additional_edges = [[], []]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if order_ids[j] == 0:  # break if the next dialogue starts
                    break

                # connect the nodes if they are neighbors less than max_distance
                if abs(order_ids[j] - order_ids[i]) <= max_distance:
                    additional_edges[0].append(i)
                    additional_edges[1].append(j)

        if additional_edges:
            additional_edges = torch.tensor(
                additional_edges, dtype=torch.long).to(edge_index.device)
            new_edge_index = torch.cat([edge_index, additional_edges], dim=1)
        else:
            new_edge_index = edge_index

        return new_edge_index, additional_edges

    def add_turn_connection(self, edge_index,
                            turn_ids) -> tuple[Tensor, Union[Tensor, list]]:
        num_nodes = len(turn_ids)

        additional_edges = [[], []]
        for i in range(num_nodes):
            # NOTE: 這邊可以簡化不用使用迴圈，由於下一個節點基本上一定是屬於同一個 turn
            for j in range(i + 1, num_nodes):
                if turn_ids[i] == turn_ids[j]:
                    additional_edges[0].append(i)
                    additional_edges[1].append(j)
                    additional_edges[0].append(j)
                    additional_edges[1].append(i)

                    break

        if additional_edges:
            additional_edges = torch.tensor(
                additional_edges, dtype=torch.long).to(edge_index.device)
            new_edge_index = torch.cat([edge_index, additional_edges], dim=1)
        else:
            new_edge_index = edge_index

        return new_edge_index, additional_edges

    def order_edge_update(self,
                          x_i: Tensor,
                          x_j: Tensor,
                          edge_index: Tensor,
                          order_ids: Tensor,
                          decay_rate: float = 0.1) -> Tensor:
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        target, source = edge_index[i], edge_index[j]

        # calculate the difference of order_ids
        order_diff = (order_ids[target] - order_ids[source]).abs()

        # use exponential decay to calculate the attention scores based on the order difference
        att_scores = torch.exp(-decay_rate * order_diff.float())

        mutable_size = x_i.shape[0]
        kwargs = self.edge_collect(
            edge_index,
            (x_i, x_j),
            att_scores.unsqueeze(-1),
            size=(mutable_size, mutable_size),
        )

        x = F.leaky_relu(kwargs.x_i + kwargs.x_j, self.negative_slope)
        alpha = (x * self.order_att).sum(dim=-1) * kwargs.edge_attr
        alpha = softmax(alpha, target)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def turn_edge_update(self, x_i: Tensor, x_j: Tensor, edge_index: Tensor,
                         turn_ids: Tensor) -> Tensor:
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        target, source = edge_index[i], edge_index[j]

        # Check if the nodes belong to the same turn
        same_turn = (turn_ids[source] == turn_ids[target])

        # Use a fixed attention score for edges within the same turn
        att_scores = torch.where(same_turn,
                                 torch.tensor(1.0, device=same_turn.device),
                                 torch.tensor(0.0, device=same_turn.device))

        mutable_size = x_i.shape[0]
        kwargs = self.edge_collect(
            edge_index,
            (x_i, x_j),
            att_scores.unsqueeze(-1),
            size=(mutable_size, mutable_size),
        )

        x = F.leaky_relu(kwargs.x_i + kwargs.x_j, self.negative_slope)
        alpha = (x * self.turn_att).sum(dim=-1) * kwargs.edge_attr
        alpha = softmax(alpha, target)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha
