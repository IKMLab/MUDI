import torch
import torch.nn as nn


class SoftPromptEmbedder(nn.Module):

    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        """appends learned embedding to 

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super().__init__()

        self.wte = wte
        self.weight = self.wte.weight
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(
            self.initialize_embedding(wte, n_tokens, random_range,
                                      initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(
            -random_range, random_range)

    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        if len(tokens.size()) == 1 or tokens.size(1) < self.n_tokens:
            return self.wte(tokens)

        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(
            input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)


class OrderEmbedding(nn.Embedding):
    """
    Order embedding layer.
    """

    def __init__(self, hidden_dim, num_order_embeddings, **kwargs):
        super().__init__(num_order_embeddings, hidden_dim, **kwargs)


class TurnEmbedding(nn.Embedding):
    """
    Turn embedding layer.
    """

    def __init__(self, hidden_dim, num_turn_embeddings, **kwargs):
        super().__init__(num_turn_embeddings, hidden_dim, **kwargs)


class RoleEmbedding(nn.Embedding):
    """
    Role embedding layer.
    """

    def __init__(self, hidden_dim, num_role_embeddings, **kwargs):
        super().__init__(num_role_embeddings, hidden_dim, **kwargs)


class DialogueEmbedder(nn.Module):
    """
    Dialogue Embedder.
    """

    def __init__(self,
                 hidden_dim,
                 num_order_embeddings,
                 num_turn_embeddings,
                 num_role_embeddings,
                 dropout_rate=0.1):
        super().__init__()

        # self.order_embedding = OrderEmbedding(hidden_dim, num_order_embeddings)
        self.turn_embedding = TurnEmbedding(hidden_dim, num_turn_embeddings)
        # self.role_embedding = RoleEmbedding(hidden_dim, num_role_embeddings)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.init_weights()

    def init_weights(self):
        # nn.init.xavier_uniform_(self.order_embedding.weight)
        nn.init.xavier_uniform_(self.turn_embedding.weight)
        # nn.init.xavier_uniform_(self.role_embedding.weight)

    def forward(self, order_ids, turn_ids, role_ids):
        # embed =  self.order_embedding(order_ids) + \
        #     self.turn_embedding(turn_ids) + \
        #     self.role_embedding(role_ids)
        embed = self.turn_embedding(turn_ids)

        return self.dropout(embed)
