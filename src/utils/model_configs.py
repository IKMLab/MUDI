from typing import Literal, Optional

import torch
from accelerate import Accelerator
from easydict import EasyDict
from transformers.models.bart.configuration_bart import BartConfig


class ModelArguments(EasyDict):
    input_dim: int
    hidden_dim: int
    output_dim: int
    edge_dim: int
    layer_type: Literal['GAT', 'GATv2', 'RGAT', 'DialogueGAT']
    num_layers: int
    # Only for Fine-tuning
    share_edge_encoder: bool
    pretrained_utterance_encoder: Literal['none', 'bert', 'roberta']
    pretrained_weights_path: str


class TrainingConfigs(EasyDict):
    device: torch.device
    distributed: bool
    model_name: str
    dataset_name: str
    checkpoint: str
    learner: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    endure_times: int
    accelerator: Optional[Accelerator]


class PersonalizedDialogueGeneratorConfig(BartConfig):

    def __init__(self,
                 dialogue_encoder_input_dim: int = 768,
                 dialogue_encoder_hidden_dim: int = 512,
                 dialogue_encoder_output_dim: int = 512,
                 dialogue_encoder_edge_dim: int = 17,
                 dialogue_encoder_layer_type: Literal['GAT', 'GATv2', 'RGAT',
                                                      'DialogueGAT'] = None,
                 dialogue_encoder_num_encoder_layers: int = 2,
                 dialogue_encoder_utterance_encoder_class: Literal[
                     'none', 'bert', 'roberta'] = 'none',
                 dialogue_encoder_pretrained_weights_path: Optional[str] = None,
                 dialogue_encoder_pretrained_encoder_weights_path: Optional[
                     str] = None,
                 dialogue_encoder_heads: int = 4,
                 dialogue_encoder_add_self_loops: bool = True,
                 freeze_dialogue_encoder: bool = False,
                 coherence_attn_strategy: Literal['SP', 'Emb',
                                                  'SP+Emb'] = 'SP+Emb',
                 graph_encoder_strategy: Literal['Attn', 'Add', 'C', 'P',
                                                 'Random', 'None'] = 'Attn',
                 **bart_config_kwargs: dict[BartConfig]):

        self.dialogue_encoder_input_dim = dialogue_encoder_input_dim
        self.dialogue_encoder_hidden_dim = dialogue_encoder_hidden_dim
        self.dialogue_encoder_output_dim = dialogue_encoder_output_dim
        self.dialogue_encoder_edge_dim = dialogue_encoder_edge_dim
        self.dialogue_encoder_layer_type = dialogue_encoder_layer_type
        self.dialogue_encoder_num_encoder_layers = dialogue_encoder_num_encoder_layers
        self.dialogue_encoder_utterance_encoder_class = dialogue_encoder_utterance_encoder_class
        self.dialogue_encoder_pretrained_weights_path = dialogue_encoder_pretrained_weights_path
        self.dialogue_encoder_pretrained_encoder_weights_path = dialogue_encoder_pretrained_encoder_weights_path
        self.dialogue_encoder_heads = dialogue_encoder_heads
        self.dialogue_encoder_add_self_loops = dialogue_encoder_add_self_loops

        self.freeze_dialogue_encoder = freeze_dialogue_encoder

        self.summary_type = 'cls_index'
        self.summary_first_dropout = 0.1
        self.summary_proj_to_labels = True
        self.summary_use_proj = True

        self.coherence_attn_strategy = coherence_attn_strategy
        self.graph_encoder_strategy = graph_encoder_strategy

        super().__init__(**bart_config_kwargs)

        self.num_labels = 2
