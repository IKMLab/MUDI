from typing import Union

import torch
from torch_geometric.data import Batch as GraphBatch

from src.data.dataset import DialogueGraphData
from src.utils.constants import ModelTrainMode


class DialogueGraphDataCollator:

    def __init__(self, train_mode: ModelTrainMode):
        self.train_mode = train_mode

    def __call__(self, data_list: list[DialogueGraphData]) -> GraphBatch:
        """Collate function for `DialogueGraphData`.

        Args:
            data_list (list[DialogueGraphData]): batch of `DialogueGraphData`

        Returns:
            batch (Batch): collated `DialogueGraphData`
        """

        batch = GraphBatch.from_data_list(data_list)

        if self.train_mode == ModelTrainMode.FINETUNEING:
            num_nodes_cum = 0
            user_personas_edge_index_batch = []
            for data in data_list:
                user_personas_edge_index = data.user_personas_edge_index + num_nodes_cum
                user_personas_edge_index_batch.append(user_personas_edge_index)
                num_nodes_cum += data.num_user_personas

            batch.user_personas_edge_index = torch.cat(
                user_personas_edge_index_batch, dim=1)

        return batch


class PersonalizedDialogueGenerationDataCollator:

    def __call__(
        self,
        data_list: list,
    ) -> dict[str, Union[GraphBatch, dict[str, torch.Tensor]]]:
        """Collate function for `ConvAI2ForPersonalizedDialogueGenerationDataset`.

        Args:
            data_list (list): batch of `DialogueGraphData` and `Generator` input

        Returns:
            batch (dict[str, Union[GraphBatch, dict[str, torch.Tensor]]]): collated `DialogueGraphData` and `Generator` input
        """

        # Collate `DialogueGraphData`
        dialogue_encoder_inputs = [
            data['dialogue_encoder_input'] for data in data_list
        ]

        dialogue_encoder_input_batch = GraphBatch.from_data_list(
            dialogue_encoder_inputs)

        num_nodes_cum = 0
        user_personas_edge_index_batch = []
        for data in dialogue_encoder_inputs:
            user_personas_edge_index = data.user_personas_edge_index + num_nodes_cum
            user_personas_edge_index_batch.append(user_personas_edge_index)
            num_nodes_cum += data.num_user_personas

        dialogue_encoder_input_batch.user_personas_edge_index = torch.cat(
            user_personas_edge_index_batch, dim=1)

        # Collate `Generator` input
        generator_inputs = [data['generator_input'] for data in data_list]
        input_ids = [data['input_ids'] for data in generator_inputs]
        attention_mask = [data['attention_mask'] for data in generator_inputs]
        decoder_input_ids = [
            data['decoder_input_ids'] for data in generator_inputs
        ]
        decoder_attention_mask = [
            data['decoder_attention_mask'] for data in generator_inputs
        ]

        generator_input_batch = {
            'input_ids':
            torch.stack(input_ids, dim=0).squeeze(1),
            'attention_mask':
            torch.stack(attention_mask, dim=0).squeeze(1),
            'decoder_input_ids':
            torch.stack(decoder_input_ids, dim=0).squeeze(1),
            'decoder_attention_mask':
            torch.stack(decoder_attention_mask, dim=0).squeeze(1)
        }

        if 'multiple_choice_label' in generator_inputs[0]:
            multiple_choice_label = [
                data['multiple_choice_label'] for data in generator_inputs
            ]
            generator_input_batch['multiple_choice_label'] = torch.stack(
                multiple_choice_label, dim=0)

        return {
            'dialogue_encoder_input': dialogue_encoder_input_batch,
            'generator_input': generator_input_batch,
        }
