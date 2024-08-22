from typing import Union

import torch
from torch import Tensor
from torch_geometric.data import Data as GraphData
from torch_geometric.data.data import BaseData as GraphBaseData

from src.base.base_dataset import BaseGraphDataset
from src.utils.constants import (
    COHERENCE_RELATIONS,
    COHREL2ID,
    NEW_SPEICAL_TOKENS_MAP,
)


class DialogueGraphData(GraphData):

    def __init__(self,
                 x: Union[Tensor, None] = None,
                 edge_index: Union[Tensor, None] = None,
                 edge_attr: Union[Tensor, None] = None,
                 y: Union[Tensor, None] = None,
                 pos: Union[Tensor, None] = None,
                 time: Union[Tensor, None] = None,
                 **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

    def get(self, key, item):
        if key == 'edge_attr':
            return self.edge_attr[item]
        else:
            return super().get(key, item)


class ConvAI2ForDialogueGraphEncodingDataset(BaseGraphDataset):

    def __init__(self,
                 root,
                 raw_file_name,
                 processed_file_dir,
                 processed_file_name=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 utterance_tokenizer=None,
                 reverse_edge=False,
                 directed=True):
        self.utterance_tokenizer = utterance_tokenizer
        self.reverse_edge = reverse_edge
        self.directed = directed

        super().__init__(root, raw_file_name, processed_file_dir,
                         processed_file_name, transform, pre_transform,
                         pre_filter)

    def _process_data(self, data: dict) -> GraphBaseData:
        conv = data.copy()

        num_utterances = len(conv['whole_dialogue'][:-1])
        num_personas = len(conv['persona'])

        edge_from_indices = []
        edge_to_indices = []
        edge_labels = []
        edge_features = []
        for item in conv['coherence']:
            from_index = item['from_index']
            to_index = item['to_index']
            labels = item['labels']

            if self.reverse_edge:
                edge_from_indices.append(to_index)
                edge_to_indices.append(from_index)
            else:
                edge_from_indices.append(from_index)
                edge_to_indices.append(to_index)

            # multi-label
            coherence_relations_label = [0] * len(COHERENCE_RELATIONS)
            for label in labels:
                coherence_relations_label[COHREL2ID[label]] = 1

            edge_labels.append(coherence_relations_label)

            edge_features.append([
                abs(to_index - from_index),
                abs(((to_index // 2) + 1) - ((from_index // 2) + 1)),
                int(from_index % 2 == to_index % 2)
            ])

        # x: Tensor, shape [num_utterances, num_node_features: 768]
        node_features = torch.tensor(conv['encoded_dialogue'][:-1],
                                     dtype=torch.float)

        # edge_index: LongTensor, shape [2, num_edges]
        edge_index = torch.tensor([edge_from_indices, edge_to_indices],
                                  dtype=torch.long)

        # edge_attr: Tensor, shape [num_edges, num_edge_features: 3]
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        # y: Tensor, shape [num_edges, num_edge_labels: 17]
        edge_labels = torch.tensor(edge_labels, dtype=torch.float)

        if not self.directed:
            from_indices = torch.cat([edge_index[0], edge_index[1]])
            to_indices = torch.cat([edge_index[1], edge_index[0]])
            edge_index = torch.stack([from_indices, to_indices], dim=0)
            edge_features = torch.cat([edge_features, edge_features], dim=0)
            edge_labels = torch.cat([edge_labels, edge_labels], dim=0)

        encoded_dialogue, encoded_persona = None, None
        if self.utterance_tokenizer:
            # encode dialogues and personas
            # encoded_dialogues: LongTensor, shape [num_utterances, 512] for each property
            encoded_dialogue = self.utterance_tokenizer(
                conv['whole_dialogue'][:-1],
                padding='max_length',
                truncation=True,
                return_tensors='pt')
            # encoded_personas: LongTensor, shape [num_personas, 512]  for each property
            encoded_persona = self.utterance_tokenizer(conv['persona'],
                                                       padding='max_length',
                                                       truncation=True,
                                                       return_tensors='pt')

        # exteranl information for node representation learning
        # order_ids: LongTensor, shape [num_utterances], order of the nodes in the conversation
        # e.g., [0, 1, 2, 3, 4, 5]
        order_ids = torch.tensor([i for i in range(num_utterances)],
                                 dtype=torch.long)
        # turn_ids: LongTensor, shape [num_utterances], turn of the nodes in the conversation
        # e.g., [0, 0, 1, 1, 2, 2]
        turn_ids = torch.tensor([i // 2 for i in range(num_utterances)],
                                dtype=torch.long)
        # role_ids: LongTensor, shape [num_utterances], which speaker is speaking. 0: partner, 1: user
        # e.g., [0, 1, 0, 1, 0, 1]
        role_ids = torch.tensor([i % 2 for i in range(num_utterances)],
                                dtype=torch.long)

        persona_edge_from_indices = []
        persona_edge_to_indices = []
        # complete graph for personas
        for i in range(num_personas):
            for j in range(num_personas):
                # skip self-loop
                if i == j:
                    continue
                persona_edge_from_indices.append(i)
                persona_edge_to_indices.append(j)

        # user_persona_edge_index: LongTensor, shape [2, num_edges: num_personas * (num_personas - 1)]
        persona_edge_index = torch.tensor(
            [persona_edge_from_indices, persona_edge_to_indices],
            dtype=torch.long)

        # x_user_personas: Tensor, shape [num_personas, num_node_features: 768]
        persona_node_features = torch.tensor(conv['encoded_persona'],
                                             dtype=torch.float)

        # Remove the last node (response node)
        max_node_id = torch.max(edge_index[1]).item()
        mask = edge_index[1] != max_node_id
        edge_index = edge_index[:, mask]

        last_edge_labels = edge_labels[-1]
        edge_labels = edge_labels[mask]

        assert len(node_features) == num_utterances
        assert len(persona_node_features) == num_personas
        assert len(edge_index[0]) == len(edge_labels)
        assert len(edge_labels[0]) == len(COHERENCE_RELATIONS)
        assert len(persona_edge_index[0]) == (num_personas * (num_personas - 1))

        return DialogueGraphData(
            x=node_features,
            x_user_personas=persona_node_features,
            edge_index=edge_index,
            user_personas_edge_index=persona_edge_index,
            edge_attr=edge_labels,
            y=edge_labels,
            y_res_type=torch.cat([
                edge_labels[(edge_index[1] - edge_index[0]) == 1].long(),
                last_edge_labels.unsqueeze(0)
            ]),
            res_type_edge_index=edge_index[:, (edge_index[1] -
                                               edge_index[0] == 1)].long(),
            utterance_input_ids=encoded_dialogue.input_ids
            if encoded_dialogue else None,
            utterance_attention_mask=encoded_dialogue.attention_mask
            if encoded_dialogue else None,
            persona_input_ids=encoded_persona.input_ids
            if encoded_persona else None,
            persona_attention_mask=encoded_persona.attention_mask
            if encoded_persona else None,
            order_ids=order_ids,
            turn_ids=turn_ids,
            role_ids=role_ids,
            num_user_personas=num_personas,
            num_utterances=num_utterances,
        )


class DialogueGraphPreTrainingDataset(BaseGraphDataset):

    def __init__(self,
                 root,
                 raw_file_name,
                 processed_file_dir,
                 processed_file_name=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, raw_file_name, processed_file_dir,
                         processed_file_name, transform, pre_transform,
                         pre_filter)

    def _process_data(self, data: dict) -> GraphBaseData:
        conv = data.copy()

        num_utterances = len(conv['encoded_dialogue'])

        # x: Tensor, shape [num_utterances, num_node_features: 768]
        node_features = torch.tensor(conv['encoded_dialogue'],
                                     dtype=torch.float)

        # edge_index: LongTensor, shape [2, num_edges]
        edge_index = torch.tensor(
            [list(range(num_utterances - 1)),
             list(range(1, num_utterances))],
            dtype=torch.long)

        # exteranl information for node representation learning
        # order_ids: LongTensor, shape [num_utterances], order of the nodes in the conversation
        # e.g., [0, 1, 2, 3, 4, 5]
        order_ids = torch.tensor([i for i in range(num_utterances)],
                                 dtype=torch.long)
        # turn_ids: LongTensor, shape [num_utterances], turn of the nodes in the conversation
        # e.g., [0, 0, 1, 1, 2, 2]
        turn_ids = torch.tensor([i // 2 for i in range(num_utterances)],
                                dtype=torch.long)

        return DialogueGraphData(
            x=node_features,
            edge_index=edge_index,
            order_ids=order_ids,
            turn_ids=turn_ids,
        )


class DailyDialogDataset(DialogueGraphPreTrainingDataset):

    def __init__(self,
                 root,
                 raw_file_name,
                 processed_file_dir,
                 processed_file_name=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, raw_file_name, processed_file_dir,
                         processed_file_name, transform, pre_transform,
                         pre_filter)


class RccDataset(DialogueGraphPreTrainingDataset):

    def __init__(self,
                 root,
                 raw_file_name,
                 processed_file_dir,
                 processed_file_name=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, raw_file_name, processed_file_dir,
                         processed_file_name, transform, pre_transform,
                         pre_filter)


class ConvAI2ForPersonalizedDialogueGenerationDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset,
                 graph_dataset: ConvAI2ForDialogueGraphEncodingDataset,
                 tokenizer,
                 nearest_k_turn: int = 5):
        self.raw_dataset = dataset
        self.graph_dataset = graph_dataset
        self.tokenizer = tokenizer

        self.nearest_k_turn = nearest_k_turn

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(
            self, index: int
    ) -> dict[str, Union[DialogueGraphData, dict[str, Tensor]]]:
        data = self.raw_dataset[index]

        num_turns = len(data['whole_dialogue']) // 2
        nearest_k_turn = min(num_turns, self.nearest_k_turn)

        context = data['whole_dialogue'][(-nearest_k_turn - 1) * 2:][:-1]
        response = data['whole_dialogue'][-1]

        input_seq = f" {NEW_SPEICAL_TOKENS_MAP['persona']} " + f" {NEW_SPEICAL_TOKENS_MAP['persona']} ".join(
            data['persona'])

        for i, utterance in enumerate(context):
            if i % 2 == 0:
                input_seq += NEW_SPEICAL_TOKENS_MAP['query'] + utterance
            else:
                input_seq += NEW_SPEICAL_TOKENS_MAP['response'] + utterance

        dialogue_encoding = self.tokenizer(input_seq,
                                           add_special_tokens=True,
                                           padding='max_length',
                                           truncation=True,
                                           return_tensors='pt')
        response_encoding = self.tokenizer(
            f'{NEW_SPEICAL_TOKENS_MAP["response"]} {response}',
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt')

        batch = {k: v for k, v in dialogue_encoding.items()}
        batch['decoder_input_ids'] = response_encoding.input_ids
        batch['decoder_attention_mask'] = response_encoding.attention_mask
        if 'is_true_response' in data:
            batch['multiple_choice_label'] = torch.tensor(
                [float(data['is_true_response'])], dtype=torch.float)

        return {
            'dialogue_encoder_input': self.graph_dataset[index],
            'generator_input': batch,
        }
