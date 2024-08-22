from enum import Enum
from typing import Optional

import torch

from src.data.dataset import DialogueGraphData
from src.utils.constants import COHERENCE_RELATIONS


class DialogueGraphDataProcessMode(Enum):
    """`DialogueGraphData` process mode.

    NONE: do not apply any filter
    SINGLE_FILTER: filter out the edges with target_labels if the edge has only one label (coherence relations)
    SINGLE_RANDOM_FILTER: filter out the edges with target_labels if the edge has only one label with 60% probability
    """

    NONE = 'none'
    SINGLE_FILTER = 'single_filter'
    SINGLE_RANDOM_FILTER = 'single_random_filter'


class ConvAI2DataProcessor:
    """Preprocess the `ConvAI2ForDialogueGraphEncodingDataset`.
    pre_filter and pre_transform is called after 'process', and transform is called after 'get'.
    """

    def __init__(self,
                 mode: DialogueGraphDataProcessMode,
                 k_hop: int,
                 filter_labels: Optional[list[str]] = ['Topic Shift']):
        self.mode = mode
        self.k_hop = k_hop
        self.FILTERED_LABELS = filter_labels

        assert all(item in COHERENCE_RELATIONS for item in self.FILTERED_LABELS), \
            f'Invalid filtered labels: {self.FILTERED_LABELS}'

    def pre_filter(self, data: DialogueGraphData) -> bool:
        return True

    def pre_transform(self, data: DialogueGraphData) -> DialogueGraphData:
        # only keep the neighbors within k_hop
        processed_data = self.filter_k_hop(data)

        return processed_data

    def transform(self, data: DialogueGraphData) -> DialogueGraphData:
        if self.mode == DialogueGraphDataProcessMode.NONE:
            return data

        processed_data = self.filter_label(data)
        return processed_data

    def filter_k_hop(self, data: DialogueGraphData) -> DialogueGraphData:
        processed_data = data.clone()
        processed_edge_index = [[], []]
        processed_edge_labels = []
        processed_edge_features = []
        for i, (edge_from_index, edge_to_index) in enumerate(
                zip(data.edge_index[0], data.edge_index[1])):
            # filter out edges with hops greater than k_hop
            # only keep the edges (context) with hops <= k_hop
            if abs(edge_to_index - edge_from_index) > self.k_hop:
                continue

            processed_edge_index[0].append(edge_from_index)
            processed_edge_index[1].append(edge_to_index)
            processed_edge_labels.append(data.y[i].tolist())
            processed_edge_features.append(data.edge_attr[i].tolist())

        processed_data.edge_index = torch.tensor(processed_edge_index,
                                                 dtype=torch.long)
        processed_data.y = torch.tensor(processed_edge_labels,
                                        dtype=torch.float)
        processed_data.edge_attr = torch.tensor(processed_edge_features,
                                                dtype=torch.float)

        return processed_data

    def filter_label(self, data: DialogueGraphData) -> DialogueGraphData:
        processed_data = data.clone()
        processed_edge_index = [[], []]
        processed_edge_labels = []
        processed_edge_features = []
        for i, (edge_from_index, edge_to_index) in enumerate(
                zip(data.edge_index[0], data.edge_index[1])):
            edge_labels = data.y[i]
            coh_rel_label_index = [
                i for i, label in enumerate(edge_labels) if label == 1
            ]

            if self.mode == DialogueGraphDataProcessMode.SINGLE_FILTER:
                # filter out edges with target_labels if the edge has only one label
                if len(coh_rel_label_index) == 1 and COHERENCE_RELATIONS[
                        coh_rel_label_index[0]] in self.FILTERED_LABELS:
                    continue
            elif self.mode == DialogueGraphDataProcessMode.SINGLE_RANDOM_FILTER:
                # filter out edges with target_labels if the edge has only one label with 70% probability
                if len(coh_rel_label_index) == 1 and COHERENCE_RELATIONS[
                        coh_rel_label_index[0]] in self.FILTERED_LABELS:
                    if torch.rand(1) > 0.4:
                        continue
            else:
                raise ValueError(f'Invalid mode: {self.mode}')

            processed_edge_index[0].append(edge_from_index)
            processed_edge_index[1].append(edge_to_index)
            processed_edge_labels.append(edge_labels.tolist())
            processed_edge_features.append(data.edge_attr[i].tolist())

        processed_data.edge_index = torch.tensor(processed_edge_index,
                                                 dtype=torch.long)
        processed_data.y = torch.tensor(processed_edge_labels,
                                        dtype=torch.float)
        processed_data.edge_attr = torch.tensor(processed_edge_features,
                                                dtype=torch.float)

        return processed_data
