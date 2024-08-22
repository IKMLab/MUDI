import os

import torch
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data.data import BaseData as GraphBaseData

from src.utils.data_utils import load_pickle


class BaseGraphDataset(GraphDataset):

    def __init__(self,
                 root,
                 raw_file_name,
                 processed_file_dir,
                 processed_file_name=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.raw_file_name = raw_file_name
        self.processed_file_dir = processed_file_dir
        self.processed_file_name = processed_file_name
        self.raw_data = None
        self.dataset = None

        if os.path.exists(self.processed_file_dir):
            self.dataset = torch.load(self.processed_paths[0])

        super().__init__(root, transform, pre_transform, pre_filter)

    def len(self) -> int:
        return len(self.dataset)

    @property
    def raw_file_names(self) -> str:
        return [self.raw_file_name]

    @property
    def processed_file_names(self) -> str:
        if self.processed_file_name:
            return [self.processed_file_name]
        return ['processed_' + self.raw_file_name]

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return self.processed_file_dir

    def process(self) -> None:
        convai2_data = load_pickle(self.raw_paths[0])
        self.raw_data = convai2_data

        convai2_dataset = []
        for item in self.raw_data:
            convai2_dataset.append(self._process_data(item))

        if self.pre_filter is not None:
            convai2_dataset = [
                data for data in convai2_dataset if self.pre_filter(data)
            ]

        if self.pre_transform is not None:
            convai2_dataset = [
                self.pre_transform(data) for data in convai2_dataset
            ]

        self.dataset = convai2_dataset
        torch.save(self.dataset, self.processed_paths[0])

    def _process_data(self, data: dict) -> GraphBaseData:
        raise NotImplementedError

    def get(self, idx: int) -> GraphBaseData:
        if self.dataset is None:
            self.dataset = torch.load(self.processed_paths[0])
        return self.dataset[idx]
