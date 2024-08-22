import glob
import json
import os
import pickle
from typing import Union

import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from easydict import EasyDict


def load_json(file_path: str) -> Union[dict, list]:
    with open(file_path) as f:
        return json.load(f)


def load_jsonl(file_path: str) -> list:
    with open(file_path) as f:
        return [json.loads(line) for line in f]


def load_pickle(file_path: str) -> object:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(file_path: str, obj: object) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def save_yaml(file_path: str, obj: object) -> None:
    with open(file_path, 'w') as f:
        yaml.dump(obj, f)


def find_and_load_model(backbone, checkpoint_dir: os.PathLike) -> nn.Module:
    # find the model with .state_dict and load the model
    model_file = glob.glob(os.path.join(checkpoint_dir, '*.state_dict'))[0]
    backbone.load_state_dict(torch.load(model_file))
    return backbone


def save_model(model: nn.Module, path: os.PathLike) -> None:
    torch.save(model, path)


def save_dist_model(accelerator: Accelerator, model: nn.Module,
                    path: os.PathLike):
    state = accelerator.get_state_dict(model)
    accelerator.save(state, path)


def convert_to_dict(d: Union[EasyDict, list, dict]) -> dict:
    if isinstance(d, EasyDict):
        return {k: convert_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_to_dict(i) for i in d]
    else:
        return d
