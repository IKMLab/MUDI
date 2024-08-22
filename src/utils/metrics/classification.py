from typing import Literal

import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_kl_divergence(posterior_mu: torch.Tensor,
                          posterior_logvar: torch.Tensor,
                          prior_mu: torch.Tensor,
                          prior_logvar: torch.Tensor) -> torch.Tensor:
    kl_div = 0.5 * torch.sum(
        prior_logvar - posterior_logvar - 1 +
        posterior_logvar.exp() / prior_logvar.exp() +
        (prior_mu - posterior_mu).pow(2) / prior_logvar.exp(), 1)

    return kl_div


def compute_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    cls_type: Literal['binary', 'multiclass',
                      'multilabel'] = 'binary') -> float:
    if cls_type == 'multiclass':
        return accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())

    preds = preds > 0.5
    correct = (preds == targets).float()
    accuracy = correct.sum() / (targets.size(0) * targets.size(1))

    return accuracy.item()


def compute_f1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    average='micro',
    cls_type: Literal['binary', 'multiclass',
                      'multilabel'] = 'binary') -> float:
    if cls_type == 'multiclass':
        return f1_score(targets.cpu().numpy(),
                        preds.cpu().numpy(),
                        average=average)

    preds = (preds > 0.5).cpu().numpy()
    targets = targets.cpu().numpy()
    f1 = f1_score(targets, preds, average=average)

    return f1
