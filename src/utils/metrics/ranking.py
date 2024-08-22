import torch
from sklearn.metrics import roc_auc_score


def compute_mrr(preds: torch.Tensor, targets: torch.Tensor):
    # Get the indices of the top k scores
    _, indices = torch.sort(preds, descending=True)

    mrr_sum = 0.0
    count = 0
    for i in range(len(targets)):
        # Get the indices where the target is 1
        target_indices = targets[i].nonzero(as_tuple=False)
        if len(target_indices) == 0:
            continue

        rank = torch.tensor(
            [indices[i].tolist().index(j) for j in target_indices.unsqueeze(1)],
            dtype=torch.float)

        if len(rank) == 0:
            continue

        reciprocal_rank = 1.0 / rank
        mrr_sum += reciprocal_rank.sum().item()  # 這邊會有問題，輸出inf
        count += len(rank)

    mrr = mrr_sum / count
    return mrr


def compute_hits_at_k(preds: torch.Tensor,
                      targets: torch.Tensor,
                      k: int = 5) -> float:

    # Get the indices of the top k scores
    top_k_preds = preds.topk(k, dim=1).indices

    # Calculate hits by checking the intersection of top k predictions and target indices
    hits = []
    for i in range(len(targets)):
        # Get the indices where the target is 1
        target_indices = targets[i].nonzero(as_tuple=False)
        if len(target_indices) == 0:
            continue

        target_indices = target_indices[:k].squeeze(1)

        hit_counts = [1 for idx in target_indices if idx in top_k_preds[i]]
        hits.append(sum(hit_counts) / len(target_indices))

    # Calculate the average hits across all samples
    average_hits = sum(hits) / len(hits)
    return average_hits


def compute_roc_auc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.detach().cpu().numpy()
    targets = targets.cpu().numpy()

    return roc_auc_score(targets, preds)
