import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_contrastive_loss(
    embeddings,
    anchor_indices,
    pos_indices,
    neg_indices,
    similarity_func=nn.CosineSimilarity(dim=-1),
    contrastive_learning_loss_func=nn.CrossEntropyLoss()):
    anchor_embeddings = embeddings[anchor_indices].unsqueeze(0)
    pos_embeddings = embeddings[pos_indices].unsqueeze(0)
    neg_embeddings = embeddings[neg_indices].unsqueeze(0)

    pos_sim = similarity_func(anchor_embeddings, pos_embeddings)
    neg_sim = similarity_func(anchor_embeddings, neg_embeddings)

    # 將正負範例堆疊，並擴展維度以適應CrossEntropyLoss
    sims = torch.cat([neg_sim, pos_sim], dim=1)

    # 創建標籤，正範例的索引為1
    labels = torch.ones(sims.size(0), dtype=torch.long, device=sims.device)

    return contrastive_learning_loss_func(sims, labels)


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits,
                                                target=labels,
                                                reduction='none')

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits -
                              gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def compute_loss_with_class_balanced(labels,
                                     logits,
                                     samples_per_cls,
                                     no_of_classes,
                                     loss_type,
                                     beta,
                                     gamma,
                                     is_onehot=False):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    if not is_onehot:
        labels_one_hot = F.one_hot(labels, no_of_classes).float()
    else:
        labels_one_hot = labels.float()  # 傳入的labels已經是one-hot形式

    weights = torch.tensor(weights, device=labels_one_hot.device).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == 'focal':
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == 'sigmoid':
        cb_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels_one_hot,
                                                     weight=weights)
    elif loss_type == 'softmax':
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred,
                                         target=labels_one_hot,
                                         weight=weights)
    return cb_loss


if __name__ == '__main__':
    no_of_classes = 5
    logits = torch.rand(10, no_of_classes).float()
    labels = torch.randint(0, no_of_classes, size=(10, ))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2, 3, 1, 2, 2]
    loss_type = 'focal'
    cb_loss = compute_loss_with_class_balanced(labels, logits, samples_per_cls,
                                               no_of_classes, loss_type, beta,
                                               gamma)
    print(cb_loss)
