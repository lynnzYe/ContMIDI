"""
Author: Lynn Ye
Created on: 2025/4/25
Brief:
"""
import torch
import functools


def cross_entropy_loss(logits, labels, masks=None, consider_mask=None):
    if masks is None:
        raise ValueError("Masks are needed to evaluate MLM, else all tokens will be considered!")
    if consider_mask is None:
        valid_mask = torch.greater(masks, 0)
    else:
        assert isinstance(consider_mask, torch.Tensor)
        valid_mask = torch.isin(masks, consider_mask)

    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]
    if valid_logits.numel() == 0:
        return 0  # No valid elements to calculate loss
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(valid_logits.view(-1, valid_logits.size(-1)), valid_labels.view(-1)).item()


def hits_at_k(logits, labels, k=3, masks=None, consider_mask=None):
    """
    HITS@K metric
    :param logits:
    :param labels:
    :param k:
    :param masks:
    :param consider_mask:
    :return:
    """
    if masks is None:
        raise ValueError("Masks are needed to evaluate MLM, else all tokens will be considered!")
    if consider_mask is None:
        valid_mask = torch.greater(masks, 0)
    else:
        assert isinstance(consider_mask, torch.Tensor)
        valid_mask = torch.isin(masks, consider_mask)

    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]

    if valid_labels.numel() == 0:
        return float('nan')

    _, valid_topk_indices = torch.topk(valid_logits, k, dim=-1)
    # .mean() effectively calculates the binary hit/miss values
    hits = (valid_topk_indices == valid_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return hits


def accuracy_within_n(logits, labels, n=1, masks=None, consider_mask=None):
    """
    Calculate accuracy ± n for ordinal classification
    :param logits:
    :param labels:
    :param n:
    :param masks:
    :param consider_mask:
    :return:
    """
    if masks is None:
        raise ValueError("Masks are needed to evaluate MLM, else all tokens will be considered!")
    if consider_mask is None:
        valid_mask = torch.greater(masks, 0)
    else:
        assert isinstance(consider_mask, torch.Tensor)
        valid_mask = torch.isin(masks, consider_mask)

    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]

    if valid_labels.numel() == 0:
        return float('nan')
    valid_pred = valid_logits.argmax(dim=-1)
    acc = (torch.abs(valid_pred - valid_labels) <= n).float().mean().item()
    return acc


def bind_metric(func, **kwargs):
    partial_func = functools.partial(func, **kwargs)
    arg_str = "_".join(f"{key}{value}" for key, value in kwargs.items())
    partial_func.__name__ = f"{func.__name__}_{arg_str}"
    return partial_func


def get_mlm_metrics(model_config_path):
    hits_1_all, hits_2_all, hits_3_all = (bind_metric(hits_at_k, k=1, consider_mask=torch.tensor([1, 2])),
                                          bind_metric(hits_at_k, k=2, consider_mask=torch.tensor([1, 2])),
                                          bind_metric(hits_at_k, k=3, consider_mask=torch.tensor([1, 2])))
    hits_1, hits_2, hits_3 = (bind_metric(hits_at_k, k=1, consider_mask=torch.tensor([1])),
                              bind_metric(hits_at_k, k=2, consider_mask=torch.tensor([1])),
                              bind_metric(hits_at_k, k=3, consider_mask=torch.tensor([1])))
    acc_w_1_all, acc_w_2_all, acc_w_3_all = (bind_metric(accuracy_within_n, n=1, consider_mask=torch.tensor([1, 2])),
                                             bind_metric(accuracy_within_n, n=2, consider_mask=torch.tensor([1, 2])),
                                             bind_metric(accuracy_within_n, n=3, consider_mask=torch.tensor([1, 2])))
    acc_w_1, acc_w_2, acc_w_3 = (bind_metric(accuracy_within_n, n=1, consider_mask=torch.tensor([1])),
                                 bind_metric(accuracy_within_n, n=2, consider_mask=torch.tensor([1])),
                                 bind_metric(accuracy_within_n, n=3, consider_mask=torch.tensor([1])))

    return [hits_1, hits_2, hits_3, hits_1_all, hits_2_all, hits_3_all, acc_w_1, acc_w_2, acc_w_3, acc_w_1_all,
            acc_w_2_all, acc_w_3_all]
