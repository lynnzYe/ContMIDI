"""
Author: Lynn Ye
Created on: 2025/4/25
Brief:
"""
import torch
import functools

from src.util.definitions import IGNORE_LABEL_INDEX, NOTE_TYPE, TS_TYPE


def obtain_target_input(all_logits: torch.Tensor, token_types: torch.Tensor, labels: torch.Tensor, tgt_type):
    # Mask out the continuous tokens
    note_mask = (token_types != tgt_type)
    if not note_mask.any():
        return torch.tensor(0.0, device=all_logits.device)
    logits = all_logits[note_mask]
    labels = labels[note_mask]

    # Consider only masked input (label != IGNORE_LABEL_INDEX)
    valid_mask = (labels != IGNORE_LABEL_INDEX)
    logits = logits[valid_mask]
    labels = labels[valid_mask]
    return logits, labels


def discrete_metric_template(all_logits: torch.Tensor, token_types: torch.Tensor, labels: torch.Tensor):
    """
    Metric template for discrete token classification
    :param all_logits: mixed token types (both discrete tokens & continuous tokens, the direct model output)
    :param token_types:
    :param labels: input ids (ignore label for all unmasked input)
    :return:
    """
    # Mask out the continuous tokens
    logits, labels = obtain_target_input(all_logits, token_types, labels, tgt_type=NOTE_TYPE)

    # Calculate the metric
    if logits.numel() == 0:
        return float('nan')
    valid_pred = logits.argmax(dim=-1)
    acc = (valid_pred == labels).float().mean().item()
    return acc


def continuous_metric_template(all_preds, token_types, labels, tgt_type=TS_TYPE):
    """
    Metric template for regressions
    :param all_preds: numeric values (regression output)
    :param token_types:
    :param labels: input ids (ignore label for all unmasked input)
    :param tgt_type: which type of continuous token?
    :return:
    """
    preds, labels = obtain_target_input(all_preds, token_types, labels, tgt_type=tgt_type)

    # Calculate the metric (we use L1 loss for regression)
    if preds.numel() == 0:
        return float('nan')
    acc = (torch.abs(preds - labels) <= 1).float().mean().item()
    return acc


def hits_at_k(all_logits: torch.Tensor, token_types: torch.Tensor, labels: torch.Tensor, top_k=3):
    """
    Metric template for discrete token classification
    :param all_logits: mixed token types (both discrete tokens & continuous tokens, the direct model output)
    :param token_types:
    :param labels: input ids (ignore label for all unmasked input)
    :param top_k:
    :return:
    """
    logits, labels = obtain_target_input(all_logits, token_types, labels, tgt_type=NOTE_TYPE)

    _, valid_topk_indices = torch.topk(logits, top_k, dim=-1)
    # .mean() effectively calculates the binary hit/miss values
    hits = (valid_topk_indices == logits.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return hits


def accuracy_within_n(all_logits: torch.Tensor, token_types: torch.Tensor, labels: torch.Tensor, wt_n=2):
    """
    Metric template for discrete token ordinal classification (assumes classes are arranged in increasing continuous order)
    :param all_logits: mixed token types (both discrete tokens & continuous tokens, the direct model output)
    :param token_types:
    :param labels: input ids (ignore label for all unmasked input)
    :param wt_n: within n classes (important in ordinal classification)
    :return:
    """
    logits, labels = obtain_target_input(all_logits, token_types, labels, tgt_type=NOTE_TYPE)

    if logits.numel() == 0:
        return float('nan')
    valid_pred = logits.argmax(dim=-1)
    acc = (torch.abs(valid_pred - labels) <= wt_n).float().mean().item()
    return acc


def l1_loss(all_preds, token_types, labels, tgt_type=TS_TYPE):
    """
    L1 loss for continuous predictions
    :param all_preds: numeric values (regression output)
    :param token_types:
    :param labels: input ids (ignore label for all unmasked input)
    :param tgt_type: which type of continuous token?
    :return:
    """
    preds, labels = obtain_target_input(all_preds, token_types, labels, tgt_type=tgt_type)

    # Calculate the metric (we use L1 loss for regression)
    if preds.numel() == 0:
        return float('nan')
    acc = torch.abs(preds - labels).mean().item()
    return acc


def percent_loss(all_preds, token_types, labels, token_type=TS_TYPE):
    """
    Deviance from gt represented as percent of GT (+ for overestimation, - for underestimation)
    The closer to 0 the better
    :param all_preds:
    :param token_types:
    :param labels:
    :param token_type:
    :return:
    """
    preds, labels = obtain_target_input(all_preds, token_types, labels, tgt_type=token_type)

    # Calculate the metric (we use L1 loss for regression)
    if preds.numel() == 0:
        return float('nan')
    acc = (preds - labels) / labels
    acc[labels == 0] = 0
    return acc.mean().item(), torch.std(acc).item()


def bind_metric(func, **kwargs):
    partial_func = functools.partial(func, **kwargs)
    arg_str = "_".join(f"{key}{value}" for key, value in kwargs.items())
    partial_func.__name__ = f"{func.__name__}_{arg_str}"
    return partial_func


def get_mlm_metrics(model_config_path):
    hits_1_all, hits_2_all, hits_3_all = (bind_metric(hits_at_k, top_k=1),
                                          bind_metric(hits_at_k, top_k=2),
                                          bind_metric(hits_at_k, top_k=3))

    return [hits_1_all, hits_2_all, hits_3_all]
