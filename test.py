"""
Author: Lynn Ye
Created on: 2025/4/25
Brief: functions for testing the performance of continuous MIDI model
"""
import torch
import math

from src.model.bert import mask_input
from src.util.metrics import hits_at_k, accuracy_within_n


def test_mlm(model, perf_config, test_dataloader, metrics=None, device=torch.device('mps')):
    if metrics is None:
        metrics = [hits_at_k, accuracy_within_n]

    test_metrics = {metric.__name__: [] for metric in metrics}
    tokenizer = perf_config.encoder_decoder
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            input_ids, attention_mask, token_types = batch
            masked_input, labels = mask_input(input_ids, token_types)
            output_dict = model(input_ids, token_types, attention_mask)
            cls_logits = output_dict['note']
            vel_preds = output_dict['velocity']
            ts_preds = output_dict['timeshift']
            # Mask tokens
            inputs, labels, mask_type_tensor = mask_perf_tokens(input_ids, perf_config=perf_config, mask_prob=0.15,
                                                                # normal_mask_ratio=.3,
                                                                special_ids=(note_seq.PerformanceEvent.VELOCITY,))
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss, logits = model(inputs, attention_mask=attention_mask, labels=labels)
            total_loss += loss.item()
            count += 1

            # Calculate and store each metric
            for metric in metrics:
                metric_value = metric(logits, labels, masks=mask_type_tensor)
                if not math.isnan(metric_value):
                    test_metrics[metric.__name__].append(metric_value)

            # if step == len(test_dataloader) - 1 or not NDEBUG:
            #     decoded = [tokenizer.class_index_to_event(i, None) for i in inputs[0, :].tolist()]
            #     print("Input ids:")
            #     print_perf_seq(decoded)
            #     print("Decoded:")
            #     print_perf_seq(decode_batch_perf_logits(logits, tokenizer._one_hot_encoding, idx=0))

            if not NDEBUG:
                break

    avg_test_loss = total_loss / count
    # print(f'Test Loss: {avg_test_loss}')

    # Average out metrics over all batches
    avg_test_metrics = {metric: sum(values) / len(values) for metric, values in test_metrics.items()}
    # for metric_name, avg_value in avg_test_metrics.items():
    #     print(f'{metric_name}: {avg_value}')

    # Save the checkpoint
    return avg_test_loss, avg_test_metrics


def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
