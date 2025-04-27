"""
Author: Lynn Ye
Created on: 2025/4/25
Brief: functions for testing the performance of continuous MIDI model
"""
import torch
import math

from src.model.bert import mask_input
from src.util.definitions import NDEBUG
from src.util.metrics import hits_at_k, accuracy_within_n
from src.util.magenta.models.performance_rnn import performance_model


def test_mlm(model, test_dataloader, note_loss_fn, timeshift_loss_fn, velocity_loss_fn, loss_weighting_fn,
             metrics=None, config_name='performance_with_dynamics'):
    if metrics is None:
        metrics = [hits_at_k, accuracy_within_n]

    test_metrics = {metric.__name__: [] for metric in metrics}
    model.eval()
    total_loss = 0
    count = 0

    token_config = performance_model.default_configs[config_name]
    tokenizer = token_config.encoder_decoder

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            input_ids, attention_mask, token_types = batch
            masked_input, labels = mask_input(input_ids, token_types)
            output_dict = model(input_ids, token_types, attention_mask)
            cls_logits = output_dict['note']
            vel_preds = output_dict['velocity']
            ts_preds = output_dict['timeshift']

            note_loss = note_loss_fn(cls_logits, token_types, labels)
            velocity_loss = velocity_loss_fn(vel_preds, token_types, labels)
            timeshift_loss = timeshift_loss_fn(ts_preds, token_types, labels)

            # Calculate the total loss
            weighted_loss = loss_weighting_fn(note=note_loss, vel=velocity_loss, ts=timeshift_loss)

            total_loss += weighted_loss
            count += 1

            # Calculate and store each metric
            for metric in metrics:
                metric_value = metric(cls_logits, token_types, labels)
                if not math.isnan(metric_value):
                    test_metrics[metric.__name__].append(metric_value)

            # if step == len(test_dataloader) - 1 or not NDEBUG:
            # TODO @Bmois add decode function (return tokens, and tokens to MIDI func)

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


# TODO @Bmois add interactive demo (given a MIDI file, generate a new one)


def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
