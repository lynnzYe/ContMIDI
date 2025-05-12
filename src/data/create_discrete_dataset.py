"""
Author: Lynn Ye
Created on: 2025/4/6
Brief: create discrete/continuous mixed dataset

Data Structure per batch:
    - input_ids
    - masks
    - token_types
        - 'discrete': NOTE_TYPE,
        - 'timeshift': TS_TYPE,
        - 'velocity': VEL_TYPE
"""

import note_seq
import pretty_midi as pm
import numpy as np
import torch
import random
import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import os

from src.data.create_dataset import collect_files, prepare_token_ids, decode_output_ids, get_token_type, MixTokenDataset
from src.util.definitions import TS_TYPE, VEL_TYPE, NOTE_TYPE, DUM_TYPE
from src.util.magenta.models.performance_rnn import performance_model
from src.util.magenta.pipelines import performance_pipeline


def get_token_types(token_ids, mask, encoder_decoder):
    """
    Convert token ids to discrete & continuous token ids. Will merge timeshift events
    :param token_ids:
    :param mask:
    :param encoder_decoder:
    :return:
    """
    assert len(token_ids) == len(mask)
    assert token_ids.ndim == 2
    n_sample, seq_len = token_ids.shape

    token_types = []

    for i in range(token_ids.shape[0]):
        tokens = decode_output_ids(token_ids[i], encoder_decoder._one_hot_encoding)
        type_list = [get_token_type(performance_event=token) for token in tokens]
        token_types.append(torch.tensor(type_list, dtype=torch.int))
        assert len(token_types[-1]) == seq_len
    assert len(token_types) == n_sample
    return torch.stack(token_types, dim=0)


def create_discrete_input_from_midi(file_path, config, max_seq_len=128):
    encoder_decoder = config.encoder_decoder
    token_ids, mask = prepare_token_ids(file_path, perf_config=config, max_seq_len=max_seq_len)
    types = get_token_types(token_ids, mask, encoder_decoder)
    return token_ids, mask, types


def create_discrete_dataset(midi_dir, save_dir=None, split_ratios=(0.8, 0.1, 0.1), seed=0,
                            config_name='performance', max_seq_len=128):
    """
    Create dataset for continuous MIDI training
    :param midi_dir:
    :param save_dir: create directory to store pt data for each sample (will contain many files)
    :param split_ratios: train, val, test
    :param seed: for train test split
    :param config_name:
    :param max_seq_len:
    :return:
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    midi_files = collect_files(midi_dir, key=['.mid'])
    random.seed(seed)
    random.shuffle(midi_files)

    # Split files
    total = len(midi_files)
    n_train = int(split_ratios[0] * total)
    n_val = int(split_ratios[1] * total)

    splits = {
        'val': midi_files[n_train:n_train + n_val],
        'test': midi_files[n_train + n_val:],
        'train': midi_files[:n_train],
    }

    config = performance_model.default_configs[config_name]
    data_info = {
        'tokenizer_type': config_name,
        'vocab_size': config.encoder_decoder.num_classes,  # In this project, velocity & timeshift vocab are useless
        'seq_len': max_seq_len
    }
    torch.save(data_info, save_dir / 'data_info.pt')

    # Process and save
    for split_name, files in splits.items():
        input_ids_list = []
        masks_list = []
        types_list = []

        for midi_file in tqdm.tqdm(files, desc=f"Processing {split_name}"):
            try:
                token_ids, masks, types = create_discrete_input_from_midi(midi_file, config, max_seq_len=max_seq_len)
            except Exception as e:
                print(f"Skipping {midi_file}: {e}")
                continue
            input_ids_list.extend(token_ids)
            masks_list.extend(masks)
            types_list.extend(types)
        if input_ids_list:
            input_ids_tensor = torch.stack(input_ids_list)
            masks_tensor = torch.stack(masks_list)
            types_tensor = torch.stack(types_list)

            split_data = {
                'input_ids': input_ids_tensor,
                'masks': masks_tensor,
                'types': types_tensor,
            }

            data_info[split_name + "_size"] = input_ids_tensor.size(0)
            out_path = save_dir / f"{split_name}.pt"
            torch.save(split_data, out_path)
            print(f"Saved {split_name} set with {input_ids_tensor.size(0)} samples to {out_path}")
        else:
            print(f"No valid samples found for {split_name}. Skipping saving.")


def load_dataset(data_dir):
    train_dataset = MixTokenDataset(os.path.join(data_dir, 'train.pt'))
    val_dataset = MixTokenDataset(os.path.join(data_dir, 'val.pt'))
    test_dataset = MixTokenDataset(os.path.join(data_dir, 'test.pt'))

    return train_dataset, val_dataset, test_dataset


def ftest_decode_ids():
    token_ids, mask = prepare_token_ids('debug_files/horowitz_gb.mid',
                                        perf_config=performance_model.default_configs['performance'])
    decoder = performance_model.default_configs['performance'].encoder_decoder._one_hot_encoding
    tokens = decode_output_ids(token_ids[0], decoder)
    pass


def ftest_create_input():
    config = performance_model.default_configs['performance_with_dynamics']
    token_ids, masks, types = create_discrete_input_from_midi('debug_files/horowitz_gb.mid', config)
    print("No. of items:", token_ids.shape[0])
    print("Seq len:", token_ids.shape[1])
    pass


if __name__ == "__main__":
    ''''''
    data_path = '/Users/kurono/Documents/github/ContinuousMIDI/tmp/maestro-v3.0.0'
    save_path = '/Users/kurono/Documents/github/ContinuousMIDI/tmp/maestro_data_discrete'
    create_discrete_dataset(data_path, save_dir=save_path, config_name='performance_with_dynamics')
    # ftest_create_input()
    # ftest_merge_timeshifts()
    # ftest_decode_ids()
