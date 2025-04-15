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

from src.util.definitions import TS_TYPE, VEL_TYPE, NOTE_TYPE, DUM_TYPE
from src.util.magenta.models.performance_rnn import performance_model
from src.util.magenta.pipelines import performance_pipeline


def get_noteseq(filepath):
    return note_seq.midi_io.midi_to_note_sequence(pm.PrettyMIDI(filepath))


def translate_perf_data(seq_example):
    feature_lists = seq_example.feature_lists
    result = {}
    for key, feature_list in feature_lists.feature_list.items():
        # Extract values from the feature list
        values = []
        for feature in feature_list.feature:
            # Assuming the features are float lists (you can adapt this to your needs)
            if feature.HasField('float_list'):
                values.append(feature.float_list.value)
            elif feature.HasField('int64_list'):
                values.append(feature.int64_list.value[0])
            elif feature.HasField('bytes_list'):
                values.append(feature.bytes_list.value[0])
            else:
                raise ValueError("Unsupported feature type")
        result[key] = values
    return result


def onehot(label, nclass):
    one_hot = [0.0] * nclass
    one_hot[label] = 1.0
    return one_hot


def cleanup_perf_dict(perf_dict):
    """
    Because magenta make 128 into [0~127, 1~128] for input/label pairs
    We manually add the last token to inputs for Bert MLM training

    [WARNING] removes key: labels in place

    :param perf_dict:
    :return:
    """
    perf_dict['inputs'].append(onehot(perf_dict['labels'][-1], len(perf_dict['inputs'][0])))
    if 'labels' in perf_dict:
        del perf_dict['labels']  # We don't use the labels


def extract_tokens_from_midi(filepath, config=None, max_seq=256, min_seq=32):
    """
    Extract tokens for Bert MLM training
    :param filepath:
    :param config:  performance_model.default_configs
    :param max_seq:
    :param min_seq:
    :return:
    """
    if config is None:
        config = performance_model.default_configs['performance']

    seq = get_noteseq(filepath=filepath)
    pipeline_inst = performance_pipeline.get_pipeline(
        min_events=min_seq,
        max_events=max_seq,  # Magenta will reduce input by 1 element and shift label by 1
        eval_ratio=0.0,
        config=config)
    result = pipeline_inst.transform(seq)
    assert len(result['training_performances']) > 0
    token_list = []
    for e in result['training_performances']:
        perf_data = translate_perf_data(e)
        cleanup_perf_dict(perf_data)  # Necessary
        token_list.append(perf_data)
    return token_list


def get_attention_mask(tokens, max_seq_len):
    """
    Return array of ones and zeros (masks)
    :param tokens:
    :param max_seq_len:
    :return:
    """
    if len(tokens) > max_seq_len:
        raise ValueError("Token seq length exceeds max_seq_len")
    if len(tokens) == max_seq_len:
        return [1] * max_seq_len
    mask = [1] * len(tokens)
    mask.extend([0] * (max_seq_len - len(tokens)))
    return mask


def convert_onehot_to_ids(onehot):
    """
    Input single onehot vector
    :param onehot:
    :return:
    """
    assert isinstance(onehot[0], (int, float))
    return np.argmax(onehot)


def pad_onehot_perf_tokens(token_seq, perf_config, max_seq_len=128):
    """
    Return lists of "labels" padded to max_seq_len
    Make a copy of the ids, and pad token seq to max seq len
    :param token_seq: dict -> { 'inputs': [onehot encodings], 'labels': [list of class idx] }
    :param perf_config:
    :param max_seq_len: default events will be padded to the end till max seq len is reached
    :return:
    """
    if perf_config is None:
        raise ValueError("Config is required to use the tokenizer!")
    pad_token = perf_config.encoder_decoder.default_event_label
    # pad_onehot = onehot(pad_token, perf_config.encoder_decoder.num_classes)
    assert isinstance(token_seq, (list, torch.tensor))
    if len(token_seq) > max_seq_len:
        raise ValueError("Token seq length exceeds max_seq_len")
    if len(token_seq) == max_seq_len:
        return [convert_onehot_to_ids(e) for e in token_seq]

    # token_seq['labels'].append([pad_token] * (max_seq_len - len(token_seq)))
    out = [convert_onehot_to_ids(e) for e in token_seq]
    out.extend([pad_token] * (max_seq_len - len(out)))
    assert len(out) == max_seq_len
    return out


def decode_output_ids(out_ids, decoder):
    """
    Decode class id into performance events
    :param out_ids:
    :param decoder:
    :return:
    """
    if isinstance(out_ids, torch.Tensor):
        return [decoder.decode_event(e.item()) for e in out_ids]
    else:
        assert isinstance(out_ids, list)
        return [decoder.decode_event(e) for e in out_ids]


def prepare_token_ids(midi_path, perf_config=None, min_seq_len=64, max_seq_len=128):
    """
    Extract sequences of input ids from midi, pad with default events after the end
    Return shape:
     - [[input ids], [...]]
     - [[attention masks], [...]]

    Token Encoding:
    1: NOTEON
    2: NOTEOFF
    3: TIMESHIFT
    4: VELOCITY
    5: (DURATION) (used only in note-based encoding)

    Decoding:
    for range [1~5]
    - if < MIN~MAX:
        - event type = curr
        - event id = get(value)
    - else:
        - offset += RANGE
        - event type ++

    :param midi_path:
    :param perf_config:
    :param max_seq_len:
    :return:
    """
    if perf_config is None:
        perf_config = performance_model.default_configs['performance']
        raise ValueError("Config is required to use the tokenizer!")

    tokens = extract_tokens_from_midi(midi_path, config=perf_config, min_seq=min_seq_len, max_seq=max_seq_len)
    assert len(tokens) > 0
    attention_masks = []
    out_tokens = []
    for token_dict in tokens:
        attention_masks.append(get_attention_mask(token_dict['inputs'], max_seq_len=max_seq_len))
        out_tokens.append(pad_onehot_perf_tokens(token_dict['inputs'], perf_config, max_seq_len=max_seq_len))
    return torch.tensor(out_tokens), torch.tensor(attention_masks)


def is_continuous_token(performance_event: note_seq.PerformanceEvent):
    return (performance_event.event_type == note_seq.PerformanceEvent.TIME_SHIFT
            or performance_event.event_type == note_seq.PerformanceEvent.VELOCITY)


def get_token_type(performance_event: note_seq.PerformanceEvent):
    if performance_event.event_type == note_seq.PerformanceEvent.TIME_SHIFT:
        return TS_TYPE
    elif performance_event.event_type == note_seq.PerformanceEvent.VELOCITY:
        return VEL_TYPE
    else:
        return NOTE_TYPE


def convert_mixed_tokens_to_ids(tokenseq_list, mask_list, type_list, max_seq_len, encoder_decoder):
    def fill_tensor(value, tensor_len):
        return torch.cat([torch.tensor([value], dtype=torch.float32),
                          torch.zeros(tensor_len - 1)])

    def list2d_to_tensor(list2d):
        return torch.stack([torch.stack(ll) for ll in list2d])

    input_ids_lists = []
    out_masks = []
    out_types = []

    # Pad right and convert to tensor
    ids_pad = torch.tensor(encoder_decoder.default_event_label)
    for i_token_list in range(len(tokenseq_list)):
        # Convert tokens to ids
        tk_ids = [torch.tensor(encoder_decoder.events_to_label(tokenseq_list[i_token_list], itk))
                  if not is_continuous_token(tokenseq_list[i_token_list][itk]) else
                  torch.tensor(tokenseq_list[i_token_list][itk].event_value)
                  for itk in range(len(tokenseq_list[i_token_list]))]

        n_pad = max_seq_len - len(tokenseq_list[i_token_list])

        if n_pad < 0:
            raise ValueError("Token list exceeds max_seq_len")
        if n_pad > 0:
            input_ids_lists += [tk_ids + [ids_pad] * n_pad]  # Dummy token
        else:
            input_ids_lists += [tk_ids]
        out_masks += [mask_list[i_token_list] + [0] * n_pad]
        out_types += [type_list[i_token_list] + [DUM_TYPE] * n_pad]  # Dummy type

    return torch.tensor(input_ids_lists), torch.tensor(out_masks), torch.tensor(out_types)


def convert_to_mixed_type_tokens(token_ids, mask, encoder_decoder):
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

    merged_token_list = []
    merged_mask_list = []
    token_type_list = []

    curr_timeshift_value = -1
    for i in range(token_ids.shape[0]):
        tokens = decode_output_ids(token_ids[i], encoder_decoder._one_hot_encoding)
        merged_tokens = []
        merged_masks = []
        token_types = []
        for i_token, token in enumerate(tokens):
            if token.event_type == note_seq.PerformanceEvent.TIME_SHIFT:
                if curr_timeshift_value < 0:
                    # Start of ts token
                    curr_timeshift_value = token.event_value
                    continue
                else:
                    curr_timeshift_value += token.event_value
            else:
                if curr_timeshift_value >= 0:
                    merged_tokens.append(note_seq.PerformanceEvent(event_type=3, event_value=curr_timeshift_value))
                    merged_masks.append(1)
                    token_types.append(TS_TYPE)
                    curr_timeshift_value = -1
                merged_tokens.append(token)
                merged_masks.append(mask[i][i_token].item())  # Mask assumed to be of type tensor
                token_types.append(get_token_type(performance_event=token))
            pass
        if curr_timeshift_value >= 0:
            # Final pass
            merged_tokens.append(note_seq.PerformanceEvent(event_type=3, event_value=curr_timeshift_value))
            merged_masks.append(1)
            token_types.append(TS_TYPE)
            curr_timeshift_value = -1
        assert len(merged_tokens) == len(merged_masks) == len(token_types)

        # Pad the merged tokens to the given seq length

        merged_token_list.append(merged_tokens)
        merged_mask_list.append(merged_masks)
        token_type_list.append(token_types)

    # Convert to tensor
    return merged_token_list, merged_mask_list, token_type_list


def create_input_from_midi(file_path, config, max_seq_len=128):
    encoder_decoder = config.encoder_decoder
    token_ids, mask = prepare_token_ids(file_path, perf_config=config, max_seq_len=max_seq_len)
    tokens, masks, types = convert_to_mixed_type_tokens(token_ids, mask, encoder_decoder)
    return convert_mixed_tokens_to_ids(tokens, masks, types,
                                       max_seq_len=token_ids.shape[1],
                                       encoder_decoder=encoder_decoder)


def collect_files(data_dir, key=None, negative_key=None):
    if key is None:
        key = []
    if negative_key is None:
        negative_key = []
    out_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_dir, file_name = os.path.split(file_path)
            if all(k in file_name for k in key) and all(nk not in file_name for nk in negative_key):
                out_paths.append(file_path)
    return out_paths


class MixTokenDataset(Dataset):
    def __init__(self, dpath: str):
        self.samples = torch.load(dpath)

    def __len__(self):
        return len(self.samples['input_ids'])

    def __getitem__(self, idx):
        """
        Return input_ids, masks, types
        :param idx:
        :return:
        """
        return self.samples['input_ids'][idx], self.samples['masks'][idx], self.samples['types'][idx]


def create_dataset(midi_dir, save_dir=None, split_ratios=(0.8, 0.1, 0.1), seed=0,
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
    # TODO @Bmois check whether it decodes velocity tokens

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
                token_ids, masks, types = create_input_from_midi(midi_file, config, max_seq_len=max_seq_len)
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
    config = performance_model.default_configs['performance']
    token_ids, masks, types = create_input_from_midi('debug_files/long_rest.mid', config)
    print("No. of items:", token_ids.shape[0])
    print("Seq len:", token_ids.shape[1])
    print("Class size:", token_ids.shape[2])
    pass


if __name__ == "__main__":
    ''''''
    data_path = '/Users/kurono/Documents/github/ContinuousMIDI/tmp/maestro-v3.0.0'
    save_path = '/Users/kurono/Documents/github/ContinuousMIDI/tmp/maestro_data'
    create_dataset(data_path, save_dir=save_path, config_name='performance_with_dynamics')
    # ftest_create_input()
    # ftest_merge_timeshifts()
    # ftest_decode_ids()
