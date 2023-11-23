import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch
from torch import tensor
import numpy as np
# import tqdm
# import os
# import time
# from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

from text import text_to_sequence


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


# class BufferDataset(Dataset):
#     def __init__(self, buffer):
#         self.buffer = buffer
#         self.length_dataset = len(self.buffer)

#     def __len__(self):
#         return self.length_dataset

#     def __getitem__(self, idx):
#         return self.buffer[idx]


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    energies = [batch[ind]["energy"] for ind in cut_list]
    pitches = [batch[ind]["pitch"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    energies = pad_1D_tensor(energies)
    pitches = pad_1D_tensor(pitches)
    mel_targets = pad_2D_tensor(mel_targets)

    out = {"src_seq": texts,
           "mel_target": mel_targets,
           "dur_target": durations,
           "energy_target": energies,
           "pitch_target": pitches,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_length": max_mel_len}

    return out


def collate_fn(batch: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    len_arr = np.array([d["text"].size(0) for d in batch])
    batch_expand_size = batch[0]["batch_expand_size"]
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // batch_expand_size

    cut_list = list()
    for i in range(batch_expand_size):
        cut_list.append(index_arr[i * real_batchsize: (i + 1) * real_batchsize])

    output = list()
    for i in range(batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output


# def collate_fn(dataset_items: List[dict]):
#     """
#     Collate and pad fields in dataset items
#     """

#     result_batch = {}
#     result_batch["mixed"] = pad_sequence([rec["mixed"].squeeze(0) for rec in dataset_items], True, 0)
#     result_batch["ref"] = pad_sequence([rec["ref"].squeeze(0) for rec in dataset_items], True, 0)
#     result_batch["target"] = pad_sequence([rec["target"].squeeze(0) for rec in dataset_items], True, 0)

#     result_batch["target"] = pad(result_batch["target"], (0, result_batch["mixed"].size(-1) - result_batch["target"].size(-1)))

#     result_batch["mixed_lens"] = tensor([rec["mixed"].size(-1) for rec in dataset_items])
#     result_batch["ref_lens"] = tensor([rec["ref"].size(-1) for rec in dataset_items])
#     result_batch["speakers_ids"] = tensor([rec["speakers_ids"] for rec in dataset_items])
#     return result_batch