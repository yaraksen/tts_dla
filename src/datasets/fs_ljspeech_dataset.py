import json
import logging
import os
import shutil
from pathlib import Path
import time
import numpy as np

from tqdm import tqdm
from text import text_to_sequence

from torch.utils.data import Dataset
from torch import from_numpy
from src.utils import ROOT_PATH
# from glob import glob
from text import text_to_sequence
# from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt

def get_data_to_buffer(data_path, mel_ground_truth, alignment_path, energy_path, pitch_path, text_cleaners, batch_expand_size: int) -> list:
    buffer = list()
    text = process_text(data_path)

    start = time.perf_counter()
    for i in tqdm(range(len(text))):

        # mel
        mel_gt_name = os.path.join(mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1))
        mel_gt_target = np.load(mel_gt_name)
        
        # duration
        duration = np.load(os.path.join(alignment_path, str(i) + ".npy"))

        # energy
        energy_gt_name = os.path.join(energy_path, "ljspeech-energy-%05d.npy" % (i + 1))
        energy_gt_target = np.load(energy_gt_name)

        # pitch
        pitch_gt_name = os.path.join(pitch_path, "ljspeech-pitch-%05d.npy" % (i + 1))
        pitch_gt_target = np.load(pitch_gt_name)

        # text
        character = np.array(text_to_sequence(text[i][0: len(text[i]) - 1], text_cleaners))

        mel_gt_target = from_numpy(mel_gt_target)
        duration = from_numpy(duration)
        energy = from_numpy(energy_gt_target)
        pitch = from_numpy(pitch_gt_target.astype(np.float32))
        character = from_numpy(character)
        buffer.append({"batch_expand_size": batch_expand_size,
                       "mel_target": mel_gt_target,
                       "text": character,
                       "duration": duration,
                       "energy": energy,
                       "pitch": pitch})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))
    return buffer


class FSLJSpeechDataset(Dataset):
    def __init__(self, data_path: str, mel_ground_truth, alignment_path: str, energy_path: str, pitch_path: str, text_cleaners, batch_expand_size: int, max_buffer: int = None):
        self.buffer = get_data_to_buffer(data_path, mel_ground_truth, alignment_path, energy_path, pitch_path, text_cleaners, batch_expand_size)
        if max_buffer is not None:
            self.buffer = self.buffer[:max_buffer]

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]
