import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
import torch

from torch.utils.data import Dataset
from src.utils import ROOT_PATH
from glob import glob
from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class SpeechSeparationDataset(Dataset):
    def __init__(self, part, data_dir, config_parser: ConfigParser, *args, **kwargs):
        super().__init__()
        self._data_dir = Path(data_dir)
        self.config_parser = config_parser
        self._id_to_class = {}
        self.num_classes = 0
        self._index = self._get_or_load_index(part)

    def id_to_class(self, key):
        if key not in self._id_to_class.keys():
            self._id_to_class[key] = self.num_classes
            self.num_classes += 1
        return self._id_to_class[key]
    
    def __getitem__(self, ind):
        data_dict = self._index[ind]
        item = {
            "mixed": self.load_audio(data_dict["mixed"]),
            "ref": self.load_audio(data_dict["ref"]),
            "target": self.load_audio(data_dict["target"]),
            "speakers_ids": data_dict["speakers_ids"]
        }
        return item
    
    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def __len__(self):
        return len(self._index)

    def _get_or_load_index(self, part):
        if os.path.exists("/kaggle/"):
            index_path = Path(f"/kaggle/input/index-nowad/{part}_index.json")
        elif os.path.exists("/home/jupyter/"):
            index_path = Path(f"/home/jupyter/work/resources/{part}_index.json")
        else:
            index_path = self._data_dir / f"{part}_indices.json"
        
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            raise Exception(f"Part {part} does not exist")

        mixed_files = sorted(glob(str(split_dir / "**/*-mixed.wav")))
        ref_files = sorted(glob(str(split_dir / "**/*-ref.wav")))
        target_files = sorted(glob(str(split_dir / "**/*-target.wav")))
        assert len(mixed_files) == len(ref_files) == len(target_files), "ref/mix/target should have the same size"

        for mixed, ref, target in tqdm(
                zip(mixed_files, ref_files, target_files), desc=f"Indexing Speech Separation Dataset: {part}"
        ):
            index.append(
                {
                    "mixed": mixed,
                    "ref": ref,
                    "target": target,
                    "speakers_ids": self.id_to_class(mixed.split("/")[-1].split("_")[0])
                }
            )
        print("INDEX SIZE:", )
        print("NUM CLASSES OBSERVED:", self.num_classes)
        return index
