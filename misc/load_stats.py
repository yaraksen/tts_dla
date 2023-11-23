from tqdm import tqdm
import pyworld as pw
import scipy as sp
import numpy as np
from numpy.linalg import norm
import os
from src.utils import ROOT_PATH
import torchaudio


def get_pitch_stats() -> None:
    pitch_dir = ROOT_PATH / "data" / "pitch"
    if not os.path.exists(pitch_dir):
        os.mkdir(pitch_dir)
    
    mel_dir = ROOT_PATH / "data" / "mels"
    wav_dir = ROOT_PATH / "data" / "LJSpeech-1.1" / "wavs"
    
    wav_paths = [wav_path.name for wav_path in wav_dir.iterdir()]
    path_to_idx = {path: idx for idx, path in enumerate(sorted(wav_paths))}
    pitch_max, pitch_min = 0, 1e3
    pbar = tqdm(list(wav_dir.iterdir()), desc="Calculating pitch...")

    for wav_path in pbar:
        mel_name = "ljspeech-mel-%05d.npy" % (1 + path_to_idx[wav_path.name])
        pitch_name = "ljspeech-pitch-%05d.npy" % (1 + path_to_idx[wav_path.name])
        pitch_filename = pitch_dir / pitch_name
        mel_path = mel_dir / mel_name
        mel = np.load(mel_path)
        n_feats = mel.shape[0]

        # Loading audio and casting to numpy
        wav, sample_rate = torchaudio.load(wav_path)
        assert wav.shape[0] == 1, "wav from LJSpeech should have single channel"
        wav = wav.numpy().astype(np.float64).squeeze(0)
        wav_len = wav.shape[0]

        # took from library examples https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder/blob/fa84b8bd224d465594b211d78446a26dc6d96173/README.md?plain=1#L29
        pitch, t = pw.dio(wav,
                        sample_rate,
                        frame_period=1000 * wav_len / sample_rate / n_feats)
        pitch = pw.stonemask(wav, pitch, t, sample_rate)

        # appers a bit longer then n_feats
        pitch = pitch[:n_feats]
        # making pitch a more continuous function
        interpolate_indices = np.nonzero(pitch)[0]
        interpolated_pitch = sp.interpolate.interp1d(
                                    x=interpolate_indices,
                                    y=pitch[interpolate_indices],
                                    fill_value=(pitch[interpolate_indices][0], pitch[interpolate_indices][-1]),
                                    bounds_error=False)
        pitch = interpolated_pitch(np.arange(n_feats))
        np.save(file=pitch_filename, arr=pitch)
        pitch_max = max(np.max(pitch), pitch_max)
        pitch_min = min(np.min(pitch), pitch_min)

    print(f"Pitch predictor interval: [{pitch_min}; {pitch_max}]")


def get_energy_stats() -> None:
    energy_output_dir = ROOT_PATH / "data" / "energy"
    if not os.path.exists(energy_output_dir):
        os.mkdir(energy_output_dir)
    energy_max, energy_min = 0, 1e3
    mel_dir = ROOT_PATH / "data" / "mels"

    pbar = tqdm(list(mel_dir.iterdir()), desc="Calculating energy...")
    for mel_path in pbar:
        energy = norm(np.load(mel_path), axis=-1)
        np.save(file=energy_output_dir / mel_path.name.replace("mel", "energy"),
                arr=energy)
        energy_max = max(np.max(energy), energy_max)
        energy_min = min(np.min(energy), energy_min)

    print(f"Energy predictor interval: [{energy_min}; {energy_max}]")

if __name__ == "__main__":
    get_pitch_stats()
    get_energy_stats()