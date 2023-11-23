import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
import src.metric as module_metric
from src.utils import MetricTracker
from waveglow import get_wav, inference
import audio
from text import text_to_sequence

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def get_WaveGlow(waveglow_path: str):
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow

def synthesis(model, text, alpha: float, beta: float, gamma: float):
    text = np.stack([np.array(text)])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().cuda()
    src_pos = torch.from_numpy(src_pos).long().cuda()
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, beta=beta, gamma=gamma)["mel_pred"]
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)

def check_params(a, b, g) -> bool:
    if a == b == g:
        return True
    
    if a != 1.0:
        return (b == 1.0 and g == 1.0)
    if b != 1.0:
        return (a == 1.0 and g == 1.0)
    if g != 1.0:
        return (a == 1.0 and b == 1.0)
    
    raise Exception(f"{a, b, g} is wrong combination")
        

def main(config, waveglow_path: str, out_dir: str):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(device)

    # setup data_loader instances
    # dataloaders = get_dataloaders(config)
    # print('TEST SIZE:', len(dataloaders["test"]))

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # metrics = [
    #     config.init_obj(metric_dict, module_metric)
    #     for metric_dict in config['metrics']
    # ]

    # metric_names = []
    # for met in metrics:
    #     metric_names.append(met.name)
    
    vocoder_model = get_WaveGlow(waveglow_path).to(device)
    
    # print(metric_names)
    # evaluation_metrics = MetricTracker(*metric_names)

    test_texts = ["A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
                  "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
                  "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"]
    encoded_texts = [text_to_sequence(text, ["english_cleaners"]) for text in test_texts]
    os.makedirs(out_dir, exist_ok=True)

    coeffs = [0.8, 1.0, 1.2]
    with torch.no_grad():
        for alpha in coeffs:
            for beta in coeffs:
                for gamma in coeffs:
                    if not check_params(alpha, beta, gamma):
                        continue

                    for text_id, text in tqdm(enumerate(encoded_texts), desc=f"Processing {(alpha, beta, gamma)}..."):
                        filename = f"{text_id}-[a={alpha}_b={beta}_g={gamma}].mp3"
                        mel_cuda = synthesis(model, text, alpha, beta, gamma)[1]
                        inference.inference(
                            mel_cuda, vocoder_model,
                            f"{out_dir}/{filename}"
                        )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )
    args.add_argument(
        "-wgp",
        "--waveglow_path",
        default="waveglow.pth",
        type=str,
        help="File with checkpoint of vocoder model",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    # model_config = Path(args.resume).parent / "config.json"

    with open(args.config) as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # # update with addition configs from `args.config` if provided
    # if args.config is not None:
    #     with Path(args.config).open() as f:
    #         config.config.update(json.load(f))
    
    # # if `--test-data-folder` was provided, set it as a default test set
    # if args.test_data_folder is not None:
    #     test_data_folder = Path(args.test_data_folder).absolute().resolve()
    #     assert test_data_folder.exists()
    #     config.config["data"] = {
    #         "test": {
    #             "batch_size": args.batch_size,
    #             "num_workers": args.jobs,
    #             "datasets": [
    #                 {
    #                     "type": "SpeechSeparationDataset",
    #                     "args": {
    #                         "part": "",
    #                         "data_dir": test_data_folder
    #                     },
    #                 }
    #             ],
    #         }
    #     }

    # if config.config.get("data", {}).get("test", None) is None:
    #     assert config.config.get("data", {}).get("test-clean", None) is not None
    #     assert config.config.get("data", {}).get("test-other", None) is not None

    main(config, args.waveglow_path, args.output)
