import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.base.base_text_encoder import BaseTextEncoder
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            scheduler,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, scheduler, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.batch_expand_size = self.config["trainer"]["batch_expand_size"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler

        if self.len_epoch == 1:
            self.log_step = 1
        else:
            self.log_step = 100
            
        # assert self.len_epoch % self.grad_acc_steps == 0, f"{self.len_epoch} % {self.grad_acc_steps} != 0. I was lazy, so it should be like that to work correctly"
        # # assert (self.len_epoch // self.grad_acc_steps) % self.log_step == 0, f"{self.len_epoch // self.grad_acc_steps} % {self.log_step} != 0"

        print('self.log_step:', self.log_step)
        print('self.len_epoch:', self.len_epoch)
        print('self.batch_expand_size', self.batch_expand_size)

        metric_keys = ["mse_loss", "dur_loss", "energy_loss", "pitch_loss"]
        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m for m in metric_keys], writer=self.writer
        )
        # self.evaluation_metrics = MetricTracker(
        #     "loss", *[m.name for m in self.metrics], writer=self.writer
        # )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        keys_to_gpu = ["src_seq", "src_pos", "mel_target", "mel_pos", "dur_target", "energy_target", "pitch_target"]
        for tensor_for_gpu in keys_to_gpu:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        self.optimizer.zero_grad()

        tqdm_bar = tqdm(self.train_dataloader, desc="train", total=self.len_epoch - 1)

        for batches_idx, batches in enumerate(self.train_dataloader):
            epoch_end = False
            for batch_idx, batch in enumerate(batches):
                tqdm_bar.update(1)
                
                try:
                    batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.train_metrics
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                
                global_step = batches_idx * self.batch_expand_size + batch_idx
                if global_step % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + global_step)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(global_step), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.optimizer.param_groups[0]['lr'] # self.lr_scheduler.get_last_lr()[0]
                    )
                    # self._log_predictions(**batch)
                    # self._log_spectrogram(batch["spectrogram"])
                    # rand_idx = torch.randint(low=0, high=batch["pred_short"].shape[0], size=(1,))
                    # self._log_audio(batch["pred_short"][rand_idx], "pred")
                    # self._log_audio(batch["target"][rand_idx], "target")
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()

                if global_step + 1 >= self.len_epoch:
                    epoch_end = True
                    break
            if epoch_end:
                break
        log = last_train_metrics

        # for part, dataloader in self.evaluation_dataloaders.items():
        #     val_log = self._evaluation_epoch(epoch, part, dataloader)
        #     log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
        
        # if self.lr_scheduler is not None and isinstance(self.lr_scheduler, ReduceLROnPlateau):
        #     self.lr_scheduler.step(log["test_loss"])

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
    
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["mel_pred"] = outputs
        
        if is_train:
            loss = self.criterion(**batch)
            batch.update(loss)

            batch["loss"] = batch["mse_loss"] + batch["dur_loss"] + batch["energy_loss"] + batch["pitch_loss"]
            batch["loss"].backward()

            self._clip_grad_norm()
            self.optimizer.step()

            self.train_metrics.update("grad norm", self.get_grad_norm())
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            for k, v in batch.items():
                if "loss" in k:
                    metrics.update(k, v.item())
        
        # for met in self.metrics:
        #     metrics.update(met.name, met(**batch))
        return batch

    # def _evaluation_epoch(self, epoch, part, dataloader):
    #     """
    #     Validate after training an epoch

    #     :param epoch: Integer, current training epoch.
    #     :return: A log that contains information about validation
    #     """
    #     self.model.eval()
    #     self.evaluation_metrics.reset()
    #     with torch.no_grad():
    #         for batch_idx, batch in tqdm(
    #                 enumerate(dataloader),
    #                 desc=part,
    #                 total=len(dataloader),
    #         ):
    #             batch = self.process_batch(
    #                 batch,
    #                 is_train=False,
    #                 metrics=self.evaluation_metrics,
    #                 do_opt_step=False
    #             )
    #         self.writer.set_step(epoch * self.len_epoch, part)
    #         self._log_scalars(self.evaluation_metrics)
    #         # self._log_predictions(**batch)
    #         # self._log_spectrogram(batch["spectrogram"])

    #     # add histogram of model parameters to the tensorboard
    #     if self.config["trainer"].get("log_parameters", False):
    #         for name, p in self.model.named_parameters():
    #             self.writer.add_histogram(name, p, bins="auto")
    #     return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    # def _log_predictions(
    #         self,
    #         target: torch.Tensor,
    #         pred_short,
    #         examples_to_log=10,
    #         *args,
    #         **kwargs,
    # ):
    #     if self.writer is None:
    #         return
    #     target = target.tolist()

    #     tuples = list(zip(target, pred_short, argmax_texts_raw, audio_path))
    #     shuffle(tuples)
    #     rows = {}
    #     for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
    #         target = BaseTextEncoder.normalize_text(target)
    #         wer = calc_wer(target, pred) * 100
    #         cer = calc_cer(target, pred) * 100

    #         rows[Path(audio_path).name] = {
    #             "target": target,
    #             "pred_short": pred_short,
    #             "predictions": pred,
    #             "wer": wer,
    #             "cer": cer,
    #         }
    #     self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))
    
    def _log_audio(self, audio: torch.Tensor, tag: str):
        # audio = random.choice(audio_batch.cpu())
        self.writer.add_audio(tag, audio.cpu(), sample_rate=16000)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
