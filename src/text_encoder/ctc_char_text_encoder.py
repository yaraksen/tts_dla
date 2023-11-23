from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder

from collections import defaultdict

from pyctcdecode import build_ctcdecoder
import multiprocessing
import numpy as np
import gzip
import os, shutil, wget

from string import ascii_uppercase


class Hypothesis(NamedTuple):
    text: str
    prob: float


def download_kenlm(is_pruned: bool = False) -> str:
    if not os.path.exists('data/'):
        os.mkdir('data/')
    
    kenlm_name = "3-gram.pruned.1e-7.arpa.gz" if is_pruned else "3-gram.arpa.gz"
    kenlm_url = "http://www.openslr.org/resources/11/" + kenlm_name
    local_path = f'data/{kenlm_name}'
    unzipped_model_path = local_path[:-3]

    if not os.path.exists(local_path):
        print(f'Downloading {kenlm_name}...')
        kenlm_url = "http://www.openslr.org/resources/11/" + kenlm_name
        gzip_path = wget.download(kenlm_url, out=local_path)
        print('Finished downloading.')
    else:
        print('Model already downloaded.')

    if not os.path.exists(unzipped_model_path):
        print('Unzipping LM...')
        with gzip.open(gzip_path, 'rb') as f_in:
            with open(unzipped_model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print('Model already unzipped')
    
    return unzipped_model_path


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        decoded = ""
        last_c = self.EMPTY_TOK
        for c_ind in inds:
            if c_ind == self.char2ind[self.EMPTY_TOK]:
                last_c = self.EMPTY_TOK
                continue
            if last_c != self.ind2char[c_ind]:
                decoded += self.ind2char[c_ind]
                last_c = self.ind2char[c_ind]
        return decoded
    
    def _extend_and_merge(self, frame, state) -> dict:
        new_state = defaultdict(float)
        for next_char_idx, next_char_proba in enumerate(frame):
            for (pref, last_char), pref_proba in state.items():
                next_char = self.ind2char[next_char_idx]
                if next_char == last_char:
                    new_pref = pref
                else:
                    if next_char != self.EMPTY_TOK:
                        new_pref = pref + next_char
                    else:
                        new_pref = pref
                    last_char = next_char
                new_state[(new_pref, last_char)] += pref_proba * next_char_proba
        return new_state
    
    def _truncate_state(self, state: dict, beam_size: int):
        state_trunc = sorted(list(state.items()), key=lambda x: x[1], reverse=True)[:beam_size]
        return dict(state_trunc)

    def ctc_beam_search(self, probs: torch.tensor, probs_length: int, beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        probs = probs[:probs_length]
        
        state = {('', self.EMPTY_TOK): 1.0}
        for frame in probs:
            state = self._truncate_state(self._extend_and_merge(frame, state), beam_size)

        hypos = [Hypothesis(text, prob) for (text, _), prob in state.items()]

        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def ctc_lm_beam_search(self, log_probs: torch.tensor, log_probs_length: torch.tensor, alpha: float, beta: float) -> List[str]:
        uppercase_vocab = [""] + list(ascii_uppercase + ' ')

        assert log_probs_length.shape[0] == log_probs.shape[0]
        # log_probs: BxTxF
        list_logs_probs = [log_probs[i][:log_probs_length[i]].detach().cpu().numpy() for i in range(log_probs_length.shape[0])]
        assert len(list_logs_probs) == log_probs.shape[0]

        if not hasattr(self, "lm_decoder"):
            kenlm_path = download_kenlm(is_pruned=False)

            self.lm_decoder = build_ctcdecoder(
                labels=uppercase_vocab,
                kenlm_model_path=kenlm_path,
                alpha=alpha,
                beta=beta
            )

        with multiprocessing.get_context("fork").Pool() as pool:
            pred_list = self.lm_decoder.decode_batch(pool=pool, logits_list=list_logs_probs, beam_width=100)
        
        return pred_list
        
