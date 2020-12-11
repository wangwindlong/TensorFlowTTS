# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perform preprocessing and raw feature extraction for LibriTTS dataset."""

import os
import re
from typing import Dict, List, Union, Tuple, Any

import librosa
import numpy as np
import soundfile as sf
from dataclasses import dataclass, field
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin
from tensorflow_tts.processor import BaseProcessor
from g2p_en import g2p as grapheme_to_phonem
import unicodedata
from tensorflow_tts.processor.ch_pinyin import PINYIN_DICT, mark, zh_pattern

_pad = ["pad"]
_eos = ["eos"]
_pause = ["sil", "#0", "#1", "#2", "#3"]
# 声母
_initials = ["^", "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "sh", "t", "x", "z",
             "zh"]
_tones = ["1", "2", "3", "4", "5"]
# 韵母
_finals = ["a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er", "i", "ia", "ian", "iang", "iao", "ie", "ii",
           "iii", "in", "ing", "iong", "iou", "o", "ong", "ou", "u", "ua", "uai", "uan", "uang", "uei", "uen", "ueng",
           "uo", "v", "van", "ve", "vn"]

g2p = grapheme_to_phonem.G2p()
valid_symbols = g2p.phonemes
_punctuation = "!'(),.:;? "
_arpabet = ["@" + s for s in valid_symbols] + list(_punctuation)

LIBRITTS_SYMBOLS = _pad + _pause + _initials + [i + j for i in _finals for j in _tones] + _arpabet + _eos


def is_zh(word):
    match = zh_pattern.search(word)
    return match is not None


def split_zh_en(zh_en_str):
    zh_en_group = []
    zh_gather = ""
    en_gather = ""
    zh_status = False

    for c in zh_en_str:
        if not zh_status and is_zh(c):
            zh_status = True
            if en_gather != "":
                zh_en_group.append([mark["en"], en_gather])
                en_gather = ""
        elif not is_zh(c) and zh_status:
            zh_status = False
            if zh_gather != "":
                zh_en_group.append([mark["zh"], zh_gather])
        if zh_status:
            zh_gather += c
        else:
            en_gather += c
            zh_gather = ""

    if en_gather != "":
        zh_en_group.append([mark["en"], en_gather])
    elif zh_gather != "":
        zh_en_group.append([mark["zh"], zh_gather])

    return zh_en_group


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


@dataclass
class LibriTTSProcessor(BaseProcessor):
    pinyin_dict: Dict[str, Tuple[str, str]] = field(default_factory=lambda: PINYIN_DICT)
    target_rate: int = 24000
    mode: str = "train"
    train_f_name: str = "train.txt"
    positions = {
        "file": 0,
        "text": 1,
        "speaker_name": 2,
    }  # positions of file,text,speaker_name after split line
    f_extension: str = ".wav"
    cleaner_names: str = None

    def create_items(self):
        with open(
                os.path.join(self.data_dir, self.train_f_name), mode="r", encoding="utf-8"
        ) as f:
            for line in f:
                parts = line.strip().split(self.delimiter)
                wav_path = os.path.join(self.data_dir, parts[self.positions["file"]])
                wav_path = (
                    wav_path + self.f_extension
                    if wav_path[-len(self.f_extension):] != self.f_extension
                    else wav_path
                )
                text = parts[self.positions["text"]]
                speaker_name = parts[self.positions["speaker_name"]]
                self.items.append([text, wav_path, speaker_name])

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item
        audio, rate = sf.read(wav_path, dtype="float32")

        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": wav_path.split("/")[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def setup_eos_token(self):
        return _eos[0]
        # return None  # because we do not use this

    def text_to_sequence(self, text):
        if (
                self.mode == "train"
        ):  # in train mode text should be already transformed to phonemes
            return self.symbols_to_ids(self.clean_g2p(text.split(" ")))
        else:
            return self.inference_text_to_seq(text)

    def inference_text_to_seq(self, text: str):
        return self.symbols_to_ids(self.text_to_ph(text))

    def symbols_to_ids(self, symbols_list: list):
        return [self.symbol_to_id[s] for s in symbols_list]

    def text_to_ph(self, text: str):
        return self.clean_g2p(g2p(text))

    def clean_g2p(self, g2p_text: list):
        data = ["sil"]
        for i, txt in enumerate(g2p_text):
            if "@" + txt not in LIBRITTS_SYMBOLS and txt not in LIBRITTS_SYMBOLS:
                continue
            if txt in LIBRITTS_SYMBOLS:
                data.append(txt)
            else:
                data.append("@" + txt)
        if data[-1] in _punctuation or data[-1] == "sil":
            data = data[:-1]
        data.append("sil")
        return data
