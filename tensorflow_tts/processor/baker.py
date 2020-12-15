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
"""Perform preprocessing and raw feature extraction for Baker dataset."""

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

BAKER_SYMBOLS = _pad + _pause + _initials + _finals + ["@" + i for i in _tones] + _arpabet + _eos


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
class BakerProcessor(BaseProcessor):
    pinyin_dict: Dict[str, Tuple[str, str]] = field(default_factory=lambda: PINYIN_DICT)
    cleaner_names: str = None
    target_rate: int = 24000
    speaker_name: str = "baker"
    train_f_name: str = "train.txt"
    positions = {
        "file": 0,
        "text": 1,
        "speaker_name": 2,
    }  # positions of file,text,speaker_name after split line
    f_extension: str = ".wav"

    def __post_init__(self):
        super().__post_init__()
        self.pinyin_parser = self.get_pinyin_parser()

    def setup_eos_token(self):
        return _eos[0]

    def create_items(self):
        items = []
        if self.data_dir:
            with open(
                    os.path.join(self.data_dir, "ProsodyLabeling/000001-010000.txt"),
                    encoding="utf-8",
            ) as ttf:
                lines = ttf.readlines()
                for idx in range(0, len(lines), 2):
                    utt_id, chn_char = lines[idx].strip().split()
                    pinyin = lines[idx + 1].strip().split()
                    # if "IY1" in pinyin or "Ｂ" in chn_char:
                    #     print(f"Skip this: {utt_id} {chn_char} {pinyin}")
                    #     continue
                    phonemes = self.get_phoneme_from_char_and_pinyin(chn_char, pinyin)
                    wav_path = os.path.join(self.data_dir, "Wave", "%s.wav" % utt_id)
                    items.append(
                        [" ".join(phonemes), wav_path, utt_id, self.speaker_name]
                    )

            with open(
                    os.path.join(self.data_dir, 'libritts', self.train_f_name), mode="r", encoding="utf-8"
            ) as f:
                for line in f:
                    parts = line.strip().split(self.delimiter)
                    wav_path = os.path.join(self.data_dir, 'libritts', parts[self.positions["file"]])
                    wav_path = (
                        wav_path + self.f_extension
                        if wav_path[-len(self.f_extension):] != self.f_extension
                        else wav_path
                    )
                    text = parts[self.positions["text"]]
                    speaker_name = parts[self.positions["speaker_name"]]
                    items.append([text, wav_path, wav_path.split("/")[-1].split(".")[0], speaker_name])
            self.items = items

    def get_phoneme_from_char_and_pinyin(self, chn_char, pinyin):
        # we do not need #4, use sil to replace it
        chn_char = chn_char.replace("#4", "")
        chn_char = unicodedata.normalize('NFKC', chn_char)  # 转为英文标点符号
        char_len = len(chn_char)
        i, j = 0, 0
        result = ["sil"]
        while i < char_len:
            cur_char = chn_char[i]
            if is_zh(cur_char):
                if pinyin[j][:-1] not in self.pinyin_dict:
                    assert chn_char[i + 1] == "儿"
                    assert pinyin[j][-2] == "r"
                    tone = pinyin[j][-1]
                    a = pinyin[j][:-2]
                    a1, a2 = self.pinyin_dict[a]
                    result += [a1, a2, "@" + tone, "er", "@5"]
                    if i + 2 < char_len and chn_char[i + 2] != "#":
                        result.append("#0")

                    i += 2
                    j += 1
                else:
                    tone = pinyin[j][-1]
                    a = pinyin[j][:-1]
                    a1, a2 = self.pinyin_dict[a]
                    result += [a1, a2, "@" + tone]

                    if i + 1 < char_len and chn_char[i + 1] != "#":
                        result.append("#0")

                    i += 1
                    j += 1
            elif cur_char == "#":
                result.append(chn_char[i: i + 2])
                i += 2
            elif cur_char.encode('UTF-8').isalpha():  # 英文字母转换为英文arpabet音素
                # if cur_char.upper() == 'A':
                #     result += ["@EY1"]
                # else:
                result += ["@" + s for s in g2p(cur_char)]
                if i + 1 < char_len and chn_char[i + 1] != "#":
                    result.append("#0")
                i += 1
                j += 1
            else:
                # not ignore the unknown char and punctuation
                if cur_char in BAKER_SYMBOLS:
                    result.append(cur_char)
                i += 1
        if result[-1] == "#0":
            result = result[:-1]
        result.append("sil")
        print(result)
        assert j == len(pinyin)
        return result

    def get_one_sample(self, item):
        text, wav_file, utt_id, speaker_name = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_file, dtype="float32")
        # audio = audio.astype(np.float32)
        if rate != self.target_rate:
            assert rate > self.target_rate
            audio = librosa.resample(audio, rate, self.target_rate)

        # convert text to ids
        try:
            text_ids = np.asarray(self.text_to_sequence(text, speaker_name), np.int32)
        except Exception as e:
            print(str(e))
            print(e, utt_id, text)
            return None

        # return None
        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": utt_id,
            "speaker_name": speaker_name,
            "rate": self.target_rate,
        }

        return sample

    def get_pinyin_parser(self):
        my_pinyin = Pinyin(MyConverter())
        pinyin = my_pinyin.pinyin
        return pinyin

    def text_to_sequence(self, text, speaker_name='baker', inference=False):
        sequence = []
        tmp = ""
        if "baker" == speaker_name:
            if inference:
                pinyin = self.pinyin_parser(text, style=Style.TONE3,
                                            # errors="ignore"
                                            errors=lambda char: [i for i in char.upper() if i.isalpha()]
                                            )
                new_pinyin = []
                for x in pinyin:
                    x = "".join(x)
                    if "#" not in x:
                        new_pinyin.append(x)
                phonemes = self.get_phoneme_from_char_and_pinyin(text, new_pinyin)
                text = " ".join(phonemes)
                print(f"phoneme seq: {text}")
            try:
                for symbol in text.split():
                    tmp = symbol
                    idx = self.symbol_to_id[symbol]
                    sequence.append(idx)
            except Exception as e:
                print("text_to_sequence error", tmp)
            # add eos tokens
            sequence += [self.eos_id]
        else:
            # if not inference:  # in train mode text should be already transformed to phonemes
            #     return self.symbols_to_ids(self.clean_g2p(text.split(" ")))
            # else:
            return self.inference_text_to_seq(text)
        return sequence

    def inference_text_to_seq(self, text: str):
        return self.symbols_to_ids(self.text_to_ph(text))

    def symbols_to_ids(self, symbols_list: list):
        return [self.symbol_to_id[s] for s in symbols_list]

    def text_to_ph(self, text: str):
        return self.clean_g2p(g2p(text))

    def clean_g2p(self, g2p_text: list):
        data = ["sil"]
        for i, txt in enumerate(g2p_text):
            if "@" + txt not in BAKER_SYMBOLS and txt not in BAKER_SYMBOLS:
                continue
            if txt in BAKER_SYMBOLS:
                data.append(txt)
            else:
                data.append("@" + txt)
        if data[-1] in _punctuation or data[-1] == "sil":
            data = data[:-1]
        data.append("sil")
        return data
