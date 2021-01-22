import json
import re
import unicodedata
from g2p_en import g2p as grapheme_to_phonem

import librosa
import numpy as np
import soundfile as sf
#
# print("1"[:-1])
# print("#".isupper())
# print("P".encode('UTF-8').isalpha())
from pypinyin import Style
from pypinyin.core import Pinyin

from tensorflow_tts.processor.baker import MyConverter, BAKER_SYMBOLS, _punctuation
from tensorflow_tts.processor.ch_pinyin import PINYIN_DICT

zh_pattern = re.compile("[\u4e00-\u9fa5]")


def is_zh(word):
    global zh_pattern
    match = zh_pattern.search(word)
    return match is not None


print(is_zh(' '))
print('Ｂ'.isalpha())

# wav_file = 'baker_data/Wave/003351.wav'
# audio, rate = sf.read(wav_file)
# audio = audio.astype(np.float32)
# print(sf._libname)
# print(rate)
# print(audio)

# from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
# from pypinyin.converter import DefaultConverter
# from pypinyin.core import Pinyin
#
#
# class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
#
#     def post_handle_nopinyin(self, chars, style, heteronym,
#                              errors, strict,
#                              pinyin, **kwargs):
#         pass
#
#
# from pypinyin import lazy_pinyin, Style
#
# style = Style.TONE3
# print(lazy_pinyin('聪明的A小兔子', style=style))
#
#
# def get_pinyin_parser():
#     my_pinyin = Pinyin(MyConverter())
#     pinyin = my_pinyin.pinyin
#     return pinyin
#
#
# def errors(char):
#     return [i + "1" for i in char.upper() if i.isalpha()]
#
#
# def text_to_sequence(text, inference=False):
#     if inference:
#         pinparser = get_pinyin_parser()
#         pinyin = pinparser(text, style=Style.TONE3, errors=lambda char: [i + "1" for i in char.upper() if i.isalpha()])
#         print(pinyin)
#         new_pinyin = []
#         for x in pinyin:
#             x = "".join(x)
#             if "#" not in x:
#                 new_pinyin.append(x)
#         print(f"phoneme seq: {new_pinyin}")
#
#

g2p = grapheme_to_phonem.G2p()
valid_symbols = g2p.phonemes

# BAKER_SYMBOLS = _pad + _pause + _initials + [i + j for i in _finals for j in _tones] + _arpabet + _eos

with open('trained/baker_mapper_char.json', "r") as f:
    data = json.load(f)
symbol_to_id = data["symbol_to_id"]


def clean_g2p(g2p_text: list):
    data = ["sil"]
    for i, txt in enumerate(g2p_text):
        if "@" + txt not in BAKER_SYMBOLS and txt not in BAKER_SYMBOLS:
            print("clean_g2p not in BAKER_SYMBOLS: ", txt)
            continue
        if txt in BAKER_SYMBOLS:
            data.append(txt)
        else:
            data.append("@" + txt)
    if data[-1] in _punctuation or data[-1] == "sil":
        data = data[:-1]
    data.append("sil")
    return data


def text_to_ph(text: str):
    return clean_g2p(g2p(text))


def inference_text_to_seq(text: str):
    return symbols_to_ids(text_to_ph(text))


def symbols_to_ids(symbols_list: list):
    return [symbol_to_id[s] for s in symbols_list]


def alpha_handler(words):
    print("alpha_handler words=", words)
    words = unicodedata.normalize('NFKC', words)
    words_len = len(words)
    result = []
    i = 0
    while i < words_len:
        cur_char = words[i]
        if cur_char.encode('UTF-8').isalpha():
            start = i
            if cur_char.islower():
                while i + 1 < words_len and words[i + 1].isalpha() and words[i + 1].islower():
                    i += 1
                # result += [words[start: i + 1]]
            else:
                result += [cur_char]
        i += 1
    print("alpha_handler result=", result)
    return result


def text_to_sequence(text, speaker_name='baker', inference=False):
    sequence = []
    tmp = ""
    if "baker" == speaker_name:
        if inference:
            my_pinyin = Pinyin(MyConverter())
            pinyin = my_pinyin.pinyin(text, style=Style.TONE3,
                                      # errors="ignore"
                                      errors=alpha_handler
                                      )
            print("text_to_sequence pinyin=", pinyin)
            new_pinyin = []
            for x in pinyin:
                x = "".join(x)
                if "#" not in x:
                    new_pinyin.append(x)
            print("text_to_sequence new_pinyin=", new_pinyin)
            phonemes = get_phoneme_from_char_and_pinyin(text, new_pinyin)
            text = " ".join(phonemes)
            print(f"phoneme seq: {text}")
        try:
            for symbol in text.split():
                tmp = symbol
                idx = symbol_to_id[symbol]
                sequence.append(idx)
        except Exception as e:
            print("text_to_sequence error", tmp)
    else:
        if not inference:  # in train mode text should be already transformed to phonemes
            sequence = symbols_to_ids(clean_g2p(text.strip().split(" ")))
        else:
            sequence = inference_text_to_seq(text)
    # add eos tokens
    sequence += ['eos_id']
    return sequence


def get_phoneme_from_char_and_pinyin(chn_char, pinyin):
    # we do not need #4, use sil to replace it "这图#2难不成#2是#1P#1过的？" zhe4 tu2 nan2 bu4 cheng2 shi4 P1 guo4 de5
    print(chn_char)
    print(pinyin)
    chn_char = chn_char.replace("#4", "")
    chn_char = unicodedata.normalize('NFKC', chn_char)
    char_len = len(chn_char)
    i, j = 0, 0
    result = ["sil"]
    while i < char_len:
        cur_char = chn_char[i]
        if is_zh(cur_char):
            print(cur_char, i, j)
            if pinyin[j][:-1] not in PINYIN_DICT:
                assert chn_char[i + 1] == "儿"
                assert pinyin[j][-2] == "r"
                tone = pinyin[j][-1]
                a = pinyin[j][:-2]
                a1, a2 = PINYIN_DICT[a]
                result += [a1, a2, '@' + tone, "er", '@5']
                if i + 2 < char_len and chn_char[i + 2] != "#":
                    result.append("#0")

                i += 2
                j += 1
            else:
                tone = pinyin[j][-1]
                a = pinyin[j][:-1]
                a1, a2 = PINYIN_DICT[a]
                result += [a1, a2, '@' + tone]

                if i + 1 < char_len and chn_char[i + 1] != "#":
                    result.append("#0")

                i += 1
                j += 1
        elif cur_char == "#":
            result.append(chn_char[i: i + 2])
            i += 2
        elif cur_char.encode('UTF-8').isalpha():  # 英文字母转换为英文arpabet音素
            start = i
            if cur_char.islower():  # 小写字母当作单词处理
                while i + 1 < char_len and chn_char[i + 1].isalpha() and chn_char[i + 1].islower():
                    i += 1
                    # print(i)
                i += 1
                # print(chn_char[start: i])
                result += ["@" + s for s in g2p(chn_char[start: i])]
            else:
                if cur_char == 'A':
                    result += ["@EY1"]
                else:
                    result += ["@" + s for s in g2p(cur_char)]
                i += 1
                j += 1
            if i < char_len and chn_char[i] != "#":
                result.append("#1")
        else:
            # not ignore the unknown char and punctuation
            # if cur_char in BAKER_SYMBOLS:
            #     result.append(cur_char)
            i += 1
    if result[-1] == "#0":
        result = result[:-1]
    result.append("sil")
    print(result)
    print(pinyin)
    print(j)

    assert j == len(pinyin)
    return result


if __name__ == '__main__':
    han = "我是#2curb#1善良#1soda#1活泼#3、好奇心#1egoism#1旺盛的#2tent#1B型血#4。"
    pinyin = "wo3 shi4 shan4 liang2 huo2 po1 hao4 qi2 xin1 wang4 sheng4 de5 B xing2 xie3".strip().split()
    print(get_phoneme_from_char_and_pinyin(han, pinyin))
    print(g2p("A"))
    print("aaswdfsdf"[1:6])
    input_text = "A B C D E Fhello你好"
    print(text_to_sequence(input_text, inference=True))
    # print("#2".islower())
    # print("Ｐ".encode('UTF-8'))  # b'\xef\xbc\xb0'
    # print("P".encode('UTF-8'))  # b'P'
