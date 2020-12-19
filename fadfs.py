"""Utility functions."""
# import re
#
# text = "23.555.234(3.4)5"
# digit_list = re.findall("\d+(?:\.\d+)?", text)
# print(digit_list)
#
# import tensorflow as tf
#
# a = tf.constant(2)
# b = tf.constant(2)
# print(a+b)

import fnmatch
import glob
import os
import re

import numpy as np


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.
    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.
    Returns:
        list: List of found filenames.
    """
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


# bak_dir = "../dump_baker_fixa/npy.bak"
# root_dir = "../dump_baker_fixa"
# dest_dir = "../data/dump_libritts"
# # npy_files = sorted(find_files(root_dir, "*.npy"))
# npy_files = glob.glob(os.path.join(root_dir, "*.npy"), recursive=False)
# print(root_dir, "total npy files:", len(npy_files))

# for npy in npy_files:
#     a = np.load(npy)
#     b = np.load(os.path.join(dest_dir, os.path.basename(npy)))
#
#     c = np.append(a, b)
#     np.save(npy, c)
# print(np.load("../dump_baker_fixa/stats.npy"))
# print(np.load(os.path.join(bak_dir, "stats_f0.npy")))
# print(np.load(os.path.join(dest_dir, "stats_f0.npy")))


# def rm_files():
#     npy_files = find_files(root_dir, "*.npy")
#     for npy_file in npy_files:
#         if "_" in os.path.basename(npy_file):
#             os.remove(npy_file)
#             print(os.path.basename(npy_file))

# rm_files()

# _whitespace_re = re.compile(r"\s+")
# text = "how  are you"
# print(re.sub(_whitespace_re, " ", text))
# _curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
# m = _curly_re.match(text)
# if m:
#     print(m.group(0))


import time
from g2p_en import g2p as grapheme_to_phonem

# aa = 1601367888
# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1601453884)))
# print(int(time.time()))

alphas = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_whitespace_re = re.compile(r"\s+")
g2p = grapheme_to_phonem.G2p()
_punctuation = "!'(),.:;? "
punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def remove_punc(word: str):
    return word.translate(str.maketrans({i: '' for i in punctuation}))


def isword_alpha(word: str):
    word = remove_punc(word.upper())
    for alpha in word:
        if alpha not in alphas:
            return False
    return True


def issentence_alpha(words):
    for word in words:
        if not isword_alpha(word):
            return False
    return True


def clean_punc(texts:str):
    return remove_punc(texts.replace("e.g.", "for example").replace("'s", ""))


def text2phoneme(texts):
    results = "    "
    for word in collapse_whitespace(texts).split(" "):
        phonemes = []
        for x in g2p(word):
            if x.strip():
                prefix = ""
                if x.strip() not in _punctuation:
                    prefix = "@"
                phonemes.append(prefix + x.strip())
        results += " ".join(phonemes) + " "
    return results


f1 = open("/home/wangyl/work/code/python/TTS/data/libritts/train.txt")
lines = f1.readlines()
texts = []
ids = []
for idx in range(0, len(lines), 1):
    line = lines[idx].strip().split('|')
    words = line[-1].split()
    if len(words) < 8 or not issentence_alpha(words):
        continue
    text = line[-1].replace("-", " ").replace("\'s", "")
    text = collapse_whitespace(clean_punc(text))
    if text[-1] in _punctuation:
        text = text[:-1]
    if len(text.split()) > 15:
        if "," in text:
            ttt = text.rsplit(",", 1)
            part1 = ttt[0][:-1].strip()
            part2 = ttt[1].strip()
            if len(part1.split()) >= 8 and len(part1.split()) <= 14:
                texts.append(part1)
                ids.append(line[0])
            if len(part2.split()) >= 8 and len(part2.split()) <= 14:
                texts.append(part2)
                ids.append(line[0] + "_1")
    else:
        texts.append(text)
        ids.append(line[0])
    if len(texts) > 6000:
        break

with open('libritts.txt', "w") as file:
    for id in range(0, len(ids)):
        file.write(ids[id] + "	" + texts[id])
        file.write("\n")
        file.write(text2phoneme(texts[id]))
        file.write("\n")

#
# print(g2p(remove_punc("e.g.")))
# print(g2p("e.g."))