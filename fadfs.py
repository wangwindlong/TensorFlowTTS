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


bak_dir = "../dump_baker_fixa/npy.bak"
root_dir = "../dump_baker_fixa"
dest_dir = "../data/dump_libritts"
# npy_files = sorted(find_files(root_dir, "*.npy"))
npy_files = glob.glob(os.path.join(root_dir, "*.npy"), recursive=False)
print(root_dir, "total npy files:", len(npy_files))

# for npy in npy_files:
#     a = np.load(npy)
#     b = np.load(os.path.join(dest_dir, os.path.basename(npy)))
#
#     c = np.append(a, b)
#     np.save(npy, c)
# print(np.load("../dump_baker_fixa/stats.npy"))
# print(np.load(os.path.join(bak_dir, "stats_f0.npy")))
# print(np.load(os.path.join(dest_dir, "stats_f0.npy")))


def rm_files():
    npy_files = find_files(root_dir, "*.npy")
    for npy_file in npy_files:
        if "_" in os.path.basename(npy_file):
            os.remove(npy_file)
            print(os.path.basename(npy_file))

# rm_files()

