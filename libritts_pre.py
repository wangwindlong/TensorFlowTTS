import os
import random
import shutil
import soundfile as sf

libri_path = "../data/LibriTTS"  # absolute path to TensorFlowTTS.
dataset_path = "../data/libritts"  # Change to your paths. This is a output of re-format dataset.
subset = "train-clean-100"
with open(os.path.join(libri_path, "SPEAKERS.txt")) as f:
    data = f.readlines()

dataset_info = {}
max_speakers = 20  # Max number of speakers to train on
min_len = 20  # Min len of speaker narration time
max_file_len = 11  # max audio file lenght
min_file_len = 2  # min audio file lenght
possible_dataset = [i.split("|") for i in data[12:] if
                    i.split("|")[2].strip() == subset and float(i.split("|")[3].strip()) >= min_len]

ids = [i[0].strip() for i in possible_dataset]

possible_map = {}
subset_path = os.path.join(libri_path, subset)
for i in os.listdir(subset_path):
    if i in ids:
        id_path = os.path.join(subset_path, i)
        id_dur = 0
        id_included = []

        for k in os.listdir(id_path):
            for j in os.listdir(os.path.join(id_path, k)):
                if ".wav" in j:
                    f_path = os.path.join(id_path, k, j)
                    sf_file = sf.SoundFile(f_path)
                    dur = len(sf_file) / sf_file.samplerate
                    if max_file_len < dur or dur < min_file_len:
                        continue
                    else:
                        id_included.append(f_path)
                        id_dur += dur

        possible_map[i] = {"dur": id_dur, "included": id_included}

# %%

poss_speakers = {k: v["included"] for k, v in possible_map.items() if v["dur"] / 60 >= min_len}

# %%

to_move = list(poss_speakers.keys())
random.shuffle(to_move)
to_move = to_move[:max_speakers]

# %%

text_included = []
for sp_id, v in poss_speakers.items():
    if sp_id in to_move:
        for j in v:
            f_name = j.split(os.path.sep)[-1]
            text_f_name = f_name.split(".wav")[0] + ".txt"
            os.makedirs(os.path.join(dataset_path, sp_id), exist_ok=True)
            f_path = os.path.join(dataset_path, sp_id, f_name)
            shutil.copy(j, f_path)
            text_file = j.replace(".wav", ".normalized.txt")
            shutil.copy(text_file, os.path.join(dataset_path, sp_id, text_f_name))
            with open(text_file) as text_file:
                text = str(f_path) + "|" + (
                    " ".join([line.strip() for line in text_file.readlines()])) + "|" + sp_id
            text_included.append(text)

with open(os.path.join(dataset_path, "train.txt"), "w") as f:
    for text in text_included:
        f.write(text)
        f.write('\n')
    f.close()
