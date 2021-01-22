import datetime

import tensorflow as tf

import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# import IPython.display as ipd
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow_tts.configs.tacotron2 import Tacotron2Config

from tensorflow_tts.models.tacotron2 import TFTacotron2

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

starttime = datetime.datetime.now()

# Tacotron2
tacotron2_config = AutoConfig.from_pretrained('examples/tacotron2/conf/tacotron2.baker.v1.yaml')
tacotron2 = TFAutoModel.from_pretrained(
    config=tacotron2_config,
    # pretrained_path="trained/model-60000.h5",
    # pretrained_path="trained/taco_char_server_126k.h5",
    pretrained_path="trained/taco_local_word_new_82k.h5",
    name="tacotron2"
)

# FastSpeech2
fastspeech2_config = AutoConfig.from_pretrained('examples/fastspeech2/conf/fastspeech2.baker.v2.yaml')
fastspeech2 = TFAutoModel.from_pretrained(
    config=fastspeech2_config,
    pretrained_path="trained/fs_local_135k.h5",
    name="fastspeech2"
)

# MB-MelGAN
mb_melgan_config = AutoConfig.from_pretrained('examples/multiband_melgan/conf/multiband_melgan.baker.v1.yaml')
mb_melgan = TFAutoModel.from_pretrained(
    config=mb_melgan_config,
    # pretrained_path="trained/mb.melgan.char-800k.h5",  # "trained/mb.melgan-1M.h5"
    pretrained_path="trained/mb.melgan_word_428k.h5",  # "trained/mb.melgan-1M.h5"
    # is_build=False,  # don't build model if you want to save it to pb. (TF related bug)
    name="mb_melgan"
)

# LJSpeechProcessor
# processor = AutoProcessor.from_pretrained("trained/baker_mapper_char.json")
processor = AutoProcessor.from_pretrained("trained/baker_mapper_word.json")


# save tacotron2 to pb
def save_tacotron2_pb():
    input_text = "i love you so much."
    input_ids = processor.text_to_sequence(input_text)

    tacotron2.setup_window(win_front=3, win_back=3)
    tacotron2.setup_maximum_iterations(3000)
    decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    )
    tacotron2.load_weights("examples/tacotron2/exp/baker.mix.ali/checkpoints/model-22000.h5")
    # save model into pb and do inference. Note that signatures should be a tf.function with input_signatures.
    tf.saved_model.save(tacotron2, "./saved_tacotron2", signatures=tacotron2.inference)


# Load tacotron2 and Inference
def test_tacotron2(input_text):
    # tacotron2 = tf.saved_model.load("./saved_tacotron2")

    tacotron2.setup_window(win_front=3, win_back=3)
    tacotron2.setup_maximum_iterations(3000)
    input_ids = processor.text_to_sequence(input_text, "baker", True)
    print(input_ids)

    decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )
    return mel_outputs


# Load tacotron2 and Inference
def test_fs2(input_text):
    input_ids = processor.text_to_sequence(input_text, inference=True)
    mel_before, mel_outputs, duration_outputs, _, _ = fastspeech2.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
    return mel_outputs


# save melgan to pb
def save_melgan_pb():
    # processor = AutoProcessor.from_pretrained(pretrained_path="trained/baker_mapper.json")  # BakerProcessor
    fake_mels = tf.random.uniform(shape=[4, 256, 80], dtype=tf.float32)
    audios = mb_melgan.inference(fake_mels)
    mb_melgan.load_weights(
        "examples/multiband_melgan/exp/train")
    tf.saved_model.save(mb_melgan, "./saved_mb_melgan", signatures=mb_melgan.inference)


# load melgan and Inference
def test_melgan():
    mb_melgan = tf.saved_model.load("./saved_mb_melgan")
    mels = np.load("../dump_lj/valid/norm-feats/LJ001-0009-norm-feats.npy")
    # print(mels)
    print(type(mels))
    print(mels.shape)
    audios = mb_melgan.inference(mels[None, ...])
    print(type(audios))
    print(audios.shape)
    audios = audios[0, :-1024, 0]
    write('test_en.wav', 24000, audios.numpy())


def test_tacotron2_melgan(mels):
    # mb_melgan = tf.saved_model.load("./saved_mb_melgan")
    print("test_tacotron2_melgan mels=", mels)
    audios = mb_melgan.inference(mels)
    print(type(audios))
    print(audios.shape)
    audios = audios[0, :-2048, 0]
    write('test_ta2.wav', 24000, audios.numpy())


# save_tacotron2_pb()
# save_melgan_pb()
# test_melgan()
input_text = "你好吗?我很好,有空一起玩啊,请A二九五号到F零三号窗口请M二九五号到N零三号窗口"
input_text = "ABCD四个选项中哪个是正确答案"
# input_text = "与#1importation#1女演员#1land#1拍吻戏#3，陈小春#2agenda#1不怕#1应采儿#1吃醋#4。"
# input_text = "ABCD四个选项中,哪一个是正确答案?"
# input_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ结束了"
# input_text = "OPQRST结束了"
input_text = "A computer vision 是个好工作"
# input_text = "Unless you work on a ship, it's unlikely that you use the word boatswain in everyday conversation, " \
#              "so it's understandably a tricky one. The word - which refers to a petty officer in charge of hull " \
#              "maintenance is not pronounced boats-wain Rather, it's bo-sun to reflect the salty pronunciation of " \
#              "sailors, as The Free Dictionary explains. "

# test_tacotron2_melgan(test_tacotron2(input_text))
test_tacotron2_melgan(test_fs2(input_text))
