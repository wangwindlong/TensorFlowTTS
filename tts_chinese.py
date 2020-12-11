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
    pretrained_path="trained/model-74000.h5",
    name="tacotron2"
)

# # FastSpeech2
# fastspeech2_config = AutoConfig.from_pretrained('examples/fastspeech2/conf/fastspeech2.baker.v2.yaml')
# fastspeech2 = TFAutoModel.from_pretrained(
#     config=fastspeech2_config,
#     pretrained_path="trained/fastspeech2-200k.h5",
#     name="fastspeech2"
# )

# MB-MelGAN
mb_melgan_config = AutoConfig.from_pretrained('examples/multiband_melgan/conf/multiband_melgan.baker.v1.yaml')
mb_melgan = TFAutoModel.from_pretrained(
    config=mb_melgan_config,
    pretrained_path="trained/mb.melgan.fixa_250k.h5",
    name="mb_melgan"
)

processor = AutoProcessor.from_pretrained(pretrained_path="trained/baker_mapper_mix.json")  # BakerProcessor


def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
    input_ids = processor.text_to_sequence(input_text, inference=True)

    # text2mel part
    if text2mel_name == "TACOTRON":
        _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32)
        )
    elif text2mel_name == "FASTSPEECH2":
        mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        )
    else:
        raise ValueError("Only TACOTRON, FASTSPEECH2 are supported on text2mel_name")

    # vocoder part
    if vocoder_name == "MB-MELGAN":
        # tacotron-2 generate noise in the end symtematic, let remove it :v.
        if text2mel_name == "TACOTRON":
            remove_end = 1024
        else:
            remove_end = 1
        audio = vocoder_model.inference(mel_outputs)[0, :-remove_end, 0]
    else:
        raise ValueError("Only MB_MELGAN are supported on vocoder_name")

    if text2mel_name == "TACOTRON":
        return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
    else:
        return mel_outputs.numpy(), audio.numpy()


def visualize_attention(alignment_history):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title(f'Alignment steps')
    im = ax.imshow(
        alignment_history,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.show()
    plt.close()


def visualize_mel_spectrogram(mels):
    mels = tf.reshape(mels, [-1, 80]).numpy()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title(f'Predicted Mel-after-Spectrogram')
    im = ax1.imshow(np.rot90(mels), aspect='auto', interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
    plt.show()
    plt.close()


def convert_h5to_pb():
    pass

input_text = "H I M N S X F"
# ['EH1', 'M']  M发成成N音,M发音短导致?
# ['EH1', 'N']  EH1 发音很短,见300032,原音频读法问题?,可以听懂,问题不大
# ['EY1', 'CH'] CH不发音
# ['EH1', 'K', 'S']  K没发音,S结尾有部分杂音,见300063
# ['EH1', 'S'] 单独的时候 S发音成M? 句子没问题
# ['EH1', 'F'] F没发音(发音短?)
input_text = "请小新到 A 零零一窗口办理业务"
# input_text = "中国共产党,共产党,你好吗?我很好"
# setup window for tacotron2 if you want to try
tacotron2.setup_window(win_front=2, win_back=2)
### Tacotron2 + MB-MelGAN
mels, alignment_history, audios = do_synthesis(input_text, tacotron2, mb_melgan, "TACOTRON", "MB-MELGAN")
print(mels.shape)
print(mels[0].shape)
print("用时：", (datetime.datetime.now() - starttime).seconds)
visualize_attention(alignment_history[0])
visualize_mel_spectrogram(mels[0])
write('test_mix2.wav', 24000, audios)
# ipd.Audio(audios, rate=24000)
#
# # # FastSpeech2 + MB-MelGAN
# # mels, audios = do_synthesis(input_text, fastspeech2, mb_melgan, "FASTSPEECH2", "MB-MELGAN")
# # visualize_mel_spectrogram(mels[0])
# # ipd.Audio(audios, rate=24000)
# # write('test.wav', 24000, audios)


# convert_h5to_pb()
