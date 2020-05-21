<h2 align="center">
<p>TensorflowTTS
</h2>
<h2 align="center">
<p>Real-Time State-of-the-art Speech Synthesis for Tensorflow 2
</h2>

TensorflowTTS provides real-time state-of-the-art speech synthesis architectures (Tacotron-2, Melgan, FastSpeech ...) based-on TensorFlow 2. With Tensorflow 2, we can speed-up training/inference progress, optimizer further by using [fake-quantize aware](https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide) and [pruning](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras), make TTS models can be run faster than real-time and be able to deploy on mobile devices or embedded systems. 

## Features
- High performance on Speech Synthesis.
- Be able to fine-tune on other languages.
- Fast, Scalable and Reliable.
- Suitable for deployment.
- Easy to implement new model based-on abtract class.
- Mixed precision to speed training if posible.

## Requirements
This repository is tested on Ubuntu 18.04 with:

- Python 3.6+
- Cuda 10.1
- CuDNN 7.6.5
- Tensorflow 2.2
- [Tensorflow Addons](https://github.com/tensorflow/addons) 0.9.1

Different Tensorflow version should be working but not tested yet. This repo will try to work with latest stable tensorflow version.

## Installation
```bash
$ git clone https://github.com/dathudeptrai/TensorflowTTS.git
$ cd TensorflowTTS
$ pip install  .
```
If you want upgrade the repository and its dependencies:
```bash
$ git pull
$ pip install --upgrade .
```

# Supported Model achitectures
TensorflowTTS currently  provides the following architectures:

1. **MelGAN** released with the paper [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) by Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, Aaron Courville.
2. **Tacotron-2** released with the paper [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) by Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu.
3. **FastSpeech** released with the paper [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263) by Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu.

We are also implement some techniques to improve quality and convergence speed from following papers:

1. **Multi Resolution STFT Loss** released with the paper [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480) by Ryuichi Yamamoto, Eunwoo Song, Jae-Min Kim.
2. **Guided Attention Loss** released with the paper [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
](https://arxiv.org/abs/1710.08969) by Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara.


# Tutorial End-to-End

## Prepare Dataset

Prepare a dataset in the following format:
```
|- datasets/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
```

where metadata.csv has the following format: id|transcription. This is a ljspeech-like format, you can ignore preprocessing step if you have other format dataset.

## Preprocessing

The preprocessing have three steps:

1. Convert charactor to ids, calculate pre-normalize melspectrogram, normalize audio to [-1, 1], split dataset into train and valid part.
2. Computer mean/var of melspectrogram over **training** part.
3. Normalize melspectrogram based-on mean/var of training dataset.

This is a command line to do three steps above:

```
tensorflow-tts-preprocess --rootdir ./datasets/ --outdir ./dump/ --conf conf/preprocess.yaml
tensorflow-tts-compute-statistics --rootdir ./dump/train/ --outdir ./dump --config conf/preprocess.yaml
tensorflow-tts-normalize --rootdir ./dump --outdir ./dump --stats ./dump/stats.npy --config conf/preprocess.yaml

```

After preprocessing, a structure of project will become:
```
|- datasets/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
|- dump/
|   |- train/
|       |- ids/
|           |- LJ001-0001-ids.npy
|           |- ...
|       |- raw-feats/
|           |- LJ001-0001-raw-feats.npy
|           |- ...
|       |- norm-feats/
|           |- LJ001-0001-norm-feats.npy
|           |- ...
|       |- wavs/
|           |- LJ001-0001-wave.npy
|           |- ...
|   |- valid/
|       |- ids/
|           |- LJ001-0009-ids.npy
|           |- ...
|       |- raw-feats/
|           |- LJ001-0009-raw-feats.npy
|           |- ...
|       |- norm-feats/
|           |- LJ001-0009-norm-feats.npy
|           |- ...
|       |- wavs/
|           |- LJ001-0009-wave.npy
|           |- ...
|   |- stats.npy/ 
|   |- train_utt_ids.npy
|   |- valid_utt_ids.npy
```

Where stats.npy contains mean/var of train melspectrogram, train_utt_ids/valid_utt_ids contains training and valid utt ids respectively. We use suffix (ids, raw-feats, norm-feats, wave) for each type of input. 

**IMPORTANT NOTES**:
- This preprocessing step based-on [ESP-NET](https://github.com/espnet/espnet) so you can combine all models here with other models from espnet repo.

## Training models

To know how to training model from scratch or fine-tune with other datasets/languages, pls see detail at example directory.

- For Tacotron-2 tutorial, pls see [example/tacotron-2]()
- For FastSpeech tutorial, pls see [example/fastspeech]()
- For MelGAN tutorial, pls see [example/melgan]()
- For MelGAN + STFT Loss tutorial, pls see [example/melgan-stft]()

# Abstract Class Explaination

## Abstract DataLoader Tensorflow-based dataset

A detail implementation of abstract dataset class from [tensorflow_tts/dataset/abstract_dataset](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/datasets/abstract_dataset.py). There are some functions you need overide and understand:

1. **get_args**: This function return argumentation for **generator** class, normally is utt_ids.
2. **generator**: This funtion have an inputs from **get_args** function and return a inputs for models.
3. **get_output_dtypes**: This function need return dtypes for each element from **generator** function.
4. **get_len_dataset**: Return len of datasets, normaly is len(utt_ids).

**IMPORTANT NOTES**:

- A pipeline of creating dataset should be: cache -> shuffle -> map_fn -> get_batch -> prefetch.
- If you do shuffle before cache, the dataset won't shuffle when it re-iterations over datasets.
- You should apply map_fn to make each elements return from **generator** function have a same length before get batch and feed it into a model.

Some example to use this **abstract_dataset** is [tacotron_dataset.py](), [fastspeech_dataset.py](), [melgan_dataset.py]().


## Abstract Trainer Class

A detail implementation of base_trainer from [tensorflow_tts/trainer/base_trainer.py](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py). It include [Seq2SeqBasedTrainer](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py#L265) and [GanBasedTrainer](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py#L149) inherit from [BasedTrainer](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py#L16). There a some functions you **MUST** overide when implement new_trainer:

- **compile**: This function aim to define a models, and losses.
- **_train_step**: This function perform one step training logic of a model.
- **_eval_epoch**: This function perform eval epoch, include **_eval_step**, **generate_and_save_intermediate_result** and **_write_to_tensorboard**.
- **_eval_step**: This function perform evaluation steps, calculate loss and write it into tensorboard.
- **_check_log_interval**: This function write training loss into tensorboard after pre-define interval steps.
- **generate_and_save_intermediate_result**: This function will save intermediate result such as: plot alignment, save audio generated, plot mel-spectrogram ...
- **_check_train_finish**: Check if a training progress finished or not.

All models on this repo are trained based-on **GanBasedTrainer** (see [train_melgan.py](), [train_melgan_stft.py]) and **Seq2SeqBasedTrainer** (see [train_tacotron2.py](), [train_fastspeech.py]()). In the near future, i will implement MultiGPU for **BasedTrainer** class.


# References implementations
- https://github.com/Rayhane-mamah/Tacotron-2
- https://github.com/espnet/espnet
- https://github.com/mozilla/TTS
- https://github.com/kan-bayashi/ParallelWaveGAN
- https://github.com/huggingface/transformers
- https://github.com/descriptinc/melgan-neurips