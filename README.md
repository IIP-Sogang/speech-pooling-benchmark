## Introduction
This code repository is the official implementation of the [Unsupervised Speech Representation Pooling Using Vector Quantization](https://arxiv.org/abs/2304.03940)
To reproduce the experiments in this paper, perform the following three steps:
1. Download the dataset
2. Perform feature extraction
3. Train/Test



### 1. Download the dataset
We do not provide a guide for downloading the datasets here, but only provide the links. It is recommended to download the following four datasets into the `data` directory.
- [Google SpeechCommands dataset V2](https://www.tensorflow.org/datasets/catalog/speech_commands?hl=en)
- [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [fluent speech cojmmand dataset](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/)
- [IEMOCAP](https://sail.usc.edu/iemocap/)


### 2. feature extraction
Our naming convention is as follows. The term "mean" refers to the average of the representations from each transformer block. And "VQ" means the quantized representations, denoted as "**Q**", using vector quantizer module.

| | wav2vec2.0-base | wav2vec2.0-large | XLS-R |
|---------|---------|----------|----------|
|mean| _mean | _mean | _mean | 
|VQ | _VQ | _VQ | _VQ |

#### KWS
- wav2vec2.0-base mean
```
# training
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset training --tag _wav2vec2_base_mean --ext_type wav2vec2 --method mean
# validation
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset validation --tag _wav2vec2_base_mean --ext_type wav2vec2 --method mean
# testing
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset testing --tag _wav2vec2_base_mean --ext_type wav2vec2 --method mean
```
- wav2vec2.0-base VQ
```
# training
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset training --tag _wav2vec2_base_vq --ext_type VQWav2VeLargeExtractor --method vq
#validation
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset validation --tag _wav2vec2_base_vq --ext_type VQWav2VeLargeExtractor --method vq
# testing
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset testing --tag _wav2vec2_base_vq --ext_type VQWav2VeLargeExtractor --method vq

```
- wav2vec2.0-large mean
```
# training
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset training --tag _wav2vec2_large_mean --ext_type Wav2VeLargeExtractor --method mean
# validation
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset validation --tag _wav2vec2_base_mean --ext_type Wav2VeLargeExtractor --method mean
# testing
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset testing --tag _wav2vec2_base_mean --ext_type Wav2VeLargeExtractor --method mean
```
- wav2vec2.0-large VQ
```
# training
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset training --tag _wav2vec2_large_vq --ext_type VQWav2VeLargeExtractor --method mean
# validation
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset validation --tag _wav2vec2_base_vq --ext_type VQWav2VeLargeExtractor --method mean
# testing
python pre_extract_feats.py --data_name speechcommands --root data --new_root data --url speech_commands_v0.02 --subset testing --tag _wav2vec2_base_vq --ext_type VQWav2VeLargeExtractor --method mean
```

```
python pre_extract_feats.py --root data --new_root data --url speech_commands_v0.02 --subset train --tag _wav2vec2_base_mean --ext_type wav2vec2 --method vq
```


```
# Context representation - wav2vec2 base
python pre_extract_feats.py iemocap /home/nas4/DB/IEMOCAP /home/nas4/DB/IEMOCAP/IEMOCAP None None _feat_1_12

# Context representation - wav2vec2 xlsr
CUDA_VISIBLE_DEVICES=0 python pre_extract_feats.py iemocap /home/nas4/DB/IEMOCAP /home/nas4/DB/IEMOCAP/IEMOCAP None None xlsr_feat_1_12


# IC - train
python pre_extract_feats.py fluentspeechcommand /home/nas4/DB/fluent_speech_commands /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset None train _feat_1_12

# IC -valid
python pre_extract_feats.py fluentspeechcommand /home/nas4/DB/fluent_speech_commands /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset None valid _feat_1_12

# IC - test
python pre_extract_feats.py fluentspeechcommand /home/nas4/DB/fluent_speech_commands /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset None test _feat_1_12

```


### 1. train
```
# EMOTION RECOGNITION
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/er_avg.yaml --mode train
```


### 2. test
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/esc_50.yaml --mode test
```
