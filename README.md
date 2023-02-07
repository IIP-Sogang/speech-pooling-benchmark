# deeplearning_framework_pl


## Introduction


## Getting Started

### 0. feature extraction
```
python pre_extract_feats.py iemocap /home/nas4/DB/IEMOCAP /home/nas4/DB/IEMOCAP/IEMOCAP None None _feat_1_12
```


### 1. train
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/esc_50.yaml --mode train
```


### 2. test
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/esc_50.yaml --mode test
```
