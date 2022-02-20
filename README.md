# Dual-path-RNN-Pytorch
Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation implemented by Pytorch

If you have any questions, you can ask them through the issue.

If you find this project helpful, you can give me a star generously.

Demo Pages: [Results of pure speech separation model](https://cslikai.cn/project/Pure-Audio/)
# Plan

- [x] 2020-02-01: Reading article “[Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation](https://arxiv.org/abs/1910.06379 "Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation")”. Zhihu Article link "[阅读笔记”Dual-path RNN for Speech Separation“](https://zhuanlan.zhihu.com/p/104606356 "阅读笔记”Dual-path RNN for Speech Separation“")". Blog Article link "[阅读笔记《Dual-path RNN for speech separation》](https://www.likai.show/archives/dual-path-rnn "阅读笔记《Dual-path RNN for speech separation》")". Both articles are interpretations of the paper. If you have any questions, welcome to discuss with me

- [x] 2020-02-02: Complete data preprocessing, data set code. Dataset Code: [/data_loader/Dataset.py](https://github.com/JusperLee/Dual-path-RNN-Pytorch/blob/master/data_loader/Dataset.py)

- [x] 2020-02-03: Complete Conv-TasNet Framework (Update **/model/model.py, Trainer_Tasnet.py, Train_Tasnet.py**)

- [x] 2020-02-07: Complete Training code. (Update **/model/model_rnn.py**) and Test parameters and some details are being adjusted.

- [x] 2020-02-08: Fixed the code's bug.

- [x] 2020-02-11: Complete Testing code.

# Dataset
We used the WSJ0 dataset as our training, test, and validation sets. Below is the data download link and mixed audio code for WSJ0.
- [Audio mix Sample](https://www.merl.com/demos/deep-clustering/media/female-female-mixture.wav)

- [WSJ0 Dataset](https://catalog.ldc.upenn.edu/LDC93S6A)

- [Create Dataset](https://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip)

# Training
## Training for Conv-TasNet model
1. First, you need to generate the scp file using the following command. The content of the scp file is "filename && path".
```shell
python create_scp.py
```

2. Then you can modify the training and model parameters through "[config/Conv_Tasnet/train.yml](https://github.com/JusperLee/Dual-Path-RNN-Pytorch/tree/master/config/Conv_Tasnet )".
```shell
cd config/Conv-Tasnet
vim train.yml
```

3. Then use the following command in the root directory to train the model.
```shell
python train_Tasnet.py --opt config/Conv_Tasnet/train.yml
```
## Training for Dual Path RNN model
1. First, you need to generate the scp file using the following command. The content of the scp file is "filename && path".
```shell
python create_scp.py
```

2. Then you can modify the training and model parameters through "[config/Dual_RNN/train.yml](https://github.com/JusperLee/Dual-Path-RNN-Pytorch/tree/master/config/Dual_RNN "config / Dual_RNN / train.yml")".
```shell
cd config/Dual_RNN
vim train.yml
```

3. Then use the following command in the root directory to train the model.
```shell
python train_rnn.py --opt config/Dual_RNN/train.yml
```

# Inference

## Conv-TasNet
You need to modify the default parameters in the test_tasnet.py file, including test files, test models, etc.
### For multi-audio
```shell
python test_tasnet.py 
```
### For single-audio
```shell
python test_tasnet_wav.py 
```
## Dual-Path-RNN
You need to modify the default parameters in the test_dualrnn.py file, including test files, test models, etc.
### For multi-audio
```shell
python test_dualrnn.py 
```
### For single-audio
```shell
python test_dualrnn_wav.py 
```

# Pretrain Model

## Conv-TasNet

[Conv-TasNet model](https://drive.google.com/open?id=1MRe4jiwgtAFZErjz-LWuuyEG8VGSU0YS "Google Driver")

## Dual-Path-RNN
[Dual-Path-RNN model](https://drive.google.com/open?id=1TInJB-idggkKJ5YkNvnrTopum_HgX3_o "Google Driver") 

# Result

## Conv-TasNet
![](https://github.com/JusperLee/Dual-Path-RNN-Pytorch/blob/master/log/Conv_Tasnet/loss.png)

Final Results: **15.8690** is 0.56 higher than **15.3** in the paper.

## Dual-Path-RNN

Final Results: **18.98** is 0.1 higher than **18.8** in the paper.

# Reference
1. Luo Y, Chen Z, Yoshioka T. Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation[J]. arXiv preprint arXiv:1910.06379, 2019.
2. [Conv-TasNet code](https://github.com/JusperLee/Conv-TasNet "Conv-TasNet code") && [Dual-RNN code](https://github.com/yluo42/TAC/blob/master/utility/models.py "Dual-RNN code")
