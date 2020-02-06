# Dual-path-RNN-Pytorch
Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation implemented by Pytorch


# Plan

- [x] 2020-02-01: Reading article “[Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation](https://arxiv.org/abs/1910.06379 "Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation")”. Zhihu Article link "[阅读笔记”Dual-path RNN for Speech Separation“](https://zhuanlan.zhihu.com/p/104606356 "阅读笔记”Dual-path RNN for Speech Separation“")". Blog Article link "[阅读笔记《Dual-path RNN for speech separation》](https://www.likai.show/archives/dual-path-rnn "阅读笔记《Dual-path RNN for speech separation》")". Both articles are interpretations of the paper. If you have any questions, welcome to discuss with me

- [x] 2020-02-02: Complete data preprocessing, data set code. Dataset Code: [/data_loader/Dataset.py](https://github.com/JusperLee/Dual-path-RNN-Pytorch/blob/master/data_loader/Dataset.py)

- [x] 2020-02-03: Complete Conv-TasNet Framework (Update **/model/model.py, Trainer_Tasnet.py, Train_Tasnet.py**)

- [x] 2020-02-07: Complete Training code. (Update **/model/model_rnn.py**) and Test parameters and some details are being adjusted.

- [ ] 2020-02-06: Complete Train code and Test code.

- [ ] 2020-02-08: Fixed the code's bug.
