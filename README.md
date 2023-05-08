# Spam-SMS-recognition

一个简单垃圾短信识别模型，使用BERT提取词向量来训练一个LSTM模型，本科毕业设计，答辩已过，代码完整性没问题，但是可能有很多细节性上的问题。

依赖jieba, Pytorch, transformers, 以及Nvidia的CUDA和cuDNN

训练模型直接运行train.py

main.py是一个小的模型测试脚本。

数据集默认1万条，存放在data文件夹内，同文件夹下也存放有原始的80万条短信。
