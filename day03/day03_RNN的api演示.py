'''
RNN介绍
    Recurrent Neural Network
    循环神经网络 主要用于处理序列数据
    序列数据: （后面的数据对前面的数据有依赖）时间序列数据,文本数据,音频数据,视频数据
分类：
    按照输入和输出划分：
    N VS N:     输入N个，输出N个 适用于：对联，诗词
    N VS 1:     输入N个，输出1个 适用于：分类，回归，意图识别
    1 VS N:     输入1个，输出N个  适用于：机器翻译，文本生成
    N VS M:     输入N个，输出M个  适用于：图像描述

    按照内部结构划分：
    传统的RNN：
        输入层，隐藏层（词嵌入层，循环网络层），输出层
    长短期记忆网络LSTM：
        遗忘门，输入门，细胞状态，输出门
    Bi-LSTM：
        双向LSTM
    门控循环单元GRU：
        重置门，更新门
    Bi-GRU：
        双向GRU
    传统RNN的优缺点
        优点：
            内部结构简单，资源消耗少，适合处理短序列文本
        缺点：
            处理长序列数据时，因为反向传播 结合 梯度联乘 ，过大或者过小的w值会导致 梯度爆炸或者梯度消失
'''

import torch
import torch.nn as nn

#1.演示RNN的基础代码
def dm_rnn_for_base():

    #1.创建RNN模型
    #参1:词向量维度（输入维度）     参2:隐藏层维度（输出维度）     参3:隐藏层层数
    rnn = nn.RNN(5,6,1)

    #2.主备输入数据（本次的输入）
    # 参1:句子长度（sequence_length）     参2:批次大希奥（batch_size）     参3:词向量维度(输入维度)
    input = torch.randn(1,3,5)

    #3.初始化隐藏层（上一时间步的隐藏状态）
    # 参1:隐藏层的层数（num_layers）     参2:批次大小（batch_size）     参3:隐藏层维度（输出维度hidden_size）
    h0 = torch.randn(1,3,6)

    #4.运行RNN模型
    #本次输出，本次隐藏层状态= rnn（本次输入，上一时刻的隐藏状态）
    output, hn = rnn(input,h0)

    #5.打印结果
    print(f'output:{output},output.shape:{output.shape}')    #shape:(1,3,6)
    print(f'hn:{hn},hn.shape:{hn.shape}')                    #shape:(1,3,6)
    print(f'rnn模型:{rnn}')                                   #RNN(5,6)

#1.修改RNN的代码（句子长度）
def dm_rnn_for_sequence_len():

    #1.创建RNN模型
    #参1:词向量维度（输入维度）     参2:隐藏层维度（输出维度）     参3:隐藏层层数
    rnn = nn.RNN(5,6,1)

    #2.主备输入数据（本次的输入）
    # 参1:句子长度（sequence_length）     参2:批次大希奥（batch_size）     参3:词向量维度(输入维度)
    input = torch.randn(20,3,5)

    #3.初始化隐藏层（上一时间步的隐藏状态）
    # 参1:隐藏层的层数（num_layers）     参2:批次大小（batch_size）     参3:隐藏层维度（输出维度hidden_size）
    h0 = torch.randn(1,3,6)

    #4.运行RNN模型
    #本次输出，本次隐藏层状态= rnn（本次输入，上一时刻的隐藏状态）
    output, hn = rnn(input,h0)

    #5.打印结果
    print(f'output:{output},output.shape:{output.shape}')    #shape:(20,3,6)
    print(f'hn:{hn},hn.shape:{hn.shape}')                    #shape:(1,3,6)
    print(f'rnn模型:{rnn}')                                   #RNN(5,6)

#修改RNN模型，隐藏层层数
def dm_rnn_for_hidden_layers():

    #1.创建RNN模型
    #参1:词向量维度（输入维度）     参2:隐藏层维度（输出维度）     参3:隐藏层层数
    rnn = nn.RNN(5,6,2)

    #2.主备输入数据（本次的输入）
    # 参1:句子长度（sequence_length）     参2:批次大希奥（batch_size）     参3:词向量维度(输入维度)
    input = torch.randn(1,3,5)

    #3.初始化隐藏层（上一时间步的隐藏状态）
    # 参1:隐藏层的层数（num_layers）     参2:批次大小（batch_size）     参3:隐藏层维度（输出维度hidden_size）
    h0 = torch.randn(2,3,6)

    #4.运行RNN模型
    #本次输出，本次隐藏层状态= rnn（本次输入，上一时刻的隐藏状态）
    output, hn = rnn(input,h0)

    #5.打印结果
    print(f'output:{output},output.shape:{output.shape}')    #shape:(1,3,6)
    print(f'hn:{hn},hn.shape:{hn.shape}')                    #shape:(1,3,6)
    print(f'rnn模型:{rnn}')                                   #RNN(5,6,num_layer=2)

if __name__ == '__main__':
    # dm_rnn_for_base()
    # dm_rnn_for_sequence_len()
    dm_rnn_for_hidden_layers()