''''
案例:
    演示LSTM入门代码
LSTM简介:
    概述:
        Long Short Term Memory,长短时记忆结构，能够解决传统RNN的弊端处理长序列数据效果差的问题.
    组成:
        遗忘门，输入门，细胞状态，输出门
    优点:
        1.能够解决长时依赖.
        2.缓解梯度消失，爆炸的问题一 训练更稳定.
        3.能够抓取复杂的特征.
    缺点:
        1.计算慢，消耗资源大.
        2.调参相对较难.
        3.可解释性稍差.
'''

import torch
import torch.nn as nn

def lstm_demo():
    #1.创建LSTM模型对象
    #参1：词向量维度（输入的维度），参2：隐藏层的维度（输出的维度），参3：LSTM的层数，参4：是否双向
    lstm = nn.LSTM(input_size=5, hidden_size=6, num_layers=1,bidirectional=False)

    #2.构建输入张量
    #参1：句子长度（sqe_len），参2：批次大小（batch_size）,参3：词向量维度（input_size）
    input = torch.randn(4,3,5)

    #3.初始化隐藏层和细胞状态
    #参1：LSTM层数（隐藏层层数），参2：批次大小（batch_size）,参3：隐藏层维度（hidden_size）
    h0 = torch.randn(1,3,6)
    c0 = torch.randn(1,3,6)

    #4.模型计算
    #写法1：手动传入h0 c0
    #实参列表：参1（input）->输入张量，参2->隐藏状态 和细胞状态 的元组形态
    #返回值：   output->本次输出的结果，（hn，cn）->最后一个时间步的隐藏状态和细胞状态
    # output,(hn,cn) = lstm(input,(h0,c0))

    #写法二：不传入h0和c0   底层会自动创建1个全0的h0（上一时刻的隐藏状态）
    output,(hn,cn) = lstm(input)
    #打印
    print(f'output:{output},{output.shape}')
    print(f'hn:{hn},{hn.shape}')
    print(f'cn:{cn},{cn.shape}')

if __name__ == '__main__':
    lstm_demo()