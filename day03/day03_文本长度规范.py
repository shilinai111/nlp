'''
文本长度规范解释：
    概述：
        一般模型输入需要等尺寸大小的矩阵，所以需要超长文本做截断，对不足文本进行补齐
    实现方式：
        方式1：第三方包
            tensorflow#sequence
        方式2：纯python代码实现

'''
import os
from ctypes import c_ulong

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing import sequence

#截断的长度
cutlen = 10#实际开发中，对语料库的句子长度分布进行定义

#对文本输入的张量进行截断补齐
def padding(x_train):
    # 参1：待处理的文本张量
    # 参2：最大长度
    # 参3：截断策略，pre（默认，从序列前端截取(把前端截掉，保留后面十个)），post（从序列后端截取）
    # 参4：填充策略，post（默认，从序列前端填充），post（从序列后端填充）
    return sequence.pad_sequences(x_train, maxlen=cutlen,truncating='pre',padding='pre')

def padding_custom(x_train):
    #初始化列表
    list1 = []
    #遍历语料库，获取每个句子
    for sentence in x_train:
        if len(sentence) > cutlen:
            list1.append(sentence[:cutlen])
        else:
            #需要补齐0的数量
            padding_len = cutlen - len(sentence)
            list1.append(sentence + [0]*padding_len)
    return list1


if __name__ == '__main__':
    x_train = [
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        [20,21,22,23,24,25]
    ]

    # result = padding(x_train)
    result = padding_custom(x_train)
    print(f'result:{result}')