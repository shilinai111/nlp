"""
案例:
演示文件张量实现方式之 word Embedding.
大白话解释下:word2vec 和 word Embedding的区别
    word2vec:
       先办证，后干活.会预先训练处词向量(模型)，后续再带入模型做其它
    word Embedding:
        边办证，边干活。训练过程中，词向量会自动生成. 就是以前我们
# 禁用oneDNN优化包...
"""


import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.distribute import parameter_server_strategy
import jieba
import torch.nn as nn

# 1.定义函数 ，演示：word Embedding 讲文本装成词向量，并可视化

def dm01_embedding_show():
    #1.定义遍历，记录：待处理的文本
    sentence1 = "传智教育是一家上市公司，旗下有黑马程序员品牌，我是再黑马学习人工智能"
    sentence2 = "我爱自然语言处理"
    #2.把上述两句话，封装为：列表
    sentences = [sentence1, sentence2]
    #3.使用jieba进行分词处理
    #3.1定义遍历，记录：分此后的数据——>即：词语列表
    word_list = []
    #3.2遍历上述句子列表，获取到每个句子
    for sentence in sentences:
        word_list.append(jieba.lcut(sentence))
    #3.4打印分词结果
    """
    [
        []
        []
    ]
    """
    print(f'分词结果:{word_list}')
if __name__ == "__main__":
    dm01_embedding_show()