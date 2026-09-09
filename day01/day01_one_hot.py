"""
演示 one-hot编码，分别演示复杂版 和简单版.
文本张量相关介绍:
概述:
对文本进行切词，把每个词转成对应的词向量，即:用词向量的形式来描述文本文本张量(也叫:词向量表示法)
作用:
模型无法直接解析文本，可以转成 文本张量(词向量)，作为模型的输入来进行各种处理。
实现方式:
one-hot编码:
稀疏词向量表示法
独热编码，01编码，有这个词就用1表示，没有这个词就用0 表示.
列表长度= 文本切词去重后总长
word2vec:稠密词向量表示法
CBOW：连续词袋模式
SkipGram 跳字模式
word Embedding: 稠密词向量表示法
词嵌入表示法



one-hot编码:
优点:
操作简单，容易理解.
缺点:
完全割裂了词与词之间的关系，且在大语料数据集下，每个向量的长度过长，占用大量内存.
针对于缺点的解决方案:
采用稠密向量表示法，例如:word2vec，word embedding続点:
针对于缺点的解决方案:

"""

import jieba

from tensorflow.keras.preprocessing.text import Tokenizer  #导入keras中的（词汇映射器Tokenizer）

import joblib
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
#1.获取one-hot编码
def dm01_onehot_gen():
    #1.准备预料
    vocabs = {'胡光亮','埃维军','周杰伦','王力宏','陈奕迅','李宗盛','范冰冰'}
    #2.实例化，词汇映射器Tokenizer
    my_tokenizer = Tokenizer()
    #3.通过词汇映射器，在语序上进行调整
    my_tokenizer.fit_on_texts(vocabs)
    #4.打印word_index字典
    print(my_tokenizer.word_index)
    print("=="*10)
#5.对每个词进行onehot编码
    for vocab in vocabs:
        #创建长度
        zero_list = [0] * len(vocabs)
        #获取索引
        idx = my_tokenizer.word_index[vocab] - 1
        #对应位置修改为1
        zero_list[idx] = 1
        print(f'{vocab}的one-hot编码是:{zero_list}')

#6.保存词汇映射器
    joblib.dump(my_tokenizer,'./model/onehot_tokenizer.pkl')
    print("one-hot编码器保存成功")
# dm01_onehot_gen()

def use_onehot_gen():
    #1，加载训练好的词汇映射器
    my_tokenizer = joblib.load('./model/onehot_tokenizer.pkl')
    #2.加载打印好的词汇映射器的word_index字典，查看词汇和索引的映射关系
    print(my_tokenizer.word_index)
    #3.对指定词汇进行one-hot编码
    token = "周杰伦"
    #4.创建长度=（词汇和索引字典）长度猎鸟，元素都为0
    zero_list = [0] * len(my_tokenizer.word_index)
    #5.获取指定词汇在word_index的索引，索引从 1开始所以-1
    idx = my_tokenizer.word_index[token] - 1
    #6.修改对应位置元素为1
    zero_list[idx] = 1
    #7.打印
    print(f'{token}的one-hot编码为:{zero_list}')
# use_onehot_gen()

def simple_one_hot():
    vocabs = {'胡光亮', '埃维军', '周杰伦', '王力宏', '陈奕迅', '李宗盛', '范冰冰'}
    word2index = {vocab: i for i, vocab in enumerate(vocabs)}
    print(word2index)

    for vocab in vocabs:
        zero_list = [0] * len(vocabs)
        idx = word2index[vocab]
        zero_list[idx] = 1
        print(f'{vocab}的one-hot编码是：:{zero_list}')
simple_one_hot()