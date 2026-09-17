"""
案例:
演示文件张量实现方式之 word Embedding.
大白话解释下:word2vec 和 word Embedding的区别
    word2vec:
       先办证，后干活.会预先训练处词向量(模型)，后续再带入模型做其它
    word Embedding:
        边办证，边干活。训练过程中，词向量会自动生成. 就是以前我们
# 禁用oneDNN优化包...



细节:如果你要看TensorBoard的可视化界面(本质就是一个PCA主成分分析图)，代码写完后，按照如下操作即可.
step1:切换到nlpbase沙箱step3:运行如下的命令即可
step2:切换到当前项目(这里是day02)的路径下，即:有runs文件夹的那个目录下.
    tensorboard --logdir=runs --host 0.0.0.0
step4:通过localhost:6006访问即可.
注意:第一次加载会慢一点


"""


import torch
from tensorflow.keras.preprocessing.text import Tokenizer
import jieba
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


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

    #4.构建词汇表，进行文本数值化(词向量)
    #4.1初始化词汇映射器
    my_tokenizer = Tokenizer()
    #4.2拟合训练数据，统计词频，并构建：词汇表
    my_tokenizer.fit_on_texts(word_list)
    #4.3查看词和索引的映射关系
    print(f'词和索引的映射关系:{my_tokenizer.word_index}')
    #4.4.获取去重后的所有词汇列表
    my_token_list = my_tokenizer.word_index.values()
    print(my_token_list)
    #4.5.将分词后的文本->转成数字序列
    seq2id = my_tokenizer.texts_to_sequences(word_list)
    print(f"文本转成数字序列：:{seq2id}")
    #5.创建词嵌入层，把文本（即：词对应的编号）转成词向量
    #5.1创建词嵌入层对象
    #参1:词汇表大小，即:唯一的词的个数
    #参2:词向量的维度
    embed = nn.Embedding(len(my_token_list), 8)
    #5.2查看词嵌入层的权重参数（即：词向量）
    print(f'embed:{embed.weight.data}')
    print(f'embed.shape:{embed.weight.shape}')
    #6.创建TensorBoard写入器，将数据写入runs目录
    my_summary = SummaryWriter(log_dir='./runs')
    #6.2讲词向量和对应词语添加到TensorBoard中
    #参1:词向量矩阵，形状是:（20，8），20歌个词，每个词用8个数字表示（8维）
    #参2:对应的词语列表，用来标注每个点
    my_summary.add_embedding(embed.weight.data,my_token_list)
    #6.3关闭写入器
    my_summary.close()

    #7.查看每个单词对应的词向量
    for idx in range(len(my_tokenizer.word_index)):
        #7.1获取当前单词对应的词向量
        temp_vector = embed(torch.tensor(idx))

        #7.2获取当前索引对应的单词，my_tokenizer,index_word
        # print(my_tokenizer.word_index)
        # print(my_tokenizer.index_word)
        word = my_tokenizer.index_word[idx+1]
        print(f'单词:{word},词向量:{temp_vector.detach().numpy()}')

if __name__ == "__main__":
    dm01_embedding_show()