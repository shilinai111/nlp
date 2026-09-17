"""
案例:
    演示 文本分析的常见操作.
文本分析的作用:
    帮助我们理解数据语料，快速检查出语料中可能存在的问题.
    例如:
        数据质量类:错别字，语法错误，重复内容，缺失值，噪声数据...
        分布不均衡问题:标签分布不均，句子长度不同...
文本分析的方式:
    1.标签的数量分布
    2.句子长度分布
    3.词频统计和关键字词云
"""

import jieba                    #分词包
import seaborn as sns           #画图包
import pandas as pd             #数据处理
import matplotlib.pyplot as plt #画图
from itertools import chain     #迭代器工具
import jieba.posseg as pseg     #词性标注包  动词，名词等
from wordcloud import WordCloud #词云包

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#1.实现:训练集和测试集的标签分布的可视化统计
def dm01_lable_sns_countplot():
    #1.设置538风格->一种具有现代感的可视化风格(可选项)
    plt.style.use('fivethirtyeight')
    #2.读取训练集和测试集
    #参1:文件路径  参2:列分割符(csv文件用,tsv文件用\t)
    train_data = pd.read_csv('data/train.tsv',sep='\t')
    dev_data = pd.read_csv('data/dev.tsv',sep='\t')
    # print(train_data.head())
    #3.统计训练集标签的0(负样本)1(正样本)的数量，并可视化，采用技术柱状图
    #参1:x轴标签   参2:数据集   参3:用于分组的分类变量   参4:是否显示图例
    sns.countplot(x='label',data=train_data,hue='label',legend=False)
    plt.title('train_label')
    #紧凑布局
    plt.tight_layout()
    plt.show()

def dm02_len_sns_distplot():
    #1.读取训练集和测试集
    train_data = pd.read_csv('data/train.tsv',sep='\t')
    dev_data = pd.read_csv('data/dev.tsv',sep='\t')
    #2.计算训练集的每个句子的长度
    #思路1:map() 函数的方式实现
    train_data['sentence_length'] = list(map(lambda x :len(x),train_data['sentence']))

    #思路2:apply()函数的方式实现
    #train_data['sentence_length'] = train_data['sentence'].apply(lambda x : len(x))

    #3.绘制训练集的句子长度分布
    #图1:计数柱状图
    sns.countplot(x = 'sentence_length',data=train_data)
    plt.title('训练集句子长度分布—计数柱状图')
    plt.xticks([])
    plt.show()
    #图2:密度曲线图
    sns.histplot(x='sentence_length',data=train_data,kde=True)
    plt.title('训练集句子长度分布-密度曲线图')
    plt.show()
    #4.计算测试集的每个句子长度
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))
    # 图1:计数柱状图
    sns.countplot(x='sentence_length', data=dev_data)
    plt.title('训练集句子长度分布—计数柱状图')
    plt.xticks([])
    plt.show()
    # 图2:密度曲线图
    sns.histplot(x='sentence_length', data=dev_data, kde=True)
    plt.title('训练集句子长度分布-密度曲线图')
    plt.show()
#3. 实现训练集和测试集的正负样本长度散点分布
def dm03_sns_stripplot():
    train_data = pd.read_csv('data/train.tsv',sep='\t')
    dev_data = pd.read_csv('data/dev.tsv',sep='\t')

    #获取训练集数据长度列
    train_data['sentence_length'] = list(map(lambda x:len(x),train_data['sentence']))
    #获取测试集数据长度列
    dev_data['sentence_length'] = list(map(lambda x:len(x),dev_data['sentence']))

    #统计正负样本长度的散点分布
    #训练集
    #参1：x轴标签，参2：y轴标签，参3：数据集,参4：分组，不同组颜色不一样
    sns.stripplot(x='label',y = 'sentence_length',data=train_data,hue='label')
    plt.title('训练集正负样本长度散点分部')
    plt.show()

    sns.stripplot(x='label', y='sentence_length', data=dev_data)
    plt.title('测试集正负样本长度散点分部')
    plt.show()

#训练集和测试集的词汇总数
def dm04_get_word_count():
    #读取训练集和测试集
    train_data = pd.read_csv('data/train.tsv',sep='\t')
    dev_data = pd.read_csv('data/dev.tsv',sep='\t')
    #统计训练集的词汇总数
    train_vocab =set(chain(*map(lambda x: jieba.lcut(x), train_data['sentence'])))
    print(f'训练集包含不同词汇的总数是:{len(train_vocab)}')

    #统计测试集的词汇总数(去重后)
    dev_vocab = set(chain(*map(lambda x: jieba.lcut(x), dev_data['sentence'])))
    print(f'测试集共包含不同词汇总数为:{len(dev_vocab)}')

def dm05_a_list(text):
    #定义空列表用于存储形容词列表
    a_list = []

    #使用jieba的词性标注功能，切分文本，并获取到每个词
    for value in pseg.lcut(text):
        # print(f'value:{value}')
        # print(f'value.word:{value.word}')
        # print(f'value.flag:{value.flag}')
        if value.flag == 'a':
            a_list.append(value.word)
        return a_list

#根据词列表和=产生词云图
def dm05_get_word_cloud(keywords_list):
    #实例化词云生成器
    wordcloud = WordCloud(
        font_path='data/SimHei.ttf',     #字体路径
        max_words=100,                   #最大显示数
        background_color='white',        #背景色
    )
    #将关键词列表转换为空格分割的字符串，适配：词云输入格式
    keyword_str = ' '.join(keywords_list)
    #根据关键词字符串,生成词云
    wordcloud.generate(keyword_str)

    #配置并显示词云图像
    #船舰会话窗口
    plt.figure()
    #生成词云
    #参1：生成词云图像数据  参2：设置图像插值方法为：双线性插值
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis("off")
    plt.show()
def dm05_word_cloud():
    train_data = pd.read_csv('data/train.tsv',sep='\t')

    #处理训练集的正样本（label = 1）-》生成词云
    #筛选lable=1
    p_train_data = train_data[train_data['label']==0]['sentence']
    #对每个正样本句子，提取形容词里列表，合并成一个完整的形容词列表
    p_a_train_vocab = chain(*map(lambda x: dm05_a_list(x), p_train_data))

    dm05_get_word_cloud(p_a_train_vocab)





from itertools import chain
def dm05_chain_show():
    b1 = [1, 2, 3]
    b2 = ['x', 'y', 'z']
    c1 = chain(b1, b2)
    print('c1--->', list(c1))
    # chain(b1,b2) → 先遍历b1，再遍历b2
    # 输出：c1---> [1,2,3,'x','y','z']

    a = [(1, 2, 3), ('x', 'y', 'z')]
    c2 = chain(a)
    print('c2--->', list(c2))
    # chain(a) 只接收1个参数a，a是一个包含2个元组的列表
    # 遍历a，取出a里面的两个元组本身：
    # 输出：c2---> [(1,2,3), ('x','y','z')]

    a = [(1, 2, 3), ('x', 'y', 'z')]
    c3 = chain(*a)
    print('c3--->', list(c3))
    # *a 解包，等价于 chain( (1,2,3), ('x','y','z') )
    # 遍历第一个元组，再遍历第二个元组
    # 输出：c3---> [1,2,3,'x','y','z']










if __name__ == '__main__':
    # dm01_lable_sns_countplot()
    # dm02_len_sns_distplot()
    # dm03_sns_stripplot()
    # dm04_get_word_count()
    # dm05_a_list("房间里有电脑，虽然房间的条件略显简陋，但环境、服务还有饭菜都还是很不错的。如果下次去无锡，我还是会选择这里的。")
    # dm05_word_cloud()
    dm05_chain_show()


