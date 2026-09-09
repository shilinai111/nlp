
"""
K例:
演示jieba 分库的应用.
分词的相关介绍:
分词过程=找到 分界符 的过程，找到分界符就可以分词了。
每个分河结果=1个Token
常见的分河包:
jieba:精确模式，全模式，搜索引模式，繁体字符，用户白定义词典...
IK分词器:ElasticSearch搜索引用的多.
SnowNLP:基于概率算法的中文白然语言处理工具包
pyltp:哈工大的
THULAC:清华的...
jieba的作用介绍:
A.支持多种分模式
精确模式:试图将句子最精确的切开，适合:文本分析.
全模式:把句子中所有的可以成词的词语都扫描出来，速度非常快，但是不能消除歧义。
搜索引擎模式:在精确模式的基础上，对长词再次切分，提高召回率。
B.支持中文繁体分河
C.支持用户白定义词典
"""

import jieba

#精确模式，文本分析
def demo1():
    content = "阿俊在上课路上拍了一张阿亮走路的照片，并把照片动漫化，附文小黑子"
    result1 = jieba.cut(content, cut_all=False)#生成器，精确模式
    print(f'result:{result1}')
    # print(next(result1))
    # print(next(result1))
    # print("="*10)
    #
    # for item in result1:
    #     print(item)
    # print("=" * 10)

    #列表实现
    list1 = list(result1)
    print(f'list1:{list1}')

    #思路2
    list2 = jieba.lcut(content, cut_all=False)#精确模式
    print(f'list2:{list2}')

#全模式,关键词提取
def demo2():
    content = "阿俊在上课路上拍了一张阿亮走路的照片，并把照片动漫化，附文小黑子"
    result1 = jieba.cut(content, cut_all=True)#生成器，全模式
    print(f'result:{result1}')
    # print(next(result1))
    # print(next(result1))
    # print("="*10)
    #
    # for item in result1:
    #     print(item)
    # print("=" * 10)

    #列表实现
    list1 = list(result1)
    print(f'list1:{list1}')

    #思路2
    list2 = jieba.lcut(content, cut_all=True)#全模式
    print(f'list2:{list2}')

#jieba搜索引擎
"""
解释:
搜索引分词模式 一在精确模式分词的基础上，对长词进行再次切分，提高召回率，
例如:
场景1:用户录入“程序员”
精确模式:
只能配包含完整“程序员”的文档。
搜索引擎模式:不仅能匹配“程序员”文档，还能匹配“程序”，“员”的文档，提高召回。
场景2:实际应用场景(电商搜索)，商品标题为:《苹果手机保护套>，用户搜索 《苹果套>>
无法匹配，分词为:(“苹果"，"手机”，"保护套”)
精确模式:
搜索引擎模式:能匹配，分词为:("苹果”，“手机”，"保护”，"套")
"""
def demo3():
    content = "阿俊在上课路上拍了一张阿亮走路的照片，并把照片动漫化，附文小黑子"
    result1 = jieba.cut_for_search(content)  # 生成器，全模式
    print(f'result:{result1}')
    # print(next(result1))
    # print(next(result1))
    # print("="*10)
    #
    # for item in result1:
    #     print(item)
    # print("=" * 10)

    # 列表实现
    list1 = list(result1)
    print(f'list1:{list1}')

    # 思路2
    list2 = jieba.lcut_for_search(content)  # 全模式
    print(f'list2(搜索引擎模式):{list2}')

    list3 = jieba.lcut(content,cut_all=False)
    print(f'list3(精确模式):{list3}')

    list4 = jieba.lcut(content,cut_all=True)
    print(f'list4(全模式):{list4}')

#繁体字分词
def demo4():
    content = "煩惱即是菩提，我暂且不提"

    result1 = jieba.lcut(content, cut_all=False)
    print(f'result1:{result1}')

#自定义词典，适用于生僻词，专业词语
"""
添加自定义词典后，jieba能够准确识别出词典中出现的词汇，提升整体的识别准确率.
词典(文件的)格式，每一行分3个部分，分别是:
词语(必选)词频(可省略)词性(可省略)         中间用空格隔开，顺序不能颠倒
"""
def demo5():
    content = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    list1 = jieba.lcut(content)
    print(f'list1(精确分词):{list1}')

    jieba.load_userdict('./data/userdict.txt')
    list2 = jieba.lcut(content)
    print(f"list(加载自定义词典):{list2})")


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    # demo4()
    demo5()