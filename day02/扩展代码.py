from itertools import chain
import pandas as pd
import jieba


def fun(x):
    return x+2

def dm01_map():
    #1.将上述fun()函数，作用到:列表每个元素上
    # map():返回1个迭代对象，此时并未实际运算
    result = map(fun,[1,2,3,4,5])
    print(f'result:{result}')

    #第一次遍历map对象
    for i in result:
        print(i)
    print('--'*20)

    #3.第二次尝试遍历map对象
    print(f'result:{list(result)}')

    #4.使用lambda表达式实现 相同功能
    result2 = list(map(lambda x:x+2,[10,20,30]))
    print(f'result2:{result2}')

def dm02_chain():
    #1.定义两个列表
    list1,list2 = [1,2,3],[2,3,4]
    #2.chain()：是惰性的。创建了chain()对象。不会立即遍历。，只有再遍历是才会读取数据
    #一旦遍历完毕再次返回同一个chain()对象，不会得到数据，因为迭代器已经耗尽
    result = chain(list1,list2)
    print(f'result:{result}')
    print(f'result:{list(result)}')

    #chain()函数，有点类似鱼list.extend(list2)
    list1.extend(list2)
    print(f'result2:{list1}')

    # #重新定义列表
    # list1 = ['今天天气很好','今天天气很热']
    # #对上述句子进行切词
    # tmp = map(lambda x:jieba.lcut(x),list1)
    # #print(f'tmp:{tmp}')
    # print(f'tmp:{list(tmp)}')
    #
    # #用chain（）函数，连接两个列表
    # # result = list(chain(*tmp))
    # result = set(chain(*tmp))
    # print(f'result:{result}')
    # print(f'tmp:{list(result)}')

    #合并
    list1 = ['今天天气很好','今天天气很热']
    result = set(chain(*map(lambda x : jieba.lcut(x),list1)))
    print(f'result2:{result}')

#获取测试集训练集的词汇总数
def dm03_get_word_count():
    train_data = pd.read_csv('data/train.tsv',sep='\t')
    dev_data = pd.read_csv('data/dev.tsv',sep='\t')

    #统计训练集的词汇总数
    train_vocab = set(chain(*map(lambda x :jieba.lcut(x),train_data['sentence'])))
    print(f'训练集共包含不同词汇总数为:{len(train_vocab)}')

    #3.统计测试集的词汇总数
    dev_vocab = set(chain(*map(lambda x :jieba.lcut(x),dev_data['sentence'])))
    print(f'测试集包括不同的词汇总数为L{len(dev_vocab)}')

#zip()函数入门，功能：合并迭代对象
def dm04_zip():
    #定义列表
    list1 = [1,2,3,4,5,6,7,8]
    list2 = [2,3,4]

    #使用zip()函数，合并迭代对象
    result = zip(list1,list2)
    print(f'result:{result}')

    #把握上述zip对象，转成列表，查看内容
    print(list(result))

    #演示zip的解包用法(*)
    #定义嵌套列表（二维）
    list3 = [[1,2,3],[3,4,5]]
    #不使用解包的zip调用，会把list3当作单个迭代对象处理
    print(list(zip(list3)))
    #使用解包的zip调用，会把list3中的元素，逐个取出
    print(list(zip(*list3)))

if __name__ == '__main__':
    # dm01_map()
    # dm02_chain()
    dm04_zip()