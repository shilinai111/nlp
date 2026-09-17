from sympy.physics.units import gram

from day01.day01_词性标注 import result

#连续n个词/字，吧连续片段当作一种特征（小词组特征），帮我们分析文本
#分类：
    #uni-gram(1-gram):把每个词/句子拆出来
    #bi-gram(2-gram):把连续2个词组合
    #tri-gram(3-gram):把连续3个词组合
#目的：
    #让计算机更好的理解文本规律

#1.记录n的值，一般 n-gram中的n取2或者3
ngram_range = 2
#2.生成n-gram特征
def create_ngram(input_list):
    scline_lists = [input_list[i:]for i in range(ngram_range)]
    ngram_tuples = zip(*scline_lists)
    return set(ngram_tuples)
if __name__ == '__main__':
    input_list=[1,3,2,1,5,3]

    result = create_ngram(input_list)

    print(result)