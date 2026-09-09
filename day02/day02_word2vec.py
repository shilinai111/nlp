import fasttext


#1.训练向量模型,并保存模型
def dm01_train_save():
    #1.以无监督模式开始运行
    my_model = fasttext.train_unsupervised('./data/wh02ad')
    #2.保存模型为2进制文件，后续=通过fasttext.load_model()加载模型
    my_model.save_model("./model/wh02_fil9.bin")
    print("训练完毕，模型保存成功..")


#2加载模型，并预测
def dm02_get_word_vector():
    #1加载预训练的fasttext模型
    model = fasttext.load_model('./model/wh02_fil9.bin')
    #2.获取单个词的词向量表示
    results = model.get_word_vector('夯哥')
    #3.打印
    print(f'type:{type(results)}')
    print(f'shape:{results.shape}')
    print(f'result:{results}')

def dm03_get_similarity():
    # 1加载预训练的fasttext模型
    model = fasttext.load_model('./model/wh02_fil9.bin')
    #2.查找某个单词的近义词
    #默认是十个，可以用于检验模型的语义理解能力
    #返回格式是[(相似度分数,近义词),(相似度分数,近义词)]
    results = model.get_nearest_neighbors('dog')
    #3.输出
    print(f'results:{(results)}')

def dm04_set_parament():
    #设置模型参数
    my_model = fasttext.train_unsupervised(
        input='./data/wh02ad',  #训练数据的路径
        model = 'cbow',   #词向量模型：cbow（有两边猜中间），skipgram（有中间猜两边）
        dim = 50,   #词向量维度
        epoch = 1,   #训练轮数
        lr = 0.01,   #学习率
        thread = 10   #线程数
    )
    #保存模型-》二进制文件
    my_model.save_model('./model/wh02_fil9_new.bin')
    print("训练完毕，模型保存成功")

if __name__ == '__main__':
    # dm01_train_save()
    # dm02_get_word_vector()
    # dm03_get_similarity()
    dm04_set_parament()