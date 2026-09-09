"""
演示 Word2Vec 之 CBOW(连续词袋模式)的完整数值流程.
对应课件 图3: Word2Vec之CBOW(连续词袋模式)图解.

两种模式回顾:
    CBOW:    上下文预测中间目标, 适合高频词.
    SkipGram: 中间预测上下文目标, 适合低频词.
    结论: 训练完成后, 把模型的权重参数当做词向量表示(稠密词向量).

流程(与课件一致):
    输入层 -> 隐藏层 -> 输出层(得到预测值) -> 训练闭环(计算损失, 反向传播)

关键结论(本代码用数值验证):
    1. one-hot 向量乘权重矩阵 == 直接取权重矩阵中对应的列, 这就是取词向量的过程.
    2. 输入->隐藏权重矩阵 W_in 形状 (N, V): N=隐藏层神经元数(本例3), V=词表大小(本例5).
    3. 隐藏->输出权重矩阵 W_out 形状 (V, N): 输出层有V个神经元, 每个神经元N个权重.
    4. 输出层经 softmax 得到每个词的预测概率, 与真实值求损失, 反向传播更新权重.
    5. 课件图中隐藏层用的是"相加", 标准实现一般取"平均", 两者只差一个缩放因子.
"""

import numpy as np

# 语料库: Hope can set you free (愿你自由成长), 去重后词表大小 V=5
vocab = ["Hope", "can", "set", "you", "free"]
V = len(vocab)
N = 3  # 隐藏层神经元个数
word2index = {w: i for i, w in enumerate(vocab)}


def one_hot(i):
    """把词表中的第i个词转成one-hot编码, 长度=词表大小."""
    vec = np.zeros(V)
    vec[i] = 1
    return vec


def softmax(x):
    """把logits转成概率分布, 所有分量之和为1."""
    e = np.exp(x - x.max())  # 减最大值防止指数溢出
    return e / e.sum()


def make_weights(seed=42):
    """随机初始化两个权重矩阵: W_in(N,V) 输入->隐藏, W_out(V,N) 隐藏->输出."""
    rng = np.random.default_rng(seed)
    W_in = rng.normal(0, 0.5, (N, V))
    W_out = rng.normal(0, 0.5, (V, N))
    return W_in, W_out


# 训练样本: 全语料所有窗口 (上下文两个词 -> 中间词)
# ([Hope, set] -> can), ([can, you] -> set), ([set, free] -> you)
windows = [([0, 2], 1), ([1, 3], 2), ([2, 4], 3)]


def dm01_forward_detail():
    """第1次操作详细数值: 基于Hope和set(输入) 预测can(输出), 逐步打印与课件图对应."""
    print("=" * 60)
    print("dm01: CBOW 单次前向传播详细数值 (对应课件图)")
    print("=" * 60)
    W_in, W_out = make_weights()
    np.set_printoptions(precision=3, suppress=True)

    # 1.输入层: 上下文词的one-hot编码
    print("1.输入层 one-hot编码:")
    print("   Hope =", one_hot(0).astype(int), " set =", one_hot(2).astype(int),
          " 目标can =", one_hot(1).astype(int))

    # 2.隐藏层: one-hot x W_in 等价于直接取W_in的对应列, 再相加
    print("2.隐藏层: one-hot x W_in == 取W_in对应列, 再相加:")
    print("   Hope取第1列:", W_in[:, 0])
    print("   set 取第3列:", W_in[:, 2])
    h = W_in[:, 0] + W_in[:, 2]
    print("   相加得隐藏层h =", h, f"(3维, 对应{N}个神经元)")

    # 3.输出层: h x W_out 得到V个logits, 再经softmax
    logits = W_out @ h
    y_hat = softmax(logits)
    print("3.输出层: h x W_out(5x3) -> 5个logits, 再经softmax:")
    print("   logits =", logits)
    print("   y_hat  =", y_hat)
    print("   当前预测:", vocab[int(y_hat.argmax())], " 真实目标: can")


def dm02_train_loop(epochs=500, lr=0.05):
    """训练闭环: 对全部窗口迭代, 前向算预测 -> 算损失 -> 反向传播更新两个权重矩阵."""
    print("=" * 60)
    print("dm02: 训练闭环 (交叉熵损失 + 反向传播, 全部窗口迭代)")
    print("=" * 60)
    W_in, W_out = make_weights()

    for epoch in range(1, epochs + 1):
        loss = 0.0
        for ctx, target in windows:
            # 前向: 取上下文各词的列并相加 -> 隐藏层 -> logits -> 概率
            h = W_in[:, ctx[0]] + W_in[:, ctx[1]]
            y_hat = softmax(W_out @ h)
            loss += -np.log(y_hat[target] + 1e-12)

            # 反向: softmax+交叉熵合并求梯度, 再链式传回两层权重
            dlogits = y_hat.copy()
            dlogits[target] -= 1
            dW_out = np.outer(dlogits, h)
            dh = W_out.T @ dlogits
            dW_in = np.outer(dh, one_hot(ctx[0])) + np.outer(dh, one_hot(ctx[1]))
            W_in -= lr * dW_in
            W_out -= lr * dW_out
        if epoch in (1, 50, 100, 200, 500):
            print(f"   轮次{epoch:<4} 平均loss = {loss / len(windows):.4f}")

    # 用训练好的权重跑一次课件图中的原始任务: Hope+set -> 预测can
    h = W_in[:, 0] + W_in[:, 2]
    y_hat = softmax(W_out @ h)
    print("   训练后验证: 输入[Hope, set], 各词预测概率:")
    for i, p in enumerate(y_hat):
        flag = "  <- 命中" if i == int(y_hat.argmax()) else ""
        print(f"   P({vocab[i]:<5}) = {p:.4f}{flag}")
    print("   最终预测:", vocab[int(y_hat.argmax())])
    return W_in, W_out


def dm03_word_vectors(W_in):
    """结论验证: 训练产出就是词向量. W_in每一列即对应词的N维稠密词向量."""
    print("=" * 60)
    print("dm03: 提取词向量 (W_in 每一列 = 对应词的词向量)")
    print("=" * 60)
    np.set_printoptions(precision=3, suppress=True)
    for i, w in enumerate(vocab):
        print(f"   {w:<5} -> {W_in[:, i]}")

    # 词向量最直观的用途: 余弦相似度, 值越接近1越相似
    def cos(a, b):
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    print("   词向量间余弦相似度(越接近1越相似):")
    header = "        " + "".join(f"{w:>8}" for w in vocab)
    print(header)
    for i, w1 in enumerate(vocab):
        row = f"   {w1:<5}"
        for j, w2 in enumerate(vocab):
            row += f"{cos(W_in[:, i], W_in[:, j]):>8.2f}"
        print(row)


if __name__ == "__main__":
    dm01_forward_detail()
    print()
    W_in, W_out = dm02_train_loop()
    print()
    dm03_word_vectors(W_in)
