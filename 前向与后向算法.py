# -*- coding: UTF-8 -*-
import numpy as np


def Forward(trainsition_probability, emission_probability, pi, obs_seq):
    """
    :param trainsition_probability:trainsition_probability是状态转移矩阵
    :param emission_probability: emission_probability是发射矩阵
    :param pi: pi是初始状态概率
    :param obs_seq: obs_seq是观察状态序列
    :return: 返回结果
    """
    trainsition_probability = np.array(trainsition_probability)
    emission_probability = np.array(emission_probability)
    pi = np.array(pi)
    Row = np.array(trainsition_probability).shape[0]  # 计算有几个状态

    F = np.zeros((Row, Col))  # 最后要返回的就是F，就是我们公式中的alpha
    F[:, 0] = pi * np.transpose(emission_probability[:, obs_seq[0]])  # 这是初始化求第一列,就是初始的概率*各自的发射概率
    for t in range(1, len(obs_seq)):  # 这里相当于填矩阵的元素值，逐个逐个状态计算
        for n in range(Row):  # n是代表隐藏状态的个数，即每个状态都可能转向别的转态
            # np.dot(F[:, t - 1], trainsition_probability[:, n])先计算转态转移概率
            F[n, t] = np.dot(F[:, t - 1], trainsition_probability[:, n]) * emission_probability[
                n, obs_seq[t]]  # 对应于公式,前面是对应相乘
    return F


def Backward(trainsition_probability, emission_probability, pi, obs_seq):
    """
    :param trainsition_probability:trainsition_probability是状态转移矩阵
    :param emission_probability: emission_probability是发射矩阵
    :param pi: pi是初始状态概率
    :param obs_seq: obs_seq是观察状态序列
    :return: 返回结果
    """
    trainsition_probability = np.array(trainsition_probability)
    emission_probability = np.array(emission_probability)
    pi = np.array(pi)  # 要进行矩阵运算，先变为array类型

    Row = trainsition_probability.shape[0]  # 计算有几个状态
    Col = len(obs_seq)  # 观测状态的长度
    F = np.zeros((Row, Col))  # 最后要返回的就是F，就是我们公式中的alpha
    F[:, (Col - 1):] = 1  # 最后的每一个元素赋值为1

    for t in reversed(range(Col - 1)):  # 从后往前计算，这里不包括初始化概率及对应的发射概率
        for n in range(Row):
            F[n, t] = np.sum(F[:, t + 1] * trainsition_probability[n, :] * emission_probability[:, obs_seq[t + 1]])

    return F


if __name__ == '__main__':
    trainsition_probability = [[0.7, 0.3], [0.4, 0.6]]
    emission_probability = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]]
    pi = [0.6, 0.4]

    # 然后下面先得到前向算法,在A,B,pi参数已知的前提下，求出特定观察序列的概率是多少?
    obs_seq = [0, 1]
    Row = np.array(trainsition_probability).shape[0]
    Col = len(obs_seq)

    F = Forward(trainsition_probability, emission_probability, pi, obs_seq)  # 得到前向算法的结果
    F_backward = Backward(trainsition_probability, emission_probability, pi, obs_seq)  # 得到后向算法的结果

    # 前向的最终的概率计算
    res_forward = 0
    for i in range(Row):  # 将最后一列相加就得到了我们最终的结果
        res_forward += F[i][Col - 1]  # 求和于最后一列
    # 后向算法的概率计算
    emission_probability = np.array(emission_probability)
    # 下面是得到后向算法的结果
    res_backword = 0
    res_backward = np.sum(pi * F_backward[:, 0] * emission_probability[:, obs_seq[0]])  # 一定要乘以发射的那一列，这里的第一列相当于当前的状态
    print("res_backward = {}".format(res_backward))
    print("res_forward = {}".format(res_forward))
