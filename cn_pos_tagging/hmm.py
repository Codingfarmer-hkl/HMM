# coding=utf-8
import math
import random


# 本代码的log是把连城转成相加

class HMM:
    def __init__(self, train_file_path):
        self.file_path = train_file_path  # 读取文件的文件目录
        self.pi = {}  # 初始状态概率分布
        self.A = {}  # 状态转移概率矩阵
        self.B = {}  # 观察值概率矩阵
        self.pos = []  # 所有词性
        self.tag_fre = {}  # 词性出现频率
        self.A_class = {}  # 状态转移平滑处理矩阵
        self.B_class = {}  # 观察概率平滑处理矩阵

    def build_hmm(self):
        all_words = set()  # 这里保存所有的词，这里的set函数有去重功能
        with open(self.file_path, 'r', encoding='utf-8') as f:
            all_sentence = f.readlines()  # 读取所有的句子
        line_counts = len(all_sentence)  # 统计句子的条数
        for sen in all_sentence:
            sen = sen.rstrip('\n')
            word_count = sen.split(" ")  # 把词按空格分开
            for word_with_tag in word_count:
                words = word_with_tag.split('/')  # 按/分开词和词性
                if len(words) == 2:  # 这里过滤掉未标注的词
                    all_words.add(words[0])  # 记录所有词
                    if words[1] not in self.pos:
                        self.pos.append(words[1])  # 记录所有词性
        # 以上只是统计了所有词(有多少中词)和所有词性(有多少中词性)
        print("first step done")
        print(len(self.pos))

        # 初始化转换矩阵和发射矩阵，初始化矩阵
        for tag in self.pos:
            self.pi[tag] = 0  # 这里是初始化概率矩阵
            self.tag_fre[tag] = 0  # 此处统计每个词性的频数
            self.A[tag] = {}  # 状态转移概率矩阵
            self.B[tag] = {}  # 发射概率矩阵
            for next_tag in self.pos:
                self.A[tag][next_tag] = 0  # tag 转next_tag的概率初始化
            for word in all_words:
                self.B[tag][word] = 0  # tag转词的概率初始化

        # 计算A,B矩阵和初始化矩阵的概率
        for sen in all_sentence:
            sen = sen.rstrip('\n')
            tmp_word = sen.split(" ")
            counts = len(tmp_word)
            head_word = tmp_word[0].split('/')  # 这里只取第一个词，为了统计初始概率矩阵
            self.pi[head_word[1]] += 1  # 计算开始出现词性的频数
            self.tag_fre[head_word[1]] += 1  ## 统计词性的频数
            self.B[head_word[1]][head_word[0]] += 1  # 计算词性转词的概率
            for i in range(1, counts):  # 计算除了开始位置以外词的矩阵
                current_word = tmp_word[i].split('/')
                pre_word = tmp_word[i - 1].split('/')
                if len(current_word) == 2:
                    if current_word[-1] and pre_word[-1]:
                        self.tag_fre[current_word[-1]] += 1  ##计算词性的频数
                        self.A[pre_word[-1]][current_word[-1]] += 1  # 计算词性转词性的概率
                        self.B[current_word[-1]][current_word[0]] += 1  ## 计算词性转词的概率

        print("second step done")

        # 矩阵中零太多，做平滑处理
        for tag in self.pos:
            self.A_class[tag] = 0  # 统计计算转换矩阵为0 的个数，这里的0的个数作为计算概率分母的一部分
            self.B_class[tag] = 0  # 统计词性转词矩阵为0 的个数
            if self.pi[tag] == 0:
                self.pi[tag] = 0.5 / line_counts  # 这里的0.5是代表频数，line_counts表示总数据数
            else:
                self.pi[tag] = self.pi[tag] * 1.0 / line_counts  # tag的频率处理总数据条数，获取pi 的初始概率
            for next_tag in self.pos:
                if self.A[tag][next_tag] == 0:
                    self.A_class[tag] += 1  # 如果转态转移矩阵中为0的统计一下
                    self.A[tag][next_tag] = 0.5  # 未出现状态转化给0.5个频数 ,用于平滑使用
            for word in all_words:
                if self.B[tag][word] == 0:
                    self.B_class[tag] += 1  # 如果发射转移矩阵中为0的统计一下
                    self.B[tag][word] = 0.5  # 未出现状态转化给0.5个频数 ,用于平滑使用
        # 以上已经给self.pi, self.A, self.B做了平滑
        for tag in self.pos:
            for next_tag in self.pos:
                # 计算每一行的转化概率，self.tag_fre[tag] + self.A_class[tag]这里既考虑文本统计的tag频数也考虑平滑加进去的频数
                # self.A[tag][next_tag]表示tag转向next_tag的频数
                self.A[tag][next_tag] = self.A[tag][next_tag] * 1.0 / (
                            self.tag_fre[tag] + self.A_class[tag])  # 表示tag转向next_tag的频数 除以文本tag频数和平滑tag频数和
            for word in all_words:
                # 这个和计算self.A原理一样
                self.B[tag][word] = self.B[tag][word] * 1.0 / (self.tag_fre[tag] + self.B_class[tag])

        print("build done")

    # 以上已经统计好self.pi, self.A， self.B的概率矩阵，一下是预测
    def predict_pos_tags(self, test_file_path, truth_file_path):
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_sentence = [f.readlines()[0]]  # 读预测数据

        with open(truth_file_path, 'r', encoding='utf-8') as f:
            truth_tags = [f.readlines()[0]]  # 这是读预测数据真是的标注数据，方便下面做准确率

        with open('./data/result.txt', 'w', encoding='utf-8') as f:
            total_accu = 0
            for ind in range(len(test_sentence)):
                res = []
                sen = test_sentence[ind]
                sen = sen.rstrip('\n').split(" ")
                sen = sen[:-1]
                sen_length = len(sen)  # 获取数据的长度
                delta = [{} for i in range(sen_length)]  # 存放计算的概率
                psi = [{} for i in range(sen_length)]  # 存放解码的路径
                # delta,psi进行初始化
                for tag in self.pos:
                    for index in range(sen_length):
                        delta[index][tag] = -1e100
                        psi[index][tag] = ""

                for tag in self.pos:
                    if sen[0] in self.B[tag]:
                        delta[0][tag] = math.log(self.pi[tag] * self.B[tag][sen[0]])  # pi * B 开始计算第一个词的词性概率
                    else:
                        #  如果词不在训练词库里使用平滑数值计算，(self.B_class[tag] + self.tag_fre[tag])词的总数，即没有记录转向哪，则转向所有词的概率一样
                        delta[0][tag] = math.log(self.pi[tag] * 0.5 / (self.B_class[tag] + self.tag_fre[tag]))

                for i in range(1, sen_length):  # i 计算单前位置的词性
                    for tag in self.pos:  # tag是计算当前为tag词性的概率
                        if sen[i] in self.B[tag]:
                            tmp = math.log(self.B[tag][sen[i]])
                        else:
                            # 词不在词库里
                            tmp = math.log(0.5 / (self.B_class[tag] + self.tag_fre[
                                tag]))  # # (self.B_class[tag] + self.tag_fre[tag])未转换的词（转换率为0的词） + 词的总数，即没有记录转向哪，则转向所有词的概率一样\
                        for pre_tag in self.pos:  # 计算前一个tag转为当前tag的概率
                            if delta[i][tag] < (
                                    delta[i - 1][pre_tag] + math.log(self.A[pre_tag][tag]) + tmp):  # vitebi算法只保留最大概率
                                delta[i][tag] = delta[i - 1][pre_tag] + math.log(
                                    self.A[pre_tag][tag]) + tmp  # 保存某一隐状态的最大值
                                psi[i][tag] = pre_tag  # 保存最大状态的前一状态和当前状态,格式为{‘前一个转态’：‘当前转态’}

                # 解码，从最后的最大概率出发
                max_end = self.pos[0]  # 随便取一个词性
                # 这里判断最后一个字的最大概率词性
                for tag in self.pos:
                    if delta[sen_length - 1][max_end] < delta[sen_length - 1][tag]:
                        max_end = tag

                i = sen_length - 1
                res.append(max_end)
                while i > 0:
                    max_end = psi[i][max_end]  # 通过最后一个状态遍历状态字典{psi}
                    res.append(max_end)
                    i -= 1
                res.reverse()  # 标注好的做各翻转，解码的时候最后一个字的词性放在第一个位置了

                # 以下就是计算准确率了
                truth = truth_tags[ind].rstrip("\n").split(" ")
                truth = truth[:-1]
                if len(truth) != len(res):
                    print("预测与结果词数不符，此句有误")
                    continue

                word_count = len(truth)
                correct_num = 0

                # 计算准确率
                for j in range(word_count):
                    if truth[j] == res[j]:
                        correct_num += 1

                accu = correct_num / word_count

                f.write('expected: ' + ' '.join(truth) + '\n')
                f.write('got: ' + ' '.join(res) + '\n')
                f.write('accu: ' + str(accu) + '\n\n')
                f.flush()

                print("第%d个句子准确率为：%f" % (ind + 1, accu))

                total_accu += accu

            average_accu = total_accu / len(test_sentence)
            print("平均准确率为：%f" % average_accu)
            f.write("平均准确率为：" + str(average_accu))


if __name__ == '__main__':
    hmm = HMM('./data/simple_train_raw_data.txt')
    hmm.build_hmm()
    hmm.predict_pos_tags('./data/simple_test_words_data.txt', './data/simple_test_tags_data.txt')
