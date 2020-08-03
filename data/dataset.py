import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import gensim
from sklearn.model_selection import train_test_split


class dataset1():
    def __init__(self, opt, word_index_dict):
        self.opt = opt
        self.word_index_dict = word_index_dict

    # 返回该词的index，将每条记录转化成由index组成的list，判断其长度不足的补0
    def word2index(self, word):
        """将一个word转换成index"""
        if word in self.word_index_dict:
            return self.word_index_dict[word]
        else:
            return 0

    def sentence2index(self, sentence):
        """将一个句子转换成index的list，并截断或补零"""
        word_list = sentence.strip().split()
        index_list = list(map(self.word2index, word_list))
        len_sen = len(index_list)
        if len_sen < self.opt.fix_len:
            index_list = index_list + [0] * (self.opt.fix_len - len_sen)
        else:
            index_list = index_list[:self.opt.fix_len]
        return index_list

    def get_k_fold_data(self, k, i, X, y):  ###此过程主要是步骤（1）
        # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
        assert k > 1
        fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

        X_train, y_train = None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
            ##idx 为每组 valid
            X_part, y_part = X[idx, :], y[idx]
            if j == i:  ###第i折作valid
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
                y_train = torch.cat((y_train, y_part), dim=0)
        # print(X_train.size(),X_valid.size())
        return X_train, y_train, X_valid, y_valid

    def get_data(self, opt):
        f = open(opt.train_data_path)
        documents = f.readlines()
        sentence = []
        for words in documents:
            s = self.sentence2index(words)
            sentence.append(s)

        x = np.array(sentence)

        """取出标签"""
        y = [0] * opt.train_pos + [1] * opt.train_neg
        y = np.array(y)
        return x, y

    # 划分数据集
    def get_splite_data(self, opt):
        f = open(opt.train_data_path)
        documents = f.readlines()
        sentence = []
        for words in documents:
            s = self.sentence2index(words)
            sentence.append(s)

        x = np.array(sentence)

        """取出标签"""
        y = [0] * opt.train_pos + [1] * opt.train_neg
        y = np.array(y)

        train_x, val_x, train_y, val_y = train_test_split(
            x, y, test_size=0.1, random_state=0)

        train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

        train_loader = DataLoader(train_data, shuffle=True, batch_size=opt.batch_size)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=opt.batch_size)

        return train_loader, valid_loader

    # 划分数据集2
    def get_splite_data2(self, opt):
        f = open(opt.train_data_path)
        documents = f.readlines()
        sentence = []
        for words in documents:
            s = self.sentence2index(words)
            sentence.append(s)

        x = np.array(sentence)

        """取出标签"""
        y = [0] * opt.train_pos + [1] * opt.train_neg
        y = np.array(y)

        l = []
        for i in range(len(y)):
            l.append((x[i], y[i]))

        total = opt.train_pos + opt.train_neg

        train_dataset, test_dataset = torch.utils.data.random_split(l, [int(total * 0.9), int(total * 0.1)])
        train_data = DataLoader(train_dataset, opt.batch_size, True)
        test_data = DataLoader(test_dataset, opt.batch_size, True)

        return train_data, test_data

    # 获得训练集
    def get_trainset(self, opt):
        f = open(opt.train_data_path)
        documents = f.readlines()
        sentence = []
        for words in documents:
            s = self.sentence2index(words)
            sentence.append(s)

        x = np.array(sentence)

        """取出标签"""
        y = [0] * opt.train_pos + [1] * opt.train_neg
        y = np.array(y)

        train_data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=opt.batch_size)

        return train_loader

    # 获得测试集
    def get_testset(self, opt):
        f = open(opt.test_data_path)
        documents = f.readlines()
        sentence = []
        for words in documents:
            s = self.sentence2index(words)
            sentence.append(s)

        x = np.array(sentence)

        """取出标签"""
        y = [0] * opt.test_pos + [1] * opt.test_neg
        y = np.array(y)

        test_data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        test_loader = DataLoader(test_data, shuffle=True, batch_size=opt.batch_size)

        return test_loader
