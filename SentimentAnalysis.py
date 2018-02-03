# from creat_data import bat
import pandas as pd
import numpy as np
import os
from sentence_transform.creat_vocab_word2vec import creat_vocab_word2vec
from models.sklearn_supervised import sklearn_supervised
from models import sklearn_config
import jieba

jieba.setLogLevel('WARN')


class SentimentAnalysis():
    def __init__(self,
                 vocab_exist=False,
                 vocab_save=True,
                 vocab_path=os.getcwd() + '/sentence_transform/vocab_word2vec.model',
                 classify_exist=False,
                 classify_save=True,
                 classify_path=os.getcwd() + '/sentence_transform/classify.model'):
        self.vocab_exist = vocab_exist
        self.vocab_save = vocab_save
        self.vocab_path = vocab_path
        self.classify_exist = classify_exist
        self.classify_save = classify_save
        self.classify_path = classify_path

    # def creat_label(self, texts):
    #     results_dataframe = bat.creat_label(texts)
    #     return results_dataframe

    def get_vocab(self,
                  texts=None,
                  sg=0,
                  size=5,
                  window=5,
                  min_count=1):
        # 构建词向量词库
        self.vocab_word2vec = creat_vocab_word2vec(texts=texts,
                                                   sg=sg,
                                                   vocab_exist=self.vocab_exist,
                                                   vocab_save=self.vocab_save,
                                                   vocab_path=self.vocab_path,
                                                   size=size,
                                                   window=window,
                                                   min_count=min_count)

    def train(self,
              texts=None,
              label=None,
              model_name='SVM',
              **sklearn_param):
        self.model_name = model_name
        # 文本转词向量
        vocab_word2vec = self.vocab_word2vec
        texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
        data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
        # sklearn模型，词向量计算均值
        if model_name in ['SVM', 'KNN', 'Logistic']:
            data = [sum(i) / len(i) for i in data]
        # 配置sklearn模型参数
        if model_name == 'SVM':
            if sklearn_param == {}:
                sklearn_param = sklearn_config.SVC
        elif model_name == 'KNN':
            if sklearn_param == {}:
                sklearn_param = sklearn_config.KNN
        elif model_name == 'Logistic':
            if sklearn_param == {}:
                sklearn_param = sklearn_config.Logistic
        # 返回训练模型
        self.model = sklearn_supervised(data=data,
                                        label=label,
                                        model_exist=self.classify_exist,
                                        model_path=self.classify_path,
                                        savemodel=self.classify_save,
                                        model_name=model_name,
                                        **sklearn_param)

    def fit(self,
            texts=None):
        # 文本转词向量
        vocab_word2vec = self.vocab_word2vec
        if self.model_name in ['SVM', 'KNN', 'Logistic']:
            texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
            data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
            data = [sum(i) / len(i) for i in data]
        results = self.model.predict(data)
        return results


if __name__ == '__main__':
    train_data = ['国王喜欢吃苹果',
                  '国王非常喜欢吃苹果',
                  '国王讨厌吃苹果',
                  '国王非常讨厌吃苹果']
    train_label = ['正面', '正面', '负面', '负面']
    print('train data\n',
          pd.DataFrame({'data': train_data,
                        'label': train_label},
                       columns=['data', 'label']))
    test_data = ['涛哥喜欢吃苹果',
                 '涛哥讨厌吃苹果',
                 '涛哥非常喜欢吃苹果',
                 '涛哥非常讨厌吃苹果']
    test_label = ['正面', '负面', '正面', '负面']
    # 创建模型
    model = SentimentAnalysis(vocab_exist=False,
                              vocab_save=True,
                              vocab_path=os.getcwd() + '/vocab_word2vec.model',
                              classify_exist=False,
                              classify_save=True,
                              classify_path=os.getcwd() + '/classify.model')
    # 获取词向量模型
    model.get_vocab(texts=train_data,
                    sg=0,
                    size=5,
                    window=5,
                    min_count=1)
    # 进行机器学习
    model.train(texts=train_data,
                label=train_label,
                model_name='SVM')
    # 进行预测
    result = model.fit(texts=test_data)
    # 计算准确率
    print('score:', np.sum(result == np.array(test_label)) / len(result))
    result = pd.DataFrame({'data': test_data,
                           'label': test_label,
                           'predict': result},
                          columns=['data', 'label', 'predict'])
    print('test\n', result)
