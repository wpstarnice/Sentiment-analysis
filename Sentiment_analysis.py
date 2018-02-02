from creat_data import bat
import pymysql
import pandas as pd
import numpy as np
import os
from sentence_transform._vocab_word2vec import _vocab_word2vec
from models.sklearn_supervised import sklearn_supervised


class Sentiment_analysis():
    def __init__(self,):
        pass

    def creat_label(self, texts):
        results_dataframe = bat.creat_label(texts)
        return results_dataframe

    def creat_vocab_word2vec(self,
                             texts=None,
                             sg=0,
                             model_exist=False,
                             model_save=True,
                             path=os.getcwd() + '/sentence_transform/vocab_word2vec.model',
                             size=5,
                             window=5,
                             min_count=1):
        self.vocab_word2vec = _vocab_word2vec(texts=texts,
                                              sg=sg,
                                              model_exist=model_exist,
                                              model_save=model_save,
                                              path=path,
                                              size=size,
                                              window=window,
                                              min_count=min_count)

    def texts_word2vec(self, texts):
        model = self.vocab_word2vec
        texts_transform = [[model[word] for word in sample if word in model] for sample in texts]
        return texts_transform

    def sklearn_supervised(data=None,
                           label=None,
                           model_exist=False,
                           model_path=None,
                           model_name='SVM',
                           savemodel=True,
                           **sklearn_param):



if __name__ == '__main__':
    train_data = ['全面从严治党',
                  '国际公约和国际法',
                  '中国航天科技集团有限公司']
    test_data = ['全面从严测试']
    train_data_label = Sentiment_analysis.creat_label(train_data)
    Sentiment_analysis.creat_vocab_word2vec(texts=train_data + test_data,
                                            sg=0,
                                            model_exist=False,
                                            model_save=True,
                                            path=os.getcwd() + '/sentence_transform/vocab_word2vec.model',
                                            size=5,
                                            window=5,
                                            min_count=1)
    train_data_vec = Sentiment_analysis.texts_word2vec(train_data)
    test_data_vec = Sentiment_analysis.texts_word2vec(test_data)
