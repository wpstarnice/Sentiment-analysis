from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
import pandas as pd
import jieba
from gensim.models import word2vec, doc2vec
import numpy as np
import os

jieba.setLogLevel('WARN')


def _vocab_word2vec(texts=None,
                    sg=0,
                    model_exist=False,
                    model_save=True,
                    path=os.getcwd() + '/vocab_word2vec.model',
                    size=5,
                    window=5,
                    min_count=1):
    texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
    # 不存在模型则训练,否则导入
    if model_exist == False:
        model = word2vec.Word2Vec(texts_cut, sg=sg, size=size, window=window, min_count=min_count)
        if model_save == True:
            model.save(path)
    else:
        model = word2vec.Word2Vec.load(path)

    # train_data = [[model[word] for word in one_text] for one_text in texts_cut]
    return model


if __name__ == '__main__':
    texts = ['全面从严治党',
             '国际公约和国际法',
             '中国航天科技集团有限公司']
    vocab_word2vec = _vocab_word2vec(texts=texts,
                                     sg=0,
                                     model_exist=False,
                                     model_save=True,
                                     path=os.getcwd() + '/vocab_word2vec.model',
                                     size=5,
                                     window=5,
                                     min_count=1)
