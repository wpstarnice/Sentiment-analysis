# from creat_data import bat
import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import History
from keras.models import load_model
import jieba

from sentence_transform.creat_vocab_word2vec import creat_vocab_word2vec
from models.sklearn_supervised import sklearn_supervised
from models import sklearn_config
from models.neural_bulit import neural_bulit
from models.keras_log_plot import keras_log_plot

jieba.setLogLevel('WARN')


class SentimentAnalysis():
    def __init__(self):
        pass

    # def creat_label(self, texts):
    #     results_dataframe = bat.creat_label(texts)
    #     return results_dataframe

    def creat_vocab(self,
                    texts=None,
                    sg=0,
                    size=5,
                    window=5,
                    min_count=1,
                    vocab_savepath=os.getcwd() + '/vocab_word2vec.model'):
        # 构建词向量词库
        self.vocab_word2vec = creat_vocab_word2vec(texts=texts,
                                                   sg=sg,
                                                   vocab_savepath=vocab_savepath,
                                                   size=size,
                                                   window=window,
                                                   min_count=min_count)

    def load_vocab_word2vec(self,
                            vocab_loadpath=os.getcwd() + '/vocab_word2vec.model'):
        self.vocab_word2vec = word2vec.Word2Vec.load(vocab_loadpath)

    def train(self,
              texts=None,
              label=None,
              model_name='SVM',
              model_savepath=os.getcwd() + '/classify.model',
              batch_size=100,  # 神经网络参数
              epochs=2,  # 神经网络参数
              verbose=2,  # 神经网络参数
              maxlen=50,  # 神经网络参数
              **sklearn_param):
        self.model_name = model_name
        self.label = np.unique(np.array(label))
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
                                            model_savepath=model_savepath,
                                            model_name=model_name,
                                            **sklearn_param)
        # keras神经网络模型，
        elif model_name in ['Conv1D_LSTM', 'Conv1D', 'LSTM']:
            if maxlen == None:
                maxlen = max([len(i) for i in texts_cut])
            data = pad_sequences(data, maxlen=maxlen, padding='post', value=0, dtype='float32')
            self.maxlen = maxlen
            label_transform = np.array(pd.get_dummies(label))
            if model_name == 'Conv1D_LSTM':
                net_shape = [
                    {'name': 'InputLayer', 'input_shape': data.shape[1:]},
                    {'name': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'strides': 1, 'padding': 'same',
                     'activation': 'relu'},
                    {'name': 'MaxPooling1D', 'pool_size': 5, 'padding': 'same', 'strides': 2},
                    {'name': 'LSTM', 'units': 16, 'activation': 'tanh', 'recurrent_activation': 'hard_sigmoid',
                     'dropout': 0., 'recurrent_dropout': 0.},
                    {'name': 'Flatten'},
                    {'name': 'Dense', 'activation': 'relu', 'units': 64},
                    {'name': 'Dropout', 'rate': 0.2, },
                    {'name': 'softmax', 'activation': 'softmax', 'units': len(np.unique(label))}
                ]

            elif model_name == 'LSTM':
                net_shape = [
                    {'name': 'InputLayer', 'input_shape': data.shape[1:]},
                    {'name': 'Masking'},
                    {'name': 'LSTM', 'units': 16, 'activation': 'tanh', 'recurrent_activation': 'hard_sigmoid',
                     'dropout': 0., 'recurrent_dropout': 0.},
                    {'name': 'Dense', 'activation': 'relu', 'units': 64},
                    {'name': 'Dropout', 'rate': 0.2, },
                    {'name': 'softmax', 'activation': 'softmax', 'units': len(np.unique(label))}
                ]
            elif model_name == 'Conv1D':
                net_shape = [
                    {'name': 'InputLayer', 'input_shape': data.shape[1:]},
                    {'name': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'strides': 1, 'padding': 'same',
                     'activation': 'relu'},
                    {'name': 'MaxPooling1D', 'pool_size': 5, 'padding': 'same', 'strides': 2},
                    {'name': 'Flatten'},
                    {'name': 'Dense', 'activation': 'relu', 'units': 64},
                    {'name': 'Dropout', 'rate': 0.2, },
                    {'name': 'softmax', 'activation': 'softmax', 'units': len(np.unique(label))}
                ]
            model = neural_bulit(net_shape=net_shape,
                                 optimizer_name='Adagrad',
                                 lr=0.001,
                                 loss='categorical_crossentropy')
            history = History()
            model.fit(data, label_transform,
                      batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[history])
            train_log = pd.DataFrame(history.history)
            self.model = model
            self.train_log = train_log
            if model_savepath != None:
                model.save(model_savepath)

    def load_model(self,
                   model_loadpath=os.getcwd() + '/classify.model'):
        if self.model_name in ['SVM', 'KNN', 'Logistic']:
            self.model = joblib.load(model_loadpath)
        elif self.model_name in ['Conv1D_LSTM', 'Conv1D', 'LSTM']:
            self.model = load_model(model_loadpath)

    def predict_prob(self,
                     texts=None):
        # 文本转词向量
        vocab_word2vec = self.vocab_word2vec
        if self.model_name in ['SVM', 'KNN', 'Logistic']:
            texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
            data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
            data = [sum(i) / len(i) for i in data]
            self.testdata = data
            results = self.model.predict_proba(data)
        elif self.model_name in ['Conv1D_LSTM', 'Conv1D', 'LSTM']:
            texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
            data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
            data = pad_sequences(data, maxlen=self.maxlen, padding='post', value=0, dtype='float32')
            self.testdata = data
            results = self.model.predict(data)
        return results

    def predict(self,
                texts=None):
        # 文本转词向量
        vocab_word2vec = self.vocab_word2vec
        if self.model_name in ['SVM', 'KNN', 'Logistic']:
            texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
            data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
            data = [sum(i) / len(i) for i in data]
            self.testdata=data
            results = self.model.predict(data)
        elif self.model_name in ['Conv1D_LSTM', 'Conv1D', 'LSTM']:
            texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
            data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
            data = pad_sequences(data, maxlen=self.maxlen, padding='post', value=0, dtype='float32')
            self.testdata = data
            results = self.model.predict(data)
            results = pd.DataFrame(results, columns=model.label)
            results = results.idxmax(axis=1)
        return results


if __name__ == '__main__':
    train_data = ['国王喜欢吃苹果',
                  '国王非常喜欢吃苹果',
                  '国王讨厌吃苹果',
                  '国王非常讨厌吃苹果']
    train_label = ['正面', '正面', '负面', '负面']
    # print('train data\n',
    #       pd.DataFrame({'data': train_data,
    #                     'label': train_label},
    #                    columns=['data', 'label']))
    test_data = ['涛哥喜欢吃苹果',
                 '涛哥讨厌吃苹果',
                 '涛哥非常喜欢吃苹果',
                 '涛哥非常讨厌吃苹果']
    test_label = ['正面', '负面', '正面', '负面']
    # 创建模型
    model = SentimentAnalysis()
    # 建模获取词向量词包
    model.creat_vocab(texts=train_data,
                      sg=0,
                      size=5,
                      window=5,
                      min_count=1,
                      vocab_savepath=os.getcwd() + '/vocab_word2vec.model')

    # 导入词向量词包
    # model.load_vocab_word2vec(vocab_loadpath=os.getcwd() + '/vocab_word2vec.model')

    ###################################################################################
    # 进行机器学习
    model.train(texts=train_data,
                label=train_label,
                model_name='SVM',
                model_savepath=os.getcwd() + '/classify.model')

    # 导入机器学习模型
    # model.load_model(model_loadpath=os.getcwd() + '/classify.model')

    # 进行预测:概率
    result_prob = model.predict_prob(texts=test_data)
    result_prob = pd.DataFrame(result_prob, columns=model.label)
    result_prob['predict'] = result_prob.idxmax(axis=1)
    print('prob:\n', result_prob)

    # 进行预测:分类
    result = model.predict(texts=test_data)
    print('score:', np.sum(result == np.array(test_label)) / len(result))
    result = pd.DataFrame({'data': test_data,
                           'label': test_label,
                           'predict': result},
                          columns=['data', 'label', 'predict'])
    print('test\n', result)
    ###################################################################################
    # 进行深度学习
    # model.train(texts=train_data,
    #             label=train_label,
    #             model_name='Conv1D',
    #             batch_size=100,
    #             epochs=2,
    #             verbose=1,
    #             maxlen=None,
    #             model_savepath=os.getcwd() + '/classify.h5')
    #
    # 导入深度学习模型
    # model.load_model(model_loadpath=os.getcwd() + '/classify.h5')

    # 进行预测:概率
    # result_prob=model.predict_prob(texts=test_data)
    # result_prob = pd.DataFrame(result_prob, columns=model.label)
    # result_prob['predict'] = result_prob.idxmax(axis=1)
    # print(result_prob)
    # # 进行预测:分类
    # result = model.predict(texts=test_data)
    # print(result)
    # print('score:', np.sum(result == np.array(test_label)) / len(result))
    # result = pd.DataFrame({'data': test_data,
    #                        'label': test_label,
    #                        'predict': result},
    #                       columns=['data', 'label', 'predict'])
    # print('test\n', result)

    # keras_log_plot(model.train_log)
