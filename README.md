# Sentiment-analysis

## 语言
Python3.5<br>
## 依赖库
requests=2.18.4<br>
baidu-aip=2.1.0.0<br>
pandas=0.21.0<br>
numpy=1.13.1<br>
jieba=0.39<br>
gensim=3.2.0<br>
scikit-learn=0.19.1<br>
keras=2.1.1<br>





## 项目介绍
通过对已有标签的帖子进行训练，实现新帖子的情感分类，用法类似scikit-learn。<br>
已完成KNN、SVM和Logistic的封装。训练集为一万条记录，SVM效果最好，准确率在87%左右.<br>
***PS：该项目在上一个项目Text-Classification基础上封装而成~目前公司情感分析借鉴这个项目，有很多不足，欢迎萌新、大佬多多指导！***

## 用法简介 SentimentAnalysis
该模块包含：<br>
1.借助第三方平台，打情感分析标签。用于在缺乏标签的时候利用BAT三家的接口创建训练集，5000条文档共耗时约45分钟；<br>
2.通过gensim模块创建词向量词包<br>
3.通过scikit-learn进行机器学习并预测<br>
4.通过keras进行深度学习并预测<br>

其他说明：在训练集很小的情况下，sklearn的概率输出predict_prob会不准。目前发现，SVM会出现所有标签概率一样，暂时没看源码，猜测是离超平面过近不计算概率，predict不会出现这个情况。

``` python
from SentimentAnalysis import SentimentAnalysis
import numpy as np

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
result_prob['data'] = test_data
result_prob = result_prob[['data'] + list(model.label) + ['predict']]
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
model.train(texts=train_data,
            label=train_label,
            model_name='Conv1D',
            batch_size=100,
            epochs=2,
            verbose=1,
            maxlen=None,
            model_savepath=os.getcwd() + '/classify.h5')

# 导入深度学习模型
# model.load_model(model_loadpath=os.getcwd() + '/classify.h5')

# 进行预测:概率
result_prob = model.predict_prob(texts=test_data)
result_prob = pd.DataFrame(result_prob, columns=model.label)
result_prob['predict'] = result_prob.idxmax(axis=1)
print(result_prob)

# 进行预测:分类
result = model.predict(texts=test_data)
print(result)
print('score:', np.sum(result == np.array(test_label)) / len(result))
result = pd.DataFrame({'data': test_data,
                       'label': test_label,
                       'predict': result},
                      columns=['data', 'label', 'predict'])
print('test\n', result)

keras_log_plot(model.train_log)

```
SVM<br>
![SVM](https://github.com/renjunxiang/Sentiment-analysis/blob/master/picture/SVM.png)<br>
Conv1D<br>
![Conv1D](https://github.com/renjunxiang/Sentiment-analysis/blob/master/picture/Conv1D.png)



