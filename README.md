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
该模块包含：
1.打情感分析标签，用于在缺乏标签的时候利用BAT三家的接口创建训练集，5000条文档共耗时约45分钟；
2.通过gensim模块创建词向量词包
3.通过scikit-learn进行机器学习并预测

``` python
from SentimentAnalysis import SentimentAnalysis
import numpy as np

train_data = ['国王喜欢吃苹果',
              '国王非常喜欢吃苹果',
              '国王讨厌吃苹果',
              '国王非常讨厌吃苹果']
train_label = ['正面', '正面', '负面', '负面']

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
print('score:', np.sum(result == np.array(test_label)) / len(result))

```
