from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sentence_transform.sentence_2_sparse import sentence_2_sparse
from sentence_transform.sentence_2_vec import sentence_2_vec


def sklearn_supervised(data=None,
                       label=None,
                       model_exist=False,
                       model_path=None,
                       model_name='SVM',
                       savemodel=True,
                       **sklearn_param):
    '''
    :param data: 训练文本
    :param label: 训练文本的标签
    :param model_exist: 模型是否存在
    :param model_path: 模型路径
    :param model_name: 机器学习分类模型,SVM,KNN,Logistic
    :param savemodel: 保存模型
    :param return: 训练好的模型
    '''

    model_path = model_path
    if model_exist == False:  # 如果不存在模型,调训练集训练
        model_name = model_name
        if model_name == 'KNN':
            # 调用KNN,近邻=5
            model = KNeighborsClassifier(n_neighbors=min(len(label), sklearn_param['n_neighbors']))
        elif model_name == 'SVM':
            # 核函数为linear,惩罚系数为1.0
            model = SVC(kernel=sklearn_param['kernel'],
                        C=sklearn_param['C'])
            model.fit(data, label)
        elif model_name == 'Logistic':
            model = LogisticRegression(solver='liblinear', C=1.0)  # 核函数为线性,惩罚系数为1
            model.fit(data, label)

        if savemodel == True:
            joblib.dump(model, model_path)  # 保存模型
    else:  # 存在模型则直接调用
        model = joblib.load(model_path)
    return model


if __name__ == '__main__':
    print('example:English')
    dataset = [['国王喜欢吃苹果',
                '国王非常喜欢吃苹果',
                '国王讨厌吃苹果',
                '国王非常讨厌吃苹果'],
               ['正面', '正面', '负面', '负面']]
    print('train data\n',
          pd.DataFrame({'data': dataset[0],
                        'label': dataset[1]},
                       columns=['data', 'label']))

    model = sklearn_supervised(data=dataset[0],
                               label=dataset[1],
                               model_exist=False,
                               model_path=None,
                               model_name='SVM',
                               savemodel=True)
    print('model:', model)
