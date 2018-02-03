from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import os


def sklearn_supervised(data=None,
                       label=None,
                       model_exist=False,
                       model_path=os.getcwd() + '/sentence_transform/classify.model',
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
            model = KNeighborsClassifier(**sklearn_param)
        elif model_name == 'SVM':
            # 核函数为linear,惩罚系数为1.0
            model = SVC(**sklearn_param)
            model.fit(data, label)
        elif model_name == 'Logistic':
            model = LogisticRegression(**sklearn_param)  # 核函数为线性,惩罚系数为1
            model.fit(data, label)

        if savemodel == True:
            joblib.dump(model, model_path)  # 保存模型
    else:  # 存在模型则直接调用
        model = joblib.load(model_path)
    return model


