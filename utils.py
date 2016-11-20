# -*- encoding=utf8 -*-
"""
@author: Jerry
@contact: lvjy3.15@sem.tsinghua.edu.com
@file: build_test.py
@time: 2016/11/8 9:31
"""

import os

import jieba
import numpy as np
import pandas as pd
from tqdm import tqdm


def func():
    pass


def delete_view(views):
    views_update = [x for x in views]
    for view in views:
        for view1 in views:
            if view in view1 and view != view1:
                try:
                    views_update.remove(view)
                except:
                    pass
    return views_update


def find_view(path=u'E:\\Google Drive\\研二上\语义情感分析\\基于视角的领域情感分析-复赛\\NLP_emotion'):
    # 读取角度集合
    view = pd.read_table(path + u'\\Data\\View.txt', encoding='utf-8')
    # 读取测试数据
    test = pd.read_table(path + u'\\Data\\Test.csv', encoding='utf-8')
    test['view'] = ''
    # 在每个句子里寻找view
    #################
    #  该部分耗时很长  #
    #################
    for i in tqdm(range(len(test))):
        names = []
        for v in list(view['View']):
            if v in test.iloc[i, 1]:
                names += [v]
        test.iloc[i, 2] = '+'.join(names)
    # test.to_csv('C:\\Users\\Jerry\\Desktop\\check2.csv', encoding='utf-8')

    # view1.to_csv('C:\\Users\\Jerry\\Desktop\\test_view.csv', encoding='utf-8')
    return test


def update(view1):
    for i in tqdm(range(len(view1))):
        views = view1.iloc[i, 2]
        if isinstance(views, unicode):
            views = views.split('+')
            if len(views) > 1:
                views_update = delete_view(views)
                view1.iloc[i, 2] = '+'.join(views_update)
                # print('+'.join(views_update))
    return view1


def prepare_data(path=u'E:\\Google Drive\\研二上\语义情感分析\\基于视角的领域情感分析-复赛\\NLP_emotion'):
    os.chdir(path)
    print('Build train')
    train_single_set, train_multi_set = build_train(path)
    print('Build test')
    test_single_set, test_multi_set = build_test(path)
    return train_single_set, train_multi_set, test_single_set, test_multi_set


def add_feature(data, path=u'E:\\Google Drive\\研二上\语义情感分析\\基于视角的领域情感分析-复赛\\NLP_emotion'):
    print('Add feature values')
    features = pd.read_csv(path + '\\Data\\feature_1111.csv', encoding='utf-8')
    feature_all = features.loc[:, ['feature', 'type', 'score']]

    m = len(data)
    n = len(feature_all)
    # initialize the data set
    feature_matrix = np.zeros([m, n])
    features = list(feature_all['feature'])
    scores = list(feature_all['score'])

    # train_set[feature] = list(map(lambda x : ((x)).find((feature)),train_set['Content']))
    for i in tqdm(range(m)):
        content = list(data['Content'])[i]
        content_jieba = list(jieba.cut(content))
        for j in range(n):
            if features[j] in content_jieba:
                # feature_matrix[i,j] = 1/(content.index(feature_all.feature[j])-train_set.View_loc[i]+0.01)*100 # 0.70
                # feature_matrix[i,j] = len(content)/(np.abs(content.index(feature_all.feature[j])-train_set.View_loc[i])+0.01)*100 #0.74
                # feature_matrix[i,j] = content.index(feature_all.feature[j])-train_set.View_loc[i] # 0.69
                feature_matrix[i, j] = 1  # 0.77 0.81
                # feature_matrix[i,j] = content.index(feature_all.feature[j]) # 0.73 0.82
                # feature_matrix[i,j] = scores[j] # 0.77 0.82
    feature_df = pd.DataFrame(feature_matrix, columns=feature_all.feature)

    data_with_feature = pd.concat([data, feature_df], axis=1)
    return data_with_feature


def build_train(path=u'E:\\Google Drive\\研二上\语义情感分析\\基于视角的领域情感分析-复赛\\NLP_emotion'):
    view = pd.read_table(path + '\\Data\\View.txt', encoding='utf-8')['View']
    # 读取标签
    Data = pd.read_csv(path + '\\Data\\LabelSecond.csv', encoding='utf-8', sep='\t')
    DataID = Data['SentenceId']
    DataName = Data['View']
    DataS = Data['Opinion']
    train = pd.read_csv(path + '\\Data\\TrainSecond.csv', encoding='utf-8', sep='\t')
    # 扩充视角
    # added = pd.Series(list(set(DataName)-(set(view))))
    # view = pd.Series(list(set(view).union(set(DataName))))
    # train['A'] = map(lambda x:list(set(jieba.cut(x))-set(['，','。','：','、'])),train['Content'])
    temp = pd.DataFrame(Data.groupby('SentenceId').size())
    temp.columns = ['n']
    temp['SentenceId'] = temp.index
    train = pd.merge(train, temp, on='SentenceId', how='inner')
    # AllData = pd.Series([item for sublist in train['A'].tolist() for item in sublist])
    # mSightsData = train[train.n!=1]
    # Msightsdata = pd.Series(list(set([item for sublist in mSightsData['A'].tolist() for item in sublist])))
    train = pd.merge(train, Data, on='SentenceId', how='left')
    # posData = train[(train.n==1)&(train.Opinion=='pos')]
    # PosData = pd.Series([item for sublist in posData['A'].tolist() for item in sublist])
    # negData = train[(train.n==1)&(train.Opinion=="neg")]
    # NegData = pd.Series([item for sublist in negData['A'].tolist() for item in sublist])

    train['View_loc'] = -1
    train['View_loc'] = list(map(lambda x, y: ((y)).find((x)), train['View'], train['Content']))

    train_set = add_feature(train)

    train_single_set = train_set[train_set['n'] == 1]
    train_multi_set = train_set[train_set['n'] > 1]

    train_single_set = train_single_set.reset_index()
    train_multi_set = train_multi_set.reset_index()
    return train_single_set, train_multi_set


def build_test(path=u'E:\\Google Drive\\研二上\语义情感分析\\基于视角的领域情感分析-复赛\\NLP_emotion'):
    # 读取或者生成视角数据

    if os.path.isfile(path + '/Data/Test_view.csv'):
        print('File Exist!Read data from local disk')
        view1 = pd.read_csv(path + u'\\Data\\Test_view.csv', encoding='utf-8')  # 直接读取已有结果
    else:
        print('No test view file!Finding views in test.txt and save save the view list')
        view1 = find_view()  # 寻找每句话里包含的视角
        view1.to_csv(path + u'\\Data\\Test_view.csv', encoding='utf-8')

    view_update = update(view1)

    # 生成测试集
    # 测试集格式
    # SentenceId |View | Opinion    | Content | View_num | View_loc | features....
    # 000001     |途安L |pos/neg/neu | sentence| 视角数量  | 该视角位置 | 特征位置

    obs = []
    error = 0
    for i in range(len(view1)):
        views0 = view1['view'][i]
        try:
            view_list = views0.split('+')  # 分割
            for view in view_list:
                # 生成观测值
                if view != '':
                    ob = [view1['SentenceId'][i], view, 'unknown', view1['Content'][i], len(view_list),
                          view1['Content'][i].index(view)]
                    obs.append(ob)
        except:
            # print view1.iloc[i,2]
            pass

    test = pd.DataFrame(obs, columns=['SentenceId', 'View', 'Opinion', 'Content', 'n', 'View_loc'])
    # 加入feature
    test_set = add_feature(test)
    test_single_set = test_set[test_set['n'] == 1]
    test_multi_set = test_set[test_set['n'] > 1]
    test_single_set = test_single_set.reset_index()
    test_multi_set = test_multi_set.reset_index()
    return test_single_set, test_multi_set


if __name__ == '__main__':
    #########################################
    #######          SEETING        #########
    #########################################
    #
    # local: whether read data from local
    #
    #
    #
    x1, x2, x3, x4 = prepare_data()
