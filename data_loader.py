# coding=utf-8
"""
加载原始数据，并将categorical features转化成NN模型1的one-hot编码，将numerical features标准化
@author: yuhaitao
"""
import pandas as pd
import os
import tqdm
import numpy as np
import seaborn as sns
import json
import datetime
import multiprocessing
from sklearn.model_selection import KFold


def min_max_norm(x, feature_infos):
    # deep部分进行min-max归一化
    min_value = feature_infos[x.name]['min']
    max_value = feature_infos[x.name]['max']
    out = pd.Series(index=range(x.size))
    index = 0
    for one in x:
        if one == max_value:
            out[index] = 1.0
        elif one == min_value:
            out[index] = 0.0
        else:
            out[index] = (one - min_value) / (max_value - min_value)
        index += 1
    return out

    
def var_norm(x, feature_infos):
    # 方差归一化
    mean = feature_infos[x.name]['mean']
    std = feature_infos[x.name]['std']
    out = pd.Series(index=range(x.size))
    index = 0
    for one in x:
        out[index] = (one - mean) / std
        index += 1
    return out


class myDataLoader(object):
    """
    """

    def __init__(self, data_path):
        """
        初始化
        """
        self.data_path = data_path

    def dataset_for_boost(self, train_file, test_file, label_file):
        """
        加载数据集
        """
        train_data = pd.read_csv(os.path.join(self.data_path, train_file))
        test_data = pd.read_csv(os.path.join(self.data_path, test_file))
        train_answer = pd.read_csv(os.path.join(self.data_path, label_file))

        train_data = train_data.merge(train_answer, on='id', how='left')

        # 处理缺失值，该数据集暂时没有

        # 去掉数值全部相同的特征
        singleValuesCnt = 0
        for i in train_data.columns:
            if len(train_data[i].unique()) == 1:
                train_data.drop([i], axis=1, inplace=True)
                test_data.drop([i], axis=1, inplace=True)
                singleValuesCnt += 1
        print("{}singleValues feathers are cleaned..".format(singleValuesCnt))

        return train_data, test_data

    def normalize_to_json(self, train_data):
        """
        将训练集数据规范化后的均值，边界等指标存入json文件
        """
        feature_infos = {}
        # 随机采样80%的数据进行统计，模拟5-fold
        train_data = train_data.sample(frac=0.8, replace=False, axis=0)
        # 分开处理deep部分与wide部分
        use_cols = [col for col in train_data.columns if col !=
                    'id' and 'p' not in col]
        print(f'Number of common used features: {len(use_cols)}')
        deep_cols, wide_cols = [], []
        for col in use_cols:
            if train_data[col].dtype == float:
                deep_cols.append(col)
            else:
                wide_cols.append(col)

        # 处理deep部分
        def numeric_status(x):
            return pd.Series([x.min(), x.mean(), x.max(), x.std(), ], index=['min', 'ave', 'max', 'std'])
        deep_norm_df = train_data[deep_cols].apply(numeric_status)

        for col in deep_cols:
            c_max = min(
                deep_norm_df[col][2], deep_norm_df[col][1] + deep_norm_df[col][3] * 3)
            c_min = max(
                deep_norm_df[col][0], deep_norm_df[col][1] - deep_norm_df[col][3] * 3)
            feature_infos[col] = {'min': c_min, 'max': c_max,
                                  'mean': deep_norm_df[col][1], 'std': deep_norm_df[col][3]}

        # 处理wide部分
        def categorical_status(x):
            cat_dict = {}
            for one in x:
                if not pd.isnull(one):
                    if int(one) in cat_dict:
                        cat_dict[int(one)] += 1
                    else:
                        cat_dict[int(one)] = 1

            cat_list = [tup[0] for tup in sorted(
                cat_dict.items(), key=lambda x:x[1], reverse=True)[:min(len(cat_dict), 100)]]
            # if 0 not in cat_list:
            #     cat_list.append(0)
            cat_list.sort()
            return pd.Series([cat_list], index=['list'])
        wide_norm_df = train_data[wide_cols].apply(categorical_status)

        for col in wide_cols:
            feature_infos[col] = {'list': wide_norm_df[col][0]}

        # 处理label(暂时没标准化)
        label_cols = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
        label_norm_df = train_data[label_cols].apply(numeric_status)
        for col in label_cols:
            l_max = min(
                label_norm_df[col][2], label_norm_df[col][1] + label_norm_df[col][3] * 3)
            l_min = max(
                label_norm_df[col][0], label_norm_df[col][1] - label_norm_df[col][3] * 3)
            feature_infos[col] = {'min': l_min, 'max': l_max,
                                  'mean': label_norm_df[col][1], 'std': label_norm_df[col][3]}

        with open('./feature_info.json', 'w') as f:
            f.write(json.dumps(feature_infos))


    def prepare_nn_data(self, data, mode='train', norm_mode='min_max'):
        """
        将数据规范化后存入csv文件
        """
        with open('./feature_info.json', 'r') as f:
            feature_infos = json.load(f)

        nn_data_file = os.path.join(
            self.data_path, f'nn_{mode}_{norm_mode}.csv')

        use_cols, not_use_cols = [], []
        for col in data.columns:
            if col != 'id' and 'p' not in col:
                use_cols.append(col)
            else:
                not_use_cols.append(col)

        print(f'Number of common used features: {len(use_cols)}')
        print('*' * 120)
        # 划分wide 与 deep 不同部分的特征
        deep_cols, wide_cols = [], []
        for col in use_cols:
            if data[col].dtype == float:
                deep_cols.append(col)
            else:
                wide_cols.append(col)

        # wide部分进行one-hot编码，通过align方法保持所有编码的维度与feature info中存储的信息一致
        start_time = datetime.datetime.now()
        print(f'Number of wide features: {len(wide_cols)}')
        one_hot_list = []
        for col in wide_cols:
            for c in feature_infos[col]['list']:
                one_hot_list.append(f'w{col}_{c}')
        print(f'one hot dimension: {len(one_hot_list)}')
        wide_df = pd.DataFrame(columns=one_hot_list)
        # 生成当前数据集的one hot
        one_hot_df = pd.get_dummies(data[wide_cols].astype(
            str), prefix=['w' + col for col in wide_cols])
        print(f'current one hot dimension: {len(one_hot_df.columns)}')
        # 两个dataframe合并，以feature info中的维度为准
        _, wide_df = wide_df.align(one_hot_df, join='left', axis=1, fill_value=0)
        wide_df = wide_df.astype(np.float32)

        end_time = datetime.datetime.now()
        print(f'data processing cost time: {(end_time-start_time)}')
        print(f'wide part dimension: {wide_df.shape}')
        print('*' * 120)
        print(wide_df.columns)

        # deep部分
        if norm_mode == 'min_max':
            deep_df = data[deep_cols].apply(min_max_norm, args=(feature_infos,))
        else:
            deep_df = data[deep_cols].apply(var_norm, args=(feature_infos, ))
        print(f'Number of deep features: {len(deep_cols)}')
        print(f'deep part dimension: {deep_df.shape}')
        print('*' * 120)

        # 保存到文件
        out_df = pd.DataFrame()
        out_df = out_df.join(wide_df, how='right')
        out_df = out_df.join(deep_df, how='right')
        out_df = out_df.join(data[not_use_cols], how='right')
        out_df.to_csv(nn_data_file, index=False)


    def data_for_nn(self, train_file, test_file):
        """
        直接读取已经标准化与one hot编码好的csv文件
        """
        train_data = pd.read_csv(os.path.join(self.data_path, train_file))
        test_data = pd.read_csv(os.path.join(self.data_path, test_file))
        return train_data, test_data




if __name__ == '__main__':
    # 加载数据
    data_loader = myDataLoader('./data/molecule_open_data')
    train_data, test_data = data_loader.dataset_for_boost(
        train_file='candidate_train.csv', test_file='candidate_test_clean.csv', label_file='train_answer.csv')

    # data_loader.normalize_to_json(train_data)
    data_loader.prepare_nn_data(train_data, 'train', 'var') # 训练集
    # data_loader.prepare_nn_data(test_data, 'test', 'var') # test A
    data_loader.prepare_nn_data(test_data, 'new_test', 'var') # test B

