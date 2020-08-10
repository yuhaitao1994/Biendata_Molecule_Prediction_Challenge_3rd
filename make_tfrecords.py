# coding=utf-8
"""
将数据转化为深度模型读取的tfrecords格式，
分别为deep_model_1和deep_model_2提供数据
@author: yuhaitao
"""
import pandas as pd
import os
import tqdm
import numpy as np
import seaborn as sns
import json
import gc
import sys
import datetime
import multiprocessing
import tensorflow as tf
from random import random
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.model_selection import KFold
from data_loader import myDataLoader, var_norm, min_max_norm
from utils import get_emb_id, cross_feature, norm_and_smape, SMAPE


def make_tfrecords_train_eval(data, out_dir, prefix='standard'):
    """将训练集5折分别写入tfrecords"""
    # 划分k_fold
    all_index = range(len(data))
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    # 训练k_fold
    for fold_idx, (train_idx, val_idx) in enumerate(k_fold.split(all_index)):
        train_fold = data.iloc[train_idx]
        val_fold = data.iloc[val_idx]
        # 提取使用特征的columns
        use_cols = [col for col in data.columns if col !=
                    'id' and 'p' not in col]
        print(f'Number of common used features: {len(use_cols)}')
        print(f'Making tfrecords for fold_{fold_idx}')
        # 将train存入tfrecords
        train_x, train_labels, train_id = train_fold[use_cols], train_fold[[f'p{i+1}' for i in range(6)]], train_fold[['id']]
        with tf.io.TFRecordWriter(os.path.join(out_dir, f'{prefix}_train_{len(use_cols)}_fold_{fold_idx}.tfrecords')) as writer:
            for inputs, labels, id_ in zip(train_x.values, train_labels.values, train_id.values):
                feature = {
                    'inputs': tf.train.Feature(float_list=tf.train.FloatList(value=inputs)),
                    'labels': tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
                    'id_': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(id_[0]).encode()]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        # 将val写入tfrecords
        val_x, val_labels, val_id = val_fold[use_cols], val_fold[[f'p{i+1}' for i in range(6)]], val_fold[['id']]
        with tf.io.TFRecordWriter(os.path.join(out_dir, f'{prefix}_val_{len(use_cols)}_fold_{fold_idx}.tfrecords')) as writer:
            for inputs, labels, id_ in zip(val_x.values, val_labels.values, val_id.values):
                feature = {
                    'inputs': tf.train.Feature(float_list=tf.train.FloatList(value=inputs)),
                    'labels': tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
                    'id_': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(id_[0]).encode()]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


def make_tfrecords_test(data, out_dir, prefix=''):
    """将测试集写入tfrecords"""
    use_cols = [col for col in data.columns if col != 'id' and 'p' not in col]
    print(f'Number of common used features: {len(use_cols)}')
    print(f'Making tfrecords for test data')
    test_x, test_labels, test_id = data[use_cols], data[[f'p{i+1}' for i in range(6)]], data[['id']]
    with tf.io.TFRecordWriter(os.path.join(out_dir, f'{prefix}_test_{len(use_cols)}.tfrecords')) as writer:
        for inputs, labels, id_ in zip(test_x.values, test_labels.values, test_id.values):
            feature = {
                'inputs': tf.train.Feature(float_list=tf.train.FloatList(value=inputs)),
                'labels': tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
                'id_': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(id_[0]).encode()]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def standard_main(data, prefix, mode='train'):
    with open('./feature_info.json', 'r') as f:
        feature_infos = json.load(f)
    use_cols = [col for col in data.columns if col !='id' and 'p' not in col]
    # 划分wide 与 deep 不同部分的特征
    deep_cols, wide_cols = [], []
    for col in use_cols:
        if 'w' not in col:
            deep_cols.append(col)
        else:
            wide_cols.append(col)
    wide_df = data[wide_cols]
    deep_df = data[deep_cols]
    # 打印一些
    print(f'Number of common used features: {len(use_cols)}')
    print(f'wide part dimension: {wide_df.shape}')
    print(f'deep part dimension: {deep_df.shape}')

    imp_feat = ['34', '10', '15', '3161', '14', '28', '753', '3160', '435', '3174', '928', '729', '1459', '2505', '457', '25', '8', '2801', '2483', '23', '869', '2976', '20', '11', '36']
    cross_df = cross_feature(wide_df, imp_feat, feature_infos) 
    print(f'cross features size for {imp_feat} is: {cross_df.shape}')

    # label也标准化
    label_df = None
    if mode == 'test':
        label_df = pd.DataFrame(np.zeros((len(wide_df),6)), columns=[f'p{i+1}' for i in range(6)])
        data = pd.concat([wide_df, deep_df, cross_df, label_df, data[['id']]], axis=1)
        make_tfrecords_test(data=data, out_dir='./data/tfrecords/standard', prefix=prefix)
    else:
        label_df = data[[f'p{i+1}' for i in range(6)]]
        label_df = label_df.apply(var_norm, args=(feature_infos,))
        data = pd.concat([wide_df, deep_df, cross_df, label_df, data[['id']]], axis=1)
        make_tfrecords_train_eval(data=data, out_dir='./data/tfrecords/standard', prefix=prefix)


def feature_norm_all(data, mode):
    """
    数据全部标准化，01特征标准化成+1.-1，交叉特征也按均值方差标准化
    注意：离散特征不是按照统一计算的均值方差，而是按照离散值集合的均值方差
    """
    def w_col_norm(x, feature_infos):
        """w_cols标准化"""
        x_list = feature_infos[x.name]['list']
        x_mean, x_std = np.arange(len(x_list)).mean(), np.arange(len(x_list)).std()
        out = []
        for a in x:
            if a in x_list:
                out.append(float((x_list.index(a)-x_mean)/x_std))
            else: # 对于未出现过的特征值，在尾后面加个小随机数赋值
                out.append(float((len(x_list)-1+random()-x_mean)/x_std))
        return out

    def c_col_norm(x, feature_infos):
        """c_cols标准化"""
        i_name, j_name = x.name.split('_')[1], x.name.split('_')[2]
        i_list, j_list = feature_infos[i_name]['list'], feature_infos[j_name]['list']
        x_len = len(i_list) * len(j_list) + 1
        x_mean, x_std = np.arange(x_len).mean(), np.arange(x_len).std()
        out = []
        for a in x: # 这里不用判断a的取值是否存在，因为经过feature engineering，交叉特征里都是存在的数了
            out.append(float((a - x_mean) / x_std))
        return out

    def bu_col_norm(x, feature_infos):
        """bu_cols标准化"""
        x_mean, x_std = np.arange(12).mean(), np.arange(12).std() # 注意分桶都是12，加更大值和更小值
        out = []
        for a in x: # 这里不用判断a的取值是否存在，因为经过feature engineering，分桶特征里都是存在的数了
            out.append(float((a - x_mean) / x_std))
        return out

    def bw_col_norm(x, feature_infos):
        """c_cols标准化"""
        b_name, w_name = x.name.split('_')[1], x.name.split('_')[2]
        w_list = feature_infos[w_name]['list']
        x_len = 12 * len(w_list) + 1
        x_mean, x_std = np.arange(x_len).mean(), np.arange(x_len).std()
        out = []
        for a in x: # 这里不用判断a的取值是否存在，因为经过feature engineering，交叉特征里都是存在的数了
            out.append(float((a - x_mean) / x_std))
        return out

    def bb_col_norm(x, feature_infos):
        """c_cols标准化"""
        x_len = 12 * 12 # 注意12
        x_mean, x_std = np.arange(x_len).mean(), np.arange(x_len).std()
        out = []
        for a in x: # 这里不用判断a的取值是否存在，因为经过feature engineering，交叉特征里都是存在的数了
            out.append(float((a - x_mean) / x_std))
        return out

    with open('./feature_info.json', 'r') as f:
        feature_infos = json.load(f)

    use_cols, not_use_cols = [], []
    for col in data.columns:
        if col != 'id' and 'p' not in col:
            use_cols.append(col)
        else:
            not_use_cols.append(col)

    deep_cols, wide_cols = [], []
    for col in use_cols:
        if data[col].dtype == float:
            deep_cols.append(col)
        else:
            wide_cols.append(col)

    w_cols, c_cols, bu_cols, bw_cols, bb_cols = [], [], [], [], []
    for col in wide_cols:
        if 'c_' in col:
            c_cols.append(col)
        elif 'bu_' in col:
            bu_cols.append(col)
        elif 'bw_' in col:
            bw_cols.append(col)
        elif 'bb_' in col:
            bb_cols.append(col)
        else:
            w_cols.append(col)

    new_data = None
    new_data = data[w_cols].apply(w_col_norm, args=(feature_infos, ), axis=0)

    mm = data[c_cols].apply(c_col_norm, args=(feature_infos, ), axis=0)
    new_data = pd.concat([new_data, mm], axis=1)

    mm_bu = data[bu_cols].apply(bu_col_norm, args=(feature_infos, ), axis=0)
    new_data = pd.concat([new_data, mm_bu], axis=1)

    mm_bw = data[bw_cols].apply(bw_col_norm, args=(feature_infos, ), axis=0)
    new_data = pd.concat([new_data, mm_bw], axis=1)

    mm_bb = data[bb_cols].apply(bb_col_norm, args=(feature_infos, ), axis=0)
    new_data = pd.concat([new_data, mm_bb], axis=1)

    deep_df = data[deep_cols].apply(var_norm, args=(feature_infos, ))
    new_data = pd.concat([new_data, deep_df], axis=1)

    if mode == 'train':
        label_df = data[[f'p{i+1}' for i in range(6)]]
        label_df = label_df.apply(var_norm, args=(feature_infos,))
        new_data = pd.concat([new_data, label_df, data[['id']]], axis=1)
    else:
        label_df = pd.DataFrame(np.zeros((len(new_data),6)), columns=[f'p{i+1}' for i in range(6)])
        new_data = pd.concat([new_data, label_df, data[['id']]], axis=1)

    new_new = new_data[[i for i in new_data.columns if i != 'id']]
    new_new = new_new.astype(np.float32)
    new_new = pd.concat([new_new, new_data[['id']]], axis=1)

    return new_new


def make_train_test_tfrecords(label_id):
    """为deep_model_2准备数据，每个指标分别准备"""
    train_data = pd.read_csv(f'./data/featured_data/{label_id}_feature_data_all_train.csv')
    norm_train_data = feature_norm_all(train_data, 'train')
    make_tfrecords_train_eval(norm_train_data, out_dir=f'./data/tfrecords/{label_id}', prefix=f'{label_id}_norm')

    test_data = pd.read_csv(f'./data/featured_data/{label_id}_feature_data_all_test.csv')
    norm_test_data = feature_norm_all(test_data, 'test')
    make_tfrecords_test(norm_test_data, out_dir=f'./data/tfrecords/{label_id}', prefix=f'{label_id}_norm')


if __name__ == '__main__':

    model_id = sys.argv[1]

    if model_id == 'deep_model_1':
        # deep_model_1的tfreords生成方式
        train_data = pd.read_csv('./data/molecule_open_data/nn_train_var.csv')
        print(f'Shape of train data: {train_data.shape}')
        standard_main(train_data, prefix='standard', mode='train')

        new_test_data = pd.read_csv('./data/molecule_open_data/nn_new_test_var.csv')
        print(f'Shape of new test data: {new_test_data.shape}')
        standard_main(new_test_data, prefix='new', mode='test')

    elif model_id == 'deep_model_2':
        # deep_model_2的tfrecords生成方式，因为deep_model_2各个指标（p1~p6）使用的特征不同，所以要分别生成
        label_list = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
        # 多线程生成tfreords
        pool = multiprocessing.Pool(processes=6)
        for label_id in label_list:
            pool.apply_async(make_train_test_tfrecords, (label_id, ))
        pool.close()
        pool.join()

    else:
        print('no model!')


    
