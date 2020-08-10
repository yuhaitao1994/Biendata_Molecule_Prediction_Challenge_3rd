# coding=utf-8
"""
对train和test数据进行特征工程，生成的数据提供给make_tfrecords.py
@author: yuhaitao
"""
import pandas as pd
import os
import numpy as np
import gc
import pickle
import datetime
import logging
import sys
import json
import multiprocessing
from data_loader import myDataLoader


def make_features(data, params, out_dir, label_id, mode):
    """
    特征工程，并保存到新的数据文件中
    """
    with open('./feature_info.json', 'r') as f:
        feature_infos = json.load(f)
    feature_imps = pd.read_csv(
        f'./data/feature_imps/feature_imps_{label_id}.csv', index_col=False)
    # 去掉后面特征
    if params['drop_1500']:
        del_feats = list(feature_imps['feature'].values.astype(str)[-1500:])
        data_new = data.drop(axis=1, columns=del_feats)
        print(f'After drop 1500, data shape:{data_new.shape}')

    imp_feats = list(feature_imps['feature'].values.astype(str)[:50])
    wide_imp_feats = []
    deep_imp_feats = []
    for feat in imp_feats:
        if data[feat].dtype == float:
            deep_imp_feats.append(feat)
        elif data[feat].dtype == int:
            wide_imp_feats.append(feat)
        else:
            raise ValueError
    # wide特征交叉
    if params['wide_cross']:
        str_df = pd.DataFrame()
        for i in range(len(wide_imp_feats) - 1):
            for j in range(i + 1, len(wide_imp_feats)):
                i_name, j_name = wide_imp_feats[i], wide_imp_feats[j]
                str_df['c_' + i_name + '_' + j_name] = data_new[i_name].astype(
                    str).values + '_' + data_new[j_name].astype(str).values

        def get_cross(x, feature_infos):
            i_name, j_name = x.name.split('_')[1], x.name.split('_')[2]
            i_list, j_list = feature_infos[i_name]['list'], feature_infos[j_name]['list']
            out = []
            for one in x:
                i, j = int(one.split('_')[0]), int(one.split('_')[1])
                if i not in i_list or j not in j_list:
                    out.append(0)
                else:
                    out.append(i_list.index(i) * len(j_list) +
                               j_list.index(j) + 1)
            return out
        cross_df = str_df.apply(get_cross, args=(feature_infos,), axis=0)
        data_new = pd.concat([data_new, cross_df], axis=1)
        print(f'After wide cross, data shape:{data_new.shape}')
        # data_new.to_csv(os.path.join(
        #     out_dir, f'{label_id}_feature_data_widecross_{mode}.csv'), index=False)
        # print(f'feature data saved.')

    # deep 特征分桶
    if params['bucket']:
        def get_bucket(x, d_name, feature_infos):
            d_min, d_max = feature_infos[d_name]['min'], feature_infos[d_name]['max']
            if x[0] > d_max:
                return 11
            elif x[0] < d_min:
                return 0
            elif x[0] == d_max:
                return 10
            else:
                return int(10 * (x[0] - d_min) / (d_max - d_min)) + 1
        bucket_df = pd.DataFrame()
        for d_feat in deep_imp_feats:
            bucket_df['bu_' + d_feat] = data_new[[d_feat]].apply(get_bucket, args=(d_feat, feature_infos,), axis=1)
        data_new = pd.concat([data_new, bucket_df], axis=1)
        print(f'After bucket, data shape:{data_new.shape}')

    # bucket与wide交叉
    if params['b_w cross']:
        bucket_list = ['bu_' + d for d in deep_imp_feats]
        bw_str_df = pd.DataFrame()
        for i in range(len(bucket_list)):
            for j in range(len(wide_imp_feats)):
                i_name, j_name = bucket_list[i], wide_imp_feats[j]
                bw_str_df['bw_' + i_name.split('_')[-1] + '_' + j_name] = data_new[i_name].astype(
                    str).values + '_' + data_new[j_name].astype(str).values
        def get_bw_cross(x, feature_infos):
            j_name = x.name.split('_')[2]
            i_list, j_list = list(range(12)), feature_infos[j_name]['list'] # 注意都是分10个桶，加更大和更小值
            out = []
            for one in x:
                i, j = int(one.split('_')[0]), int(one.split('_')[1])
                if i not in i_list or j not in j_list:
                    out.append(0)
                else:
                    out.append(i_list.index(i) * len(j_list) + j_list.index(j) + 1)
            return out
        bw_cross_df = bw_str_df.apply(get_bw_cross, args=(feature_infos,), axis=0)
        data_new = pd.concat([data_new, bw_cross_df], axis=1)
        print(f'After b_w cross, data shape:{data_new.shape}')

    # bucket之间交叉
    if params['b_b cross']:
        bb_str_df = pd.DataFrame()
        for i in range(len(bucket_list)-1):
            for j in range(i+1, len(bucket_list)):
                i_name, j_name = bucket_list[i], bucket_list[j]
                bb_str_df['bb_'+i_name.split('_')[-1]+'_'+j_name.split('_')[-1]] = data_new[i_name].astype(str).values + '_' + data_new[j_name].astype(str).values
        def get_bb_cross(x):
            i_list, j_list = list(range(12)), list(range(12)) # 注意都是分10个桶，加更大和更小值
            out = []
            for one in x:
                i, j = int(one.split('_')[0]), int(one.split('_')[1])
                if i not in i_list or j not in j_list:
                    out.append(0)
                else:
                    out.append(i_list.index(i) * len(j_list) + j_list.index(j)) # 注意不加1了，因为bucket已经考虑未出现值
            return out
        bb_cross_df = bb_str_df.apply(get_bb_cross, axis=0)
        data_new = pd.concat([data_new, bb_cross_df], axis=1)
        print(f'After b_b cross, data shape:{data_new.shape}')

    data_new.to_csv(os.path.join(
        out_dir, f'{label_id}_feature_data_all_{mode}.csv'), index=False)
    print(f'feature all data saved.')


if __name__ == '__main__':
    # 执行特征工程，并且通过params确定使用哪些特征工程方法
    mode = sys.argv[1]
    # label_id = sys.argv[2]
    out_dir = './data/featured_data/'
    params = {
        'drop_1500': True,
        'wide_cross': True,
        'bucket': True,
        'b_w cross': True,
        'b_b cross': True,
    }
    data_loader = myDataLoader('./data/molecule_open_data')
    train_data, test_data = data_loader.dataset_for_boost(
        train_file='candidate_train.csv', test_file='candidate_test_clean.csv', label_file='train_answer.csv')
    print(f'Shape of train data: {train_data.shape}')
    print(f'Shape of test data: {test_data.shape}')

    # 多线程
    label_list = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
    pool = multiprocessing.Pool(processes=6)

    if mode == 'train':
        for label_id in label_list:
            print(f'make features of train data for {label_id}...')
            pool.apply_async(make_features, (train_data, params, out_dir, label_id, mode, ))
        pool.close()
        pool.join()
    elif mode == 'test':
        for label_id in label_list:
            print(f'make features of test data for {label_id}...')
            pool.apply_async(make_features, (test_data, params, out_dir, label_id, mode, ))
        pool.close()
        pool.join()
    else:
        raise ValueError
