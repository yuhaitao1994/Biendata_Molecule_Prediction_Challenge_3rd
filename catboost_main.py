# coding=utf-8
"""
catboost方法的训练与评估函数，用于生成每个指标的feature importance。
借鉴官网上的baseline：https://www.biendata.com/models/category/4068/L_notebook/
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
import shutil # 清空文件夹
import psutil # 查看占用内存
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold

from data_loader import myDataLoader
from utils import SMAPE


def catboost_train(train_data, cat_params, label_id):
    """
    使用catboost算法进行训练与评估,保存模型，记录logs
    参数：
    train_data: 训练数据dataFrame
    cat_params: 超参数字典
    label_id: 标签的名称
    """
    n_fold = cat_params['n_fold']

    # 模型与logs保存
    model_path = cat_params['model_path']
    log_path = cat_params['log_path']
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file = open(os.path.join(log_path, 'print.log'), 'w')
    stdout_backup = sys.stdout
    sys.stdout = log_file

    # 保存特征重要性的dataFrame
    feature_imps = pd.DataFrame()
    # 评估分数
    smape_score = np.zeros(n_fold)
    # 划分k_fold
    all_index = range(len(train_data))
    k_fold = KFold(n_splits=n_fold, shuffle=True, random_state=1)
    # 训练k_fold
    for fold_idx, (train_idx, val_idx) in enumerate(k_fold.split(all_index)):
        time_stamp = datetime.datetime.now()
        print('*' * 120)
        print(f"Fold [{fold_idx}]: " +
              time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))

        cat_info_path = os.path.join(log_path, f'fold_{fold_idx}')
        if not os.path.exists(cat_info_path):
            os.mkdir(cat_info_path)
        else:
            shutil.rmtree(cat_info_path)
            os.mkdir(cat_info_path)

        train_fold = train_data.iloc[train_idx]
        val_fold = train_data.iloc[val_idx]
        # 提取使用特征的columns
        use_cols = [col for col in train_data.columns if col !=
                    'id' and 'p' not in col]
        print(f'Number of common used features: {len(use_cols)}')
        # 划分x,y标签
        train_x, val_x = train_fold[use_cols], val_fold[use_cols]
        train_y, val_y = train_fold[label_id], val_fold[label_id]
        # 不知道这个Pool的意思
        cate_features = []
        train_pool = Pool(train_x, train_y, cat_features=cate_features)
        val_pool = Pool(val_x, val_y, cat_features=cate_features)
        # 模型
        cbt_model = CatBoostRegressor(iterations=cat_params['iterations'], 
                                      learning_rate=cat_params['learning_rate'],
                                      eval_metric='SMAPE',
                                      use_best_model=True,
                                      early_stopping_rounds=2000,
                                      random_seed=cat_params['random_seed'],
                                      logging_level='Info',
                                      task_type='GPU',
                                      devices=cat_params['gpu_devices'],
                                      gpu_ram_part=0.25,
                                      train_dir=cat_info_path,
                                      depth=cat_params['depth'],
                                      l2_leaf_reg=cat_params['l2_leaf_reg'],
                                      loss_function=cat_params['loss_function']
                                      )
        cbt_model.fit(train_pool, eval_set=val_pool, metric_period=500, verbose=1000)
        smape_score[fold_idx] = cbt_model.best_score_['validation']['SMAPE']
        # 模型保存
        with open(os.path.join(model_path, f'cbt_fold_{fold_idx}.pkl'), 'wb') as f:
            pickle.dump(cbt_model, f)
        # 记录特征重要性
        if fold_idx == 0:
            feature_imps['feature'] = use_cols
        feature_imps[f'score{fold_idx}'] = cbt_model.feature_importances_
        # 清理内存
        del cbt_model, train_pool, val_pool
        del train_x, train_y, val_x, val_y
        gc.collect()
        # 记录每个fold的smape
        print(f'smape_score of {label_id}: {smape_score[fold_idx]:.6f}')

    # 总的smape
    print('*' * 120)
    print(f'Mean smape in each fold of {label_id}: {np.mean(smape_score)}')
    # 输出特征相关性到文件
    feature_imps['score_mean'] = feature_imps.apply(lambda x: np.sum(x.values[1:]) / 5, axis=1)
    feature_imps = feature_imps.sort_values(
        by='score_mean', ascending=False).reset_index(drop=True)
    feature_imps.to_csv(os.path.join(log_path, 'feature_imps.csv'), index=False)
    print(feature_imps.head(20))
    # 输出重定向结束
    log_file.close()
    sys.stdout = stdout_backup


def SMAPE(y_true, y_pred):
    """
    手动实现SMAPE计算
    """
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def catboost_predict(train_data, test_data, label_id, model_dir):
    """
    加载选定的模型输出预测结果，并保存到result文件
    """
    test_preds = np.zeros(len(test_data))
    use_cols = [col for col in train_data.columns if col !=
                'id' and 'p' not in col]
    print(f'Number of common used features: {len(use_cols)}')
    # 划分k_fold
    all_index = range(len(train_data))
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    local_smape = np.zeros(5)

    # 每个指标都按原方式对验证集进行预测
    for fold_idx, (train_idx, val_idx) in enumerate(k_fold.split(all_index)):
        val_fold = train_data.iloc[val_idx]
        val_x = val_fold[use_cols]
        # 加载模型
        cbt_model = pickle.load(open(os.path.join(model_dir, f'cbt_fold_{fold_idx}.pkl'), 'rb'))
        # 预测
        val_preds = cbt_model.predict(val_x)
        test_preds += cbt_model.predict(test_data[use_cols]) / 5
        
        local_smape[fold_idx] = SMAPE(y_true=val_fold[label_id], y_pred=val_preds)
        print(f'SMAPE score of fold_{fold_idx}: {local_smape[fold_idx]:.6f}')

    print(f'Local SMAPE score of {label_id} is {np.mean(local_smape):.6f}')
    return test_preds, np.mean(local_smape)


def catboost_main(mode=None, label_str=''):
    """
    读取数据，确定参数，并且控制训练或预测
    参数：
    mode: train / predict
    label_str: 字符串，手动输入训练哪个参数
    """
    # 加载数据
    data_loader = myDataLoader('./data/molecule_open_data')
    train_data, test_data = data_loader.dataset_for_boost(
        train_file='candidate_train.csv', test_file='candidate_val.csv', label_file='train_answer.csv')
    print(f'Shape of train data: {train_data.shape}')
    print(f'Shape of test data: {test_data.shape}')

    if mode == 'train':
        model_dir = f'./models/catboost/{label_str}'
        log_dir = f'./logs/catboost/{label_str}'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        # 定义cat_params
        cat_params = {
            'n_fold': 5,
            'iterations': 250000,
            'gpu_devices': str(int(label_str[-1]) % 2),
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 150.0,
            'random_seed': 42,
            'loss_function': 'RMSE',
            'model_path': os.path.join(model_dir, '0409_cat_lr005_dep8_l2150'),
            'log_path': os.path.join(log_dir, '0409_cat_lr005_dep8_l2150'),
        }
        print(f"Catboost Training {label_str} ...")
        catboost_train(train_data, cat_params, label_str)

    elif mode == 'predict':
        label_list = ['p1','p2','p3','p4','p5','p6']
        model_dic = {
            'p1': '0409_cat_lr005_dep8_l260',
            'p2': '0409_cat_lr005_dep8_l260',
            'p3': '0409_cat_lr005_dep8_l260',
            'p4': '0409_cat_lr005_dep8_l260',
            'p5': '0409_cat_lr005_dep8_l260',
            'p6': '0409_cat_lr005_dep8_l260',
        }
        test_preds = np.zeros((len(test_data), 6))
        mean_smape = np.zeros(6)

        for i in range(len(label_list)):
            model_dir = f'./models/catboost/{label_list[i]}/{model_dic[label_list[i]]}'
            local_test, local_smape = catboost_predict(train_data, test_data, label_list[i], model_dir)
            test_preds[:, i] = local_test
            mean_smape[i] = local_smape
        print('*' * 120)
        print(f'Mean local SMAPE score is {np.mean(mean_smape):.6f}')
        # 提交文件
        pred_df = pd.DataFrame(test_preds, columns=[f'p{i+1}' for i in range(6)])
        result = pd.concat([test_data[['id']], pred_df], axis=1).reset_index(drop=True)
        r_name = model_dic['p3']
        result.to_csv(f'./results/catboost/result_{r_name}_{np.mean(mean_smape):.4f}.csv', index=False)
        print(result.head())


if __name__ == '__main__':
    catboost_main(mode=sys.argv[1], label_str=sys.argv[2])
