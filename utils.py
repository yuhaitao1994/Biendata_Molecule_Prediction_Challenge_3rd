# coding=utf-8
"""
一些工具
@author: yuhaitao
"""
import pandas as pd
import os
import numpy as np
import gc
import pickle
import datetime
import sys
import multiprocessing
import json
import tensorflow as tf
import shutil # 清空文件夹
import psutil # 查看占用内存
from data_loader import myDataLoader, var_norm, min_max_norm


def get_emb_id(one_hot_df, top_list, feature_infos):
    """
    获取交叉特征的id，用于embedding
    返回词表数mul，和一个pd.Series，保存每条数据的emb_id
    """
    top_dic = {}
    for a in top_list:
        top_dic[a] = feature_infos[a]['list']
    df_list = []
    for f in top_dic:
        one_list = []
        for w in one_hot_df.columns:
            if 'w'+f+'_' in w:
                one_list.append(w)
        df_list.append(one_hot_df[one_list])

    def cal_one_row(x):
        """计算每个特征的取值"""
        index = 1
        base = 0
        for one in x:
            base += one * index
            index += 1
        return base

    cal_df = pd.DataFrame()
    for i in range(len(df_list)):
        cal_df[top_list[i]] = df_list[i].apply(cal_one_row, axis=1)
    mul = 1
    for k, v in top_dic.items():
        mul *= (len(v) + 1)

    def cal_to_id(x, top_list, top_dic, mul):
        """计算emb id"""
        pos = 0
        for i, (one, name) in enumerate(zip(x, top_list)):
            mul /= (len(top_dic[name])+1)
            pos += int(one * mul)
        return pos
    
    return mul, cal_df.apply(cal_to_id, axis=1, args=(top_list, top_dic, mul, ))


def cross_feature(one_hot_df, feat_list, feature_infos):
    """将list中的重要特征两两交叉，然后并入wide_df"""
    def cal_one_row(x):
        """计算每个特征的取值"""
        index = 1
        base = 0
        for one in x:
            base += one * index
            index += 1
        return base
    def get_cross_df(a, b):
        """
        将两个特征交叉
        输入x，y是wide_df中两个特征的one-hot子dataframe
        输出：一个交叉之后的dataframe
        """
        new_len = (a.shape[1]+1) * (b.shape[1]+1) # 注意交叉的时候特征为空的处理
        new_name = a.columns[0].split('_')[0] + '_' + b.columns[0].split('_')[0]
        out_df = pd.DataFrame(columns=[new_name + '_' + str(i) for i in range(new_len) if int(i/(b.shape[1]+1)) != 0 and i%(b.shape[1]+1) != 0])
        a_i = a.apply(cal_one_row, axis=1)
        b_i = b.apply(cal_one_row, axis=1)
        cross = pd.DataFrame(columns=['a','b'], data=zip(a_i.values, b_i.values))
        cross = pd.DataFrame(data=cross.apply(lambda x:x[0]*(b.shape[1]+1)+x[1], axis=1), columns=[new_name])
        # 然后还是像one hot那样，主要的目的是处理缺失的值
        cross_one_hot = pd.get_dummies(cross.astype(int).astype(str))
        _, out_df = out_df.align(cross_one_hot, join='left', axis=1, fill_value=0)
        out_df = out_df.astype(np.float32)
        return out_df

    cross_df = pd.DataFrame()
    df_list = []
    for f in feat_list:
        one_list = []
        for w in one_hot_df.columns:
            if 'w'+f+'_' in w:
                one_list.append(w)
        df_list.append(one_hot_df[one_list])

    for i in range(len(feat_list)-1):
        for j in range(i+1, len(feat_list)):
            cross_df = pd.concat([cross_df, get_cross_df(df_list[i], df_list[j])], axis=1)
    return cross_df


def SMAPE(y_true, y_pred):
    """
    手动实现SMAPE计算
    """
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def norm_and_smape(y_true, y_pred, label_id, feature_infos, norm_type='var', mode='norm_all'):
    """
    先转换为原始量纲，再手动实现SMAPE计算
    """
    if norm_type == 'min_max':
        l_max = feature_infos[label_id]['max']
        l_min = feature_infos[label_id]['min']
        y_pred = y_pred * (l_max - l_min) + l_min
        if mode == 'norm_all':
            y_true = y_true * (l_max - l_min) + l_min
    elif norm_type == 'var':
        l_std = feature_infos[label_id]['std']
        l_mean = feature_infos[label_id]['mean']
        y_pred = y_pred * l_std + l_mean
        if mode == 'norm_all':
            y_true = y_true * l_std + l_mean
    elif norm_type == 'no_norm':
        pass
    else:
        raise ValueError
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

