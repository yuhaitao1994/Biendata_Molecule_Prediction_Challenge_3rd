# coding=utf-8
"""
deep_model_2，与模型1不同的是，全面放弃one-hot，所有特征数值化，并且加入cat选择的头部特征的各种特征工程方法，最佳单模型。
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
import shutil  # 清空文件夹
import psutil  # 查看占用内存
from tqdm import tqdm
from sklearn.model_selection import KFold
from utils import get_emb_id, cross_feature, norm_and_smape, SMAPE


class deep_model_explore(tf.keras.Model):
    """deep模型, 带embedding"""
    def __init__(self):
        """初始化layers"""
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(
            units=1024*4, name='deep_1', activation=tf.keras.activations.relu,
            kernel_regularizer=tf.keras.regularizers.l2(1.0))
        self.dense_add = tf.keras.layers.Dense(
            units=512*4, name='deep_add', activation=tf.keras.activations.relu,
            kernel_regularizer=tf.keras.regularizers.l2(1.0))
        self.dense_2 = tf.keras.layers.Dense(
            units=256*4, name='deep_2', activation=tf.keras.activations.relu,
            kernel_regularizer=tf.keras.regularizers.l2(1.0))
        self.dense_add_2 = tf.keras.layers.Dense(
            units=128*4, name='deep_add_2', activation=tf.keras.activations.relu,
            kernel_regularizer=tf.keras.regularizers.l2(1.0))
        self.dense_3 = tf.keras.layers.Dense(
            units=1, name='deep_3', activation=None)

    def call(self, inputs, training=False):
        """模型调用"""
        x, id_, y_t = inputs
        deep_part = self.dense_1(x)
        deep_part = self.dense_add(deep_part)
        deep_part = self.dense_2(deep_part)
        deep_part = self.dense_add_2(deep_part)
        deep_part = self.dense_3(deep_part)
        out = deep_part
        return out, id_, y_t


def make_tfrecords_dataset(tfrecords_file, batch_size, input_size, label_id, is_shuffle=True):
    """
    构造dataset
    """
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'inputs': tf.io.FixedLenFeature([input_size], tf.float32),  # 注意shape
        'labels': tf.io.FixedLenFeature([6], tf.float32),
        'id_': tf.io.FixedLenFeature([], tf.string)
    }
    i = int(label_id[-1]) - 1

    def _parse_example(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        return (feature_dict['inputs'], [feature_dict['id_']], [feature_dict['labels'][i]]), [feature_dict['labels'][i]]

    dataset = tf.data.TFRecordDataset(tfrecords_file)
    dataset = dataset.map(
        _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    if is_shuffle:
        dataset = dataset.shuffle(buffer_size=15000).repeat()
    dataset = dataset.prefetch(buffer_size=batch_size * 8)
    return dataset


def wnd_train(tfrecords_dir, wnd_params, size_dic, label_id):
    """
    wide & deep 模型训练
    """
    n_fold = wnd_params['n_fold']
    batch_size = wnd_params['batch_size']
    with open('./feature_info.json', 'r') as f:
        feature_infos = json.load(f)

    # 模型与logs保存
    model_path = wnd_params['model_path']
    log_path = wnd_params['log_path']
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file = open(os.path.join(log_path, 'print.log'), 'w')
    stdout_backup = sys.stdout
    sys.stdout = log_file

    smape_score = np.zeros(n_fold)  # 评估分数

    # 训练k_fold, 因为已经划分tfrecords了
    for fold_idx in range(n_fold):
        time_stamp = datetime.datetime.now()
        print('*' * 120)
        print(f"Fold [{fold_idx}]: " +
              time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
        # 每个fold的model
        model_path_f = os.path.join(model_path, f'fold_{fold_idx}')
        log_path_f = os.path.join(log_path, f'fold_{fold_idx}')
        if not os.path.exists(model_path_f):
            os.mkdir(model_path_f)
        else:
            shutil.rmtree(model_path_f)
            os.mkdir(model_path_f)
        if not os.path.exists(log_path_f):
            os.mkdir(log_path_f)
        else:
            shutil.rmtree(log_path_f)
            os.mkdir(log_path_f)

        # 加载dataset
        train_set = make_tfrecords_dataset(os.path.join(
            tfrecords_dir, f'{label_id}_norm_train_{size_dic[label_id]}_fold_{fold_idx}.tfrecords'), batch_size=batch_size, input_size=size_dic[label_id], label_id=label_id, is_shuffle=True)
        val_set = make_tfrecords_dataset(os.path.join(
            tfrecords_dir, f'{label_id}_norm_val_{size_dic[label_id]}_fold_{fold_idx}.tfrecords'), batch_size=batch_size * 16, input_size=size_dic[label_id], label_id=label_id, is_shuffle=False)
        train_iter = iter(train_set)  # 为训练集构造迭代器

        # 创建model
        model = deep_model_explore()
        deep_optimizer = tf.keras.optimizers.Adam(
            learning_rate=wnd_params['learning_rate'])
        # checkpoint保存参数
        checkpoint = tf.train.Checkpoint(mymodel=model)
        manager = tf.train.CheckpointManager(
            checkpoint, directory=model_path_f, max_to_keep=1)
        # 记录summary
        summary_writer = tf.summary.create_file_writer(log_path_f)     # 实例化记录器

        @tf.function
        def train_one_step(x, y):
            """训练一次, 静态图模式"""
            with tf.GradientTape() as tape:
                y_pred, id_, y_t = model(x, training=True)
                loss = tf.reduce_mean(
                    tf.keras.losses.MAE(y_true=y, y_pred=y_pred))
            deep_grads = tape.gradient(loss, model.trainable_weights)
            deep_optimizer.apply_gradients(
                grads_and_vars=zip(deep_grads, model.trainable_weights))
            return y_pred, loss

        # 定义一些训练时的指标
        loss_record, smape_record, min_val_smape, early_stop_rounds = 0., 0., 200.0, 0
        s_time = datetime.datetime.now()
        # 迭代
        for i in range(wnd_params['iterations']):
            x, y = next(train_iter)
            y_pred, loss = train_one_step(x, y)

            loss_record += loss
            smape_record += norm_and_smape(y, y_pred, label_id,
                                           feature_infos, norm_type='var', mode="norm_all")
            # eval
            if i % 1000 == 0 and i != 0:
                val_out = model.predict(x=val_set)
                val_preds = val_out[0].squeeze()
                val_true = val_out[2].squeeze()
                # 计算分数
                val_smape = norm_and_smape(
                    val_true, val_preds, label_id, feature_infos, norm_type='var', mode='norm_all')
                e_time = datetime.datetime.now()
                mem_use = psutil.Process(
                    os.getpid()).memory_info().rss / (1024**3)
                print(f'steps: {i}, train_loss: {loss_record / 1000}, train SMAPE: {(smape_record / 1000):.6f}, val SMAPE: {val_smape:.6f}, time_cost: {(e_time-s_time)}, memory_use: {mem_use:.4f}')
                with summary_writer.as_default():
                    tf.summary.scalar("mean_train_loss",
                                      (loss_record / 1000), step=i)
                    tf.summary.scalar(
                        "SMAPE/train", (smape_record / 1000), step=i)
                    tf.summary.scalar("SMAPE/val", val_smape, step=i)
                    tf.summary.scalar("memory_use", mem_use, step=i)
                # 模型保存与early stop
                if val_smape < min_val_smape:
                    min_val_smape = val_smape
                    manager.save(checkpoint_number=i)
                    smape_score[fold_idx] = val_smape
                    early_stop_rounds = 0
                else:
                    early_stop_rounds += 1
                    if early_stop_rounds >= 50:
                        break
                loss_record, smape_record = 0.0, 0.0
                s_time = datetime.datetime.now()
        print(f'best smape score of fold_{fold_idx}: {min_val_smape:.6f}')
        tf.keras.backend.clear_session()  # 清理内存
        del model, deep_optimizer, train_set, val_set, checkpoint, summary_writer, manager
        gc.collect()

    # 总的smape
    print('*' * 120)
    print(f'Mean smape in each fold of {label_id}: {np.mean(smape_score)}')
    # 输出重定向结束
    log_file.close()
    sys.stdout = stdout_backup


def predict_one_fold(fold_idx, label_id, model_dir, test_set, tfrecords_dir, size_dic):
    # 加载模型
    val_set = make_tfrecords_dataset(os.path.join(
            tfrecords_dir, f'{label_id}_norm_val_{size_dic[label_id]}_fold_{fold_idx}.tfrecords'), batch_size=1024, input_size=size_dic[label_id], label_id=label_id, is_shuffle=False)

    model = deep_model_explore()

    checkpoint = tf.train.Checkpoint(mymodel=model)
    model_dir = os.path.join(model_dir, f'fold_{fold_idx}')
    checkpoint.restore(tf.train.latest_checkpoint(model_dir))
    # val 预测
    val_out = model.predict(x=val_set)
    val_preds = val_out[0].squeeze()
    val_true = val_out[2].squeeze()
    # test 预测
    test_out = model.predict(x=test_set)
    test_preds = test_out[0].squeeze() / 5
    test_ids = test_out[1].squeeze()
    return val_preds, val_true, test_preds, test_ids


def wnd_predict(tfrecords_dir, label_id, model_dir, size_dic):
    """
    加载选定的模型输出预测结果，并保存到result文件
    参数：
    tfrecords_dir 存tfrecords的目录
    model_dir: 模型路径
    """
    with open('./feature_info.json', 'r') as f:
        feature_infos = json.load(f)
    test_preds, test_ids = None, None 
    local_smape = np.zeros(5)
    test_set = make_tfrecords_dataset(os.path.join(
            tfrecords_dir, f'{label_id}_norm_test_{size_dic[label_id]}.tfrecords'), batch_size=1024, input_size=size_dic[label_id], label_id=label_id, is_shuffle=False)
    # 每个fold对验证集进行预测
    print(f'Deep Model Predicting for {label_id} ......')
    for fold_idx in range(5):
        val_preds, val_true, one_test, one_test_id = predict_one_fold(fold_idx, label_id, model_dir, test_set, tfrecords_dir, size_dic)
        if fold_idx == 0:
            test_preds = one_test
            test_ids = one_test_id
            print(f'Test data num: {test_preds.shape}')
        else:
            test_preds += one_test

        # 计算每个指标的local score
        local_smape[fold_idx] = norm_and_smape(val_true, val_preds, label_id, feature_infos, norm_type='var', mode='norm_all')
        print(f'SMAPE score of fold_{fold_idx}: {local_smape[fold_idx]:.6f}')

        tf.keras.backend.clear_session()  # 清理内存
        gc.collect()

    print(f'Local SMAPE score of {label_id} is {np.mean(local_smape):.6f}')

    return test_preds, test_ids, np.mean(local_smape)


def wnd_main(mode=None, label_str=''):
    """
    读取数据，确定参数，并且控制训练或预测
    参数：
    mode: train / predict
    label_str: 字符串，手动输入训练哪个参数
    """
    size_dic = {
        'p1': 2871,
        'p2': 2867,
        'p3': 2863,
        'p4': 2861,
        'p5': 2860,
        'p6': 2860
    }
    if mode == 'train':
        # 动态显存分配
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(devices=gpus[int(label_str[-1])%2], device_type='GPU')
        tf.config.experimental.set_memory_growth(device=gpus[int(label_str[-1])%2], enable=True)
        # 多线程实现不执行，所以一次只训练一个指标
        model_dir = f'./models/wnd/{label_str}'
        log_dir = f'./logs/wnd/{label_str}'
        tfrecords_dir = f'./data/tfrecords/{label_str}'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        # 定义cat_params
        wnd_params = {
            'n_fold': 5,
            'iterations': 300000,
            'batch_size': 64,
            'learning_rate': 0.0001,
            'model_path': os.path.join(model_dir, 'deep_model_2'),
            'log_path': os.path.join(log_dir, 'deep_model_2'),
        }
        print(f"Training {label_str} ...")
        wnd_train(tfrecords_dir, wnd_params, size_dic, label_id=label_str)

    elif mode == 'predict':
        """预测"""
        with open('./feature_info.json', 'r') as f:
            feature_infos = json.load(f)
        # 动态显存分配
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

        label_list = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
        model_dic = {
            'p1': 'deep_model_2',
            'p2': 'deep_model_2',
            'p3': 'deep_model_2',
            'p4': 'deep_model_2',
            'p5': 'deep_model_2',
            'p6': 'deep_model_2',
        }

        test_ids = None
        test_preds = None
        mean_smape = np.zeros(6)

        for i in range(len(label_list)):
            tfrecords_dir = f'./data/tfrecords/{label_list[i]}'
            model_dir = f'./models/wnd/{label_list[i]}/{model_dic[label_list[i]]}'
            local_test, one_test_ids, local_smape = wnd_predict(tfrecords_dir, label_list[i], model_dir, size_dic)
            if i == 0: 
                test_ids = one_test_ids
                test_preds = np.zeros((len(local_test), 6)) 
            test_preds[:, i] = local_test * feature_infos[label_list[i]]['std'] + feature_infos[label_list[i]]['mean']
            mean_smape[i] = local_smape

        print('*' * 120)
        print(f'Deep Model Mean local SMAPE score is {np.mean(mean_smape):.6f}')
        # 提交文件
        id_df = pd.DataFrame(test_ids.astype(str), columns=['id'])
        pred_df = pd.DataFrame(test_preds, columns=label_list)
        result = pd.concat([id_df, pred_df], axis=1).reset_index(drop=True)
        result.to_csv(f'./results/result_deep_model_2.csv', index=False)
        print(result.head())


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    wnd_main(mode=sys.argv[1], label_str=sys.argv[2])
