# coding: utf-8
from __future__ import division, print_function
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import get_max_min
import utilities
import get_envelop

PLOT_ENVELOP = True

if __name__=='__main__':
    nb_channels = 16
    cut_len = 64
    fs = 200 # 采样频率
    ma_len = 10 # 求envelop里面mean average的长度，越长数据越平滑

    data_dir = './raw_data'
    save_dir = './processed_data/envelop'
    mat_path = utilities.walk_through_dir(data_dir)

    EMG = []
    LABEL = []

    for path in mat_path:
        emg, label = utilities.read_mat(path) # emg, label均是二维矩阵

        emg_raw = emg.copy()
        for i in range(nb_channels):
            emg[:, i] = get_envelop.envelop(emg[:, i], fs, ma_len)

        if PLOT_ENVELOP:
            plt.figure()
            plt.plot(emg_raw[0:4000, 0])
            plt.plot(emg[0:4000, 0])
            plt.show()
        
        emg_cut, label_cut = utilities.process_emg_according_to_label(emg, label, cut_len)

        if 'E1' in path:
            pass
        elif 'E2' in path:
            for i, _label in enumerate(label_cut):
                if _label != 0:
                    label_cut[i] = label_cut[i] + 12
        elif 'E3' in path:
            for i, _label in enumerate(label_cut):
                if _label != 0:
                    label_cut[i] = label_cut[i] + 12 + 17
            
        EMG.append(emg_cut)
        LABEL.append(label_cut)

    
    EMG = np.concatenate(EMG)
    LABEL = np.concatenate(LABEL)

    # 减少rest即label=0的动作
    def reduce_rest_movement(emg, label):
        emg_rest = emg[label == 0]
        label_rest = label[label == 0]

        pick_random_amount = int(np.sum(label == 1))
        pick_random_index = np.random.choice(len(label_rest), pick_random_amount)
        emg_rest = emg_rest[pick_random_index]
        label_rest = label_rest[pick_random_index]

        emg = emg[label > 0]
        label = label[label > 0]

        emg = np.concatenate((emg, emg_rest))
        label = np.concatenate((label, label_rest))
        return emg, label

    EMG, LABEL = reduce_rest_movement(EMG, LABEL)

    # 将EMG Envelop变换到 [0, 1]之间
    EMG_normalize = EMG / np.max(EMG)
    EMG_normalize[EMG_normalize > 1] = 1
    EMG_normalize[EMG_normalize < 0] = 0

    # 保存在一个文件
    np.save(os.path.join(save_dir, 'EMG.npy'), EMG_normalize)
    np.save(os.path.join(save_dir, 'label.npy'), LABEL)

    print('EMG shape: ', EMG.shape)
    print('LABEL shape: ', LABEL.shape)

    # 分为测试集和验证集
    nb_samples = len(LABEL)
    train_len = int(0.7 * nb_samples)

    id_list = np.arange(nb_samples)
    np.random.shuffle(id_list)

    train_id_list = id_list[:train_len]
    test_id_list = id_list[train_len:]

    EMG_train = EMG[train_id_list]
    LABEL_train = LABEL[train_id_list]

    EMG_test = EMG[test_id_list]
    LABEL_test = LABEL[test_id_list]

    #对测试集和验证机归一化
    max_value = np.max(EMG_train)
    EMG_train = EMG_train / max_value
    EMG_test = EMG_test / max_value

    EMG_train[EMG_train > 1] = 1
    EMG_train[EMG_train < 0] = 0

    EMG_test[EMG_test > 1] = 1
    EMG_test[EMG_test < 0] = 0

    np.save(os.path.join(save_dir, 'EMG_train.npy'), EMG_train)
    np.save(os.path.join(save_dir, 'label_train.npy'), LABEL_train)

    np.save(os.path.join(save_dir, 'EMG_test.npy'), EMG_test)
    np.save(os.path.join(save_dir, 'label_test.npy'), LABEL_test)
