# coding: utf-8
from __future__ import division, print_function
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import get_max_min
import utilities
import get_envelop

PLOT_ENVELOP = False

def str_in_str(list_of_str, str):
    results = []
    for s in list_of_str:
        results.append(s in str)
    return np.any(results)

if __name__=='__main__':
    nb_channels = 16
    cut_len = 64
    fs = 200 # 采样频率
    ma_len = 10 # 求envelop里面mean average的长度，越长数据越平滑

    data_dir = './raw_data'
    save_dir = './processed_data/envelop/divide_by_subject'
    mat_path = utilities.walk_through_dir(data_dir)

    # 选取subject的id
    nb_subjects = 10
    nb_subjects_for_train = 7
    subject_id = np.arange(nb_subjects)
    np.random.shuffle(subject_id)

    subject_id_for_train = []
    for i in range(nb_subjects_for_train):
        subject_id_for_train.append('S' + str(subject_id[i] + 1))

    EMG_train = []
    LABEL_train = []
    EMG_test = []
    LABEL_test = []

    for path in mat_path:
        emg, label = utilities.read_mat(path) # emg, label均是二维矩阵

        for i in range(nb_channels):
            emg[:, i] = get_envelop.envelop(emg[:, i], fs, ma_len)

        if PLOT_ENVELOP:
            plt.figure()
            plt.plot(emg[:, 0])
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

        if str_in_str(subject_id_for_train, path):    
            EMG_train.append(emg_cut)
            LABEL_train.append(label_cut)
        else:
            EMG_test.append(emg_cut)
            LABEL_test.append(label_cut)            
    
    EMG_train = np.concatenate(EMG_train)
    LABEL_train = np.concatenate(LABEL_train)

    EMG_test = np.concatenate(EMG_test)
    LABEL_test = np.concatenate(LABEL_test)

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

    EMG_train, LABEL_train = reduce_rest_movement(EMG_train, LABEL_train)
    EMG_test, LABEL_test = reduce_rest_movement(EMG_test, LABEL_test)

    # 将EMG归一化到[0, 1]之间
    max_value = np.max(EMG_train)
    EMG_train = EMG_train / max_value
    EMG_test = EMG_test / max_value

    EMG_train[EMG_train > 1] = 1
    EMG_train[EMG_train < 0] = 0

    EMG_test[EMG_test > 1] = 1
    EMG_test[EMG_test < 0] = 0

    # 保存
    np.save(os.path.join(save_dir, 'EMG_train.npy'), EMG_train)
    np.save(os.path.join(save_dir, 'label_train.npy'), LABEL_train)

    np.save(os.path.join(save_dir, 'EMG_test.npy'), EMG_test)
    np.save(os.path.join(save_dir, 'label_test.npy'), LABEL_test)

    print('EMG train shape: ', EMG_train.shape)
    print('LABEL train shape: ', LABEL_train.shape)

    print('EMG test shape: ', EMG_test.shape)
    print('LABEL test shape: ', LABEL_test.shape)


