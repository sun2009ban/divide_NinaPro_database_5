import numpy as np
import scipy.io as sio
import os

FILE_EXTENSIONS = [".mat"]

def is_file(filename):
    '''
    判断filename是否是以FILE_EXTENSIONS中的为结尾
    '''
    return any(filename.endswith(extension) for extension in FILE_EXTENSIONS)

def walk_through_dir(directory):
    '''
    遍历目录dir下面的以FILE_EXTENSIONS为结尾的文件
    返回值为文件的路径
    '''
    file_path = []

    for root, _, fnames in sorted(os.walk(directory)):
        for fname in sorted(fnames):
            if is_file(fname):
                path = os.path.join(directory, fname) #把目录和
                file_path.append(path)

    return file_path

def read_mat(filepath):
    '''
    读取mat文件，并处理标签
    '''
    emg = sio.loadmat(filepath)['emg']
    label = sio.loadmat(filepath)['restimulus'] 

    return emg, label

def cut_on_first_dim(data, cut_length=100):
    '''
    将第一个维度的数据按照cut_length这个长度剪裁
    '''
    length = len(data)
    nb_cut = length // cut_length
    cut_data = np.zeros((nb_cut, cut_length, *data.shape[1:]))

    for i in range(nb_cut):
        cut_data[i, :, :] = data[i*cut_length: (i+1)*cut_length , :]
    return cut_data

def process_emg_according_to_label(emg, label, cut_length=100):
    '''
    按照emg的标签，把同一个标签的数据堆到一起
    '''
    emg_list = []
    label_list = []

    nb_labels = np.max(label) + 1 # 注意，标签里面有0的，因此需要加1
    
    for i in range(int(nb_labels)):
        _emg = emg[np.squeeze(label) == i] # label是 n x 1的形式，因此利用np.squeeze()处理
        _cut_emg = cut_on_first_dim(_emg, cut_length)
        _label = i * np.ones((_cut_emg.shape[0]))

        emg_list.append(_cut_emg)
        label_list.append(_label)

    return np.concatenate(emg_list), np.concatenate(label_list)
