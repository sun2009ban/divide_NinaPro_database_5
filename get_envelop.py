import numpy as np
import matplotlib.pyplot as plt
import Filter
import utilities
import pdb

def MA(x, ma_len):
    '''
    x:  输入的数据向量
    ma_len: mean average的长度 
    '''

    ma = np.ones(ma_len) / ma_len

    y = np.convolve(x, ma, 'same')

    return y

def Rect(x):
    return np.abs(x)


def envelop(raw, fs=100, ma_len=10):
    x = np.copy(raw)

    #1. high pass filter at 10Hz
    x = Filter.butter_highpass_filter(x, 10, fs)

    #2. Rectification
    x = Rect(x)

    #3. MA
    x = MA(x, ma_len)

    #4. low pass filter at 30Hz
    x = Filter.butter_lowpass_filter(x, 30, fs)

    return x

if __name__ == '__main__':
    path = './raw_data/S1_E1_A1.mat'
    emg, _ = utilities.read_mat(path) # 
    emg = emg[:, 0]
    
    emg_env = envelop(emg, 200, 10)

    plt.figure()
    plt.plot(emg)
    plt.plot(emg_env)
    plt.show()