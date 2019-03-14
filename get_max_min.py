# coding: utf-8
from __future__ import print_function, division
import numpy as np
import scipy.io as sio
import pdb
import matplotlib.pyplot as plt
import os

# 剔除过大和过小的点
# 把数据中-+90%最大的当作有用的点
# 把数据归一化到 0～1
def preprocess(x, per=0.95):
	x_flat = x.flatten()
	#plt.hist(x_flat, 100)
	#plt.show()
	x_len = len(x_flat)
	i_max = int(x_len * per) - 1
	i_min = int(x_len * (1 - per))
	x_sort = np.sort(x_flat)
	x_max = x_sort[i_max]
	x_min = x_sort[i_min]
	x[x>x_max] = x_max
	x[x<x_min] = x_min
	x = (x - x_min) / (x_max - x_min)
	return x, x_max, x_min
	
def readmat2np(filename):
	eeg_dict = sio.loadmat(filename)
	eeg = eeg_dict['Data']
	return eeg

if __name__ == '__main__':
	directory = './topology_data'
	eeg = []
	for filename in os.listdir(directory):
		if filename.endswith(".mat"):
			print(filename)
			eeg_ = readmat2np(filename)
			eeg.append(eeg_) 

	eeg = np.concatenate(eeg)
	eeg_standard, eeg_max, eeg_min = preprocess(eeg, 0.95)
	eeg_max_min_dict = {'max':eeg_max, 'min':eeg_min}
	np.save('eeg_max_min.npy', eeg_max_min_dict)
