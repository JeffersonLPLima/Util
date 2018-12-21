import scipy.io
import numpy as np
import glob
import scipy
from scipy import sparse
import bwr
import csv

#1. download data -> $ "wget -r -nH --cut-dirs=2 --no-parent --reject="index.html*" https://physionet.org/pn3/challenge/2017/training/"

FS_hz = 300
duration_seconds = 60
length = duration_seconds*FS_hz
dir_ = 'training2017/'#data directory

files = sorted(glob.glob(dir_+"*.mat"))

train_set = []
train_labels = []
max_length = 0
for f in files:
    record = f[:-4]
    record = record[-6:]
    mat_data = scipy.io.loadmat(f[:-4] + ".mat")
    data = mat_data['val'].squeeze()
    data = np.nan_to_num(data)
    if(len(data)<=length):
        #pad with 0
        data = np.pad(data,(0,length-len(data)), 'constant')
    #signal length == FS_hz * duration_seconds
    train_set.append(data[:length])

csvfile = list(csv.reader(open(dir_+'REFERENCE-v3.csv')))
classes = ['A','N','O','~']
for row in range(len(csvfile)):
    train_labels.append(csvfile[row][1])
            
train_set = np.array(train_set)
train_labels = np.array(train_labels)
print('Train set shape: ',train_set.shape)
print('Train labels shape: ',train_labels.shape)
np.save('train_set.npy', train_set)
np.save('train_labels.npy', train_labels)

#to load 
#X = np.load('train_set.npy')
#y = np.load('train_labels.npy')
