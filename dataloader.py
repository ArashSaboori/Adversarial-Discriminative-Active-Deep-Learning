import numpy as np
import os
import random
from sklearn.decomposition import PCA
from scipy.io import loadmat
from config import Config

opt = Config().parse()
random.seed(1)

"""""""""""""""""""""
Load three dataset.  
"""""""""""""""""""""
def load(dataset):
    if dataset == 'Indian':
            data = loadmat(os.path.join(opt.DATA_DIR, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        else:
            data = loadmat(os.path.join(opt.DATA_DIR, 'Indian_' + str(opt.CHANNEL) + '.mat'))['dataDR']
        label = loadmat(os.path.join(opt.DATA_DIR, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif dataset == 'Botswana_May':
            data = loadmat(os.path.join(opt.DATA_DIR, 'Botswana_Mays_corrected.mat'))['Botswana_May_corrected']
        else:
            data = loadmat(os.path.join(opt.DATA_DIR, 'Botswana_May_' + str(opt.CHANNEL) + '.mat'))['dataDR']
        label = loadmat(os.path.join(opt.DATA_DIR, 'Botswana_May_gt.mat'))['Botswana_May_gt']
    elif dataset == 'Botswana_July':
            data = loadmat(os.path.join(opt.DATA_DIR, 'Botswana_July.mat'))['Botswana_July']
        else:
            data = loadmat(os.path.join(opt.DATA_DIR, 'Botswana_July_' + str(opt.CHANNEL) + '.mat'))['dataDR']
        label = loadmat(os.path.join(opt.DATA_DIR, 'Botswana_July_gt.mat'))['Botswana_July_gt']
    
    else:
        raise NotImplementedError

    return data, label


""""""""""
Pad the edge of a hyperspectral image with 0.
""""""""""
def pad_zeros(input, margin=4):
    output = np.zeros((input.shape[0] + 2 * margin, input.shape[1] + 2 * margin, input.shape[2]))
    row_offset = margin
    col_offset = margin
    output[row_offset:input.shape[0] + row_offset, col_offset:input.shape[1] + col_offset, :] = input
    return output


"""""""""
Get the indexes of initial labeled/unlabeled/test data in the form of (x, y).
"""""""""
def get_init_indices(data, label):
    init_tr_labeled_idx = []
    init_tr_unlabeled_idx = []
    te_idx = []
    tmp = []
    for cls in range(1, opt.N_CLS + 1):
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if label[row, col] == cls:
                    tmp.append([row, col])
        idx = random.sample(range(len(tmp)), opt.INIT_N_L)
        for i in idx:
            init_tr_labeled_idx.append(tmp[i])
        tmp = []

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if (list([row, col]) not in init_tr_labeled_idx) and (label[row, col] != 0):
                tmp.append(list([row, col]))
    idx = random.sample(range(len(tmp)), opt.INIT_N_UNL)
    for i in idx:
        init_tr_unlabeled_idx.append(tmp[i])

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if (list([row, col]) not in init_tr_labeled_idx) and (list([row, col]) not in init_tr_unlabeled_idx):
                te_idx.append(list([row, col]))
    
    return init_tr_labeled_idx, init_tr_unlabeled_idx, te_idx


""""""""""
Get data with indexes.
""""""""""
def get_data(data, label, index):
    margin = int((opt.WINDOW_SIZE - 1) / 2)
    data_padded = pad_zeros(data, margin=margin)
    _patch = np.zeros((data.shape[0]*data.shape[1], opt.WINDOW_SIZE, opt.WINDOW_SIZE, data.shape[-1]))
    _label = np.zeros((data.shape[0]*data.shape[1]))
    
    patch_index = 0

    for row in range(margin, data_padded.shape[0] - margin):
        for col in range(margin, data_padded.shape[1] - margin):
            if list([row - margin, col - margin]) in index:
                patch = data_padded[row - margin:row + margin + 1, col - margin:col + margin + 1]
                _patch[patch_index, :, :, :] = patch
                _label[patch_index] = label[row - margin, col - margin]
                patch_index += 1

    '''remove zero labels'''
    _patch = _patch[_label > 0, :, :, :]
    _label = _label[_label > 0]
    _label -= 1
    
    return _patch, _label
