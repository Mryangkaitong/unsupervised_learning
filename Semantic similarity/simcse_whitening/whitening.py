# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/5/26 14:59 下午
@Author : ykt
@File : whitening.py 
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import torch
from torch.utils.data import DataLoader
from SimBERT_data_process import DataPrecessForSentence
# from utils import train, validate
from transformers import BertTokenizer
from SimBERT_model import SimBertModel
from transformers.optimization import AdamW
import pandas as pd
import time
import torch.nn as nn
from transformers import WEIGHTS_NAME, CONFIG_NAME

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import re
import numpy as np


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    #vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def get_entity_data(file_path):
    data=pd.read_excel(file_path)
    data = data.dropna(axis=0)
    entity_list = data["entity"].tolist()
    return entity_list

def main(train_file,test_file, is_whitening=True,
         epochs=8,
         batch_size=16,
         lr=2e-05,
         patience=3,
         max_grad_norm=10.0):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    device = torch.device("cuda")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    output_dir = '%s_%s_%s_%s' % ('models/SimBERT', batch_size, lr, epochs)
    os.path.join('models',output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    entity_list = get_entity_data(train_file)
    train_data = DataPrecessForSentence(bert_tokenizer, entity_list, is_train=False)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
    
    print("\t* Loading testing data...")
    entity_list = get_entity_data(test_file)
    test_data = DataPrecessForSentence(bert_tokenizer, entity_list, is_train=False)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = SimBertModel(model_path_or_name="bert-base-chinese").to(device)
    model.eval()
    
    print("get train feature ...")
    result = np.empty(shape=[0,768])
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments) in tqdm(train_loader):
            seqs, masks, segments = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device)
            _, y_pred = model(seqs, masks, segments)
            result = np.concatenate([result,y_pred.cpu().numpy()])
    print("train shape:")
    print(result.shape)
    
    if is_whitening:
        #whitening
        kernel, bias = compute_kernel_bias(result)
        #
        result = transform_and_normalize(result, kernel, bias)
        print("train shape after whitening:")
        print(result.shape)
        np.save("whitening_train.npy", result)

        print("get test feature ...")
        result = np.empty(shape=[0,768])
        with torch.no_grad():
            for (batch_seqs, batch_seq_masks, batch_seq_segments) in tqdm(test_loader):
                seqs, masks, segments = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device)
                _, y_pred = model(seqs, masks, segments)
                result = np.concatenate([result,y_pred.cpu().numpy()])
        print("test shape:")
        print(result.shape)
        result = transform_and_normalize(result, kernel, bias)
        print("test shape after whitening:")
        print(result.shape)
        np.save("whitening_test.npy", result)
    
    else:
        np.save("original_train.npy", result)
        print("get test feature ...")
        result = np.empty(shape=[0,768])
        with torch.no_grad():
            for (batch_seqs, batch_seq_masks, batch_seq_segments) in tqdm(test_loader):
                seqs, masks, segments = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device)
                _, y_pred = model(seqs, masks, segments)
                result = np.concatenate([result,y_pred.cpu().numpy()])
        print("test shape:")
        print(result.shape)
        np.save("original_test.npy", result)

if __name__ == "__main__":
    main("train_simsce.xlsx", "test_simsce.xlsx", is_whitening=False)
