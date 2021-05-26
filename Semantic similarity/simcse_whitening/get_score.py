# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/5/7 14:31 下午 
@Author : ykt
@File : SimBERT_predict.py 
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    train_file = r"train_simsce.xlsx"
    test_file = r"test_simsce.xlsx"
    
    test_num = 100
    topK = 10
    
    train_df = pd.read_excel(train_file)
    test_df = pd.read_excel(test_file)
    test_entity = test_df["entity"].tolist()
    train_vec = np.load("original_train.npy")
    test_vec = np.load("original_test.npy")
    count = 0
    result = []
    test_vec_iterator = tqdm(test_vec[:test_num])
    for cur_test in test_vec_iterator:
        temp = train_df
        sim_list = []
        for cur_train in train_vec:
            sims = (cur_test[np.newaxis,:] * cur_train[np.newaxis,:]).sum(axis=1)
            sim_list.append(sims)
        temp["score"] = sim_list
        temp = temp.sort_values(by="score" , ascending=False)
        candidate_entity = temp["entity"].tolist()[:topK]
        candidate_score = temp["score"].tolist()[:topK]
        result.append([test_entity[count], "@".join(candidate_entity), "@".join(list(map(lambda x:str(x), candidate_score)))])
        count = count + 1
    result = pd.DataFrame(result, columns=['test', 'candidate', 'score'])
    result.to_excel("result.xlsx")
    print("successfully!!!!!!")
