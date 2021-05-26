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
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from SimBERT_model import SimBertModel
from SimBERT_data_process import DataPrecessForSentence


class SimBERT_predict():
    def __init__(self,model_name_or_path,batch_size=16):
        self.    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
        # self.pretrained_pytorch_model = os.path.join(model_name_or_path, 'pytorch_model.bin')
        self.model = SimBertModel(model_name_or_path).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(model_name_or_path, 'pytorch_model.bin')))
        self.model.eval()
        self.batch_size=batch_size

    def get_vec(self, file, batch_size=64, save_file="feature_train.npy"):
        
        data = pd.read_excel(file)
        print(data.shape)
        print(data[data.isna().any(axis=1)])
        data = data.dropna(axis=0)
        print(data.shape)
        
        #entity_list = data["entity"].tolist()
        entity_list = data["entity"].tolist()
        print(entity_list[:2])
        entity_data = DataPrecessForSentence(self.bert_tokenizer, entity_list, is_train=False)
        entity_loader = DataLoader(entity_data, shuffle=False, batch_size=batch_size)
        result = np.empty(shape=[0,768])

        with torch.no_grad():
            for (batch_seqs, batch_seq_masks, batch_seq_segments) in tqdm(entity_loader):
                seqs, masks, segments = batch_seqs.to(self.device), batch_seq_masks.to(
                    self.device), batch_seq_segments.to(self.device)
                _, y_pred = self.model(seqs, masks, segments)
                #print(y_pred.cpu().numpy().shape)
                result = np.concatenate([result,y_pred.cpu().numpy()])
        print(result.shape)
        np.save(save_file, result)


    def get_sim(self, all_vecs):
        sim_list = []
        for vecs in all_vecs:
            a_vecs, b_vecs = vecs
            sims = (a_vecs * b_vecs).sum(axis=1)
            sim_list.append(sims)
        return sim_list

if __name__ == '__main__':
    train_file = r"train_simsce.xlsx"
    test_file = r"test_simsce.xlsx"
    #提取向量
    model=SimBERT_predict('models/SimBERT_16_2e-05_8')
    model.get_vec(train_file)
    model.get_vec(test_file, save_file="feature_test.npy")
    print("successfully!!!!!!")
