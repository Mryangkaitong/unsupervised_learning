# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/5/26 14:57 下午 
@Author : ykt
@File : SimBERT_data_process.py 
'''
from torch.utils.data import Dataset
# from hanziconv import HanziConv
import pandas as pd
import torch


class DataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, entity_list, max_char_len=256, is_train=True):
        """
        bert_tokenizer :分词器
        LCQMC_file     :语料文件
        """
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_char_len
        self.seqs, self.seq_masks, self.seq_segments = self.get_input(entity_list,is_train=is_train)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx]

    # 获取文本与标签
    def get_input(self, entity_list, is_train=True):
        """
        同一个输入放两次达到dropout不同
        """

        # 切词
        tokens_entity = []
        for tokens in list(map(self.bert_tokenizer.tokenize, entity_list)):
            tokens_entity.append(tokens)
            if is_train:
                tokens_entity.append(tokens)

        # 获取定长序列及其mask
        result = list(map(self.trunate_and_pad, tokens_entity))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), \
               torch.Tensor(seq_segments).type(torch.long)

    def trunate_and_pad(self, tokens_entity):
        """
        1. 如果是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
        入参:
            seq_1       : 输入序列，在本处其为单个句子。
            seq_2       : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，单句，取值都为0 ，双句按照01切分

        """
        # 对超长序列进行截断
        if len(tokens_entity) > self.max_seq_len - 2:
            tokens_entity = tokens_entity[0:(self.max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + tokens_entity + ['[SEP]']
        seq_segment = [0] * (len(tokens_entity) + 2)
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = seq_segment + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment
