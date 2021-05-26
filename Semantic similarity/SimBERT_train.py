# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/2/14 8:07 下午
@Author : azun
@File : SimBERT_train.py 
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
def correct_predictions(output_probabilities, targets):
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()
    
def get_entity_data(file_path):
    data=pd.read_excel(file_path)
    data = data.dropna(axis=0)
    entity_list = data["entity"].tolist()
    return entity_list

def train(model, dataloader, optimizer, epoch_number, max_gradient_norm):
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        seqs, masks, segments = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(
            device)
        optimizer.zero_grad()
        loss, _ = model(seqs, masks, segments)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    return epoch_time, epoch_loss
def main(train_file,
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
    train_data = DataPrecessForSentence(bert_tokenizer, entity_list)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = SimBertModel(model_path_or_name="bert-base-chinese").to(device)
    # -------------------- Preparation for training  ------------------- #
    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []

    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training Bert model on device: {}".format(device), 20 * "=")
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}%"
              .format(epoch_time, epoch_loss))

    # 它包装在PyTorch DistributedDataParallel或DataParallel中
    model_to_save = model.module if hasattr(model, 'module') else model
    # # 如果使用预定义的名称保存，则可以使用`from_pretrained`加载
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    bert_tokenizer.save_vocabulary(output_dir)
    
    


if __name__ == "__main__":
    main("train_simsce.xlsx")
