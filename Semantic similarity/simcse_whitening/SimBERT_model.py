# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/2/9 5:27 下午 
@Author : azun
@File : BERT_model.py 
'''

# import torch
# from torch import nn
# from transformers import  BertConfig,pipeline, AutoModel
# from transformers import modeling_outputs
# from transformers.models.bert.modeling_bert import BertForSequenceClassification
# BertForSequenceClassification.save_pretrained()
from transformers.models.bert.modeling_bert import BertPreTrainedModel,BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.file_utils import  (
    # ModelOutput,
    add_code_sample_docstrings,
    # add_start_docstrings,
    add_start_docstrings_to_model_forward,
    # replace_return_docstrings,
is_torch_tpu_available,
WEIGHTS_NAME
)
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import os
from transformers import BertConfig
# import math
# import os
# import warnings
# from dataclasses import dataclass
# from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

class BertForSequenceClassification_SimBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        y_pred = self.dropout(pooled_output)
        
        #simcse loss
        idxs = torch.arange(0, y_pred.size()[0]).to(y_pred)
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        y_true = idxs_1.eq(idxs_2).to(y_pred)
        # 计算相似度
        y_pred = y_pred / (torch.sqrt(torch.sum(torch.pow(y_pred,2),dim=1))+1e-8).reshape(-1,1).repeat(1,y_pred.size()[1])
        similarities = torch.mm(y_pred, torch.t(y_pred))
        similarities = similarities - torch.eye(y_pred.size()[0]).to(similarities)* 1e12
        similarities = similarities * 20
        
        
        def multilabel_categorical_crossentropy(y_true_, y_pred_):
            """多标签分类的交叉熵
            说明：y_true和y_pred的shape一致，y_true的元素非0即1，
                 1表示对应的类为目标类，0表示对应的类为非目标类。
            """
            y_pred_ = (1 - 2 * y_true_) * y_pred_
            y_pred_neg = y_pred_ - y_true_ * 1e12
            y_pred_pos = y_pred_ - (1 - y_true_) * 1e12
            zeros = torch.zeros_like(y_pred_[..., :1])
            y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
            y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
            neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
            pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
            return neg_loss + pos_loss
        loss = multilabel_categorical_crossentropy(y_true, similarities)
        return torch.mean(loss), y_pred


class SimBertModel(nn.Module):
    def __init__(self,model_path_or_name):
        super(SimBertModel, self).__init__()
        self. bert = BertForSequenceClassification_SimBERT.from_pretrained(model_path_or_name,num_labels=2)  # /bert_pretrain/

        self. config = BertConfig.from_pretrained(model_path_or_name)
        # self. bert = BertForSequenceClassification.from_pretrained("bert-base-chinese",num_labels=2)  # /bert_pretrain/
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            assert param.requires_grad, 'param.requires_grad should be set to Ture while training !!!'
        # for param in self.bert.parameters():
        #     param.requires_grad = True

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments):
        loss, y_pred = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                                 token_type_ids=batch_seq_segments)
        return loss, y_pred


class BertModelTest(nn.Module):
    def __init__(self, model_path):
        super(BertModelTest, self).__init__()
        # config = BertConfig.from_pretrained(model_path)
        config = BertConfig.from_pretrained('/data3/azun/project_chinatel_recommendation/models')
        self.bert = BertForSequenceClassification_SimBERT(config)  # /bert_pretrain/
        # self.bert=BertForSequenceClassification_SimBERT.from_pretrained("bert-base-chinese",num_labels=2)
        self.device = torch.device("cuda")

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                                 token_type_ids=batch_seq_segments, labels=labels)
        probabilities = F.softmax(logits, dim=-1)
        return loss, logits, probabilities

# bert=SimBertModel()
# bert=BertModelTest()
