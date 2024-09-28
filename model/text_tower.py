from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from typing import Tuple, Union, Callable, Optional

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel,BertConfig, AutoTokenizer
# from sentence_transformers import SentenceTransformer

from .tokenizer import MyTokenizer


class Text_Tower(nn.Module):
    """
    wrapper for a single text encoder, to allow text-text contrastive learning
    """
    def __init__(self,
                medcpt_checkpoint: str = None,
                pubmedbert_checkpoint: str = None,
                biolord_checkpoint: str = None,
                embed_dim: int = 768,
                open_bert_layer:Union[Tuple[int, int], int] = 12,
                max_text_length : int = 256
                ):
        super().__init__()
        if pubmedbert_checkpoint:
            self.pubmedbert = self._get_pubmedbert(pubmedbert_checkpoint=pubmedbert_checkpoint, open_bert_layer=open_bert_layer)
            self.tokenizer = MyTokenizer(pubmedbert_checkpoint, max_text_length)
        else:
            self.pubmedbert = None
            
        if medcpt_checkpoint:
            self.medcpt = self._get_medcpt(medcpt_checkpoint=medcpt_checkpoint, open_bert_layer=open_bert_layer)
            self.tokenizer = MyTokenizer(medcpt_checkpoint, max_text_length)
        else:
            self.medcpt = None
            
        if biolord_checkpoint:
            self.biolord = self._get_biolord(biolord_checkpoint=biolord_checkpoint, open_bert_layer=open_bert_layer)
            self.tokenizer = MyTokenizer(biolord_checkpoint, max_text_length)
        else:
            self.biolord = None
        
        self.embed_dim = embed_dim
    
    def _get_pubmedbert(self, pubmedbert_checkpoint, open_bert_layer=None):#12
        config = BertConfig.from_pretrained(pubmedbert_checkpoint, output_hidden_states=True)#bert-base-uncased
        model = AutoModel.from_pretrained(pubmedbert_checkpoint, config=config)#, return_dict=True)
        
        # freeze
        for name, param in model.named_parameters():
            if 'encoder.layer.' in name and int(name.split('encoder.layer.')[-1].split('.')[0])>open_bert_layer:  # encoder.layer.11.xxx --> 11
                param.requires_grad = True
            elif open_bert_layer < 11 and ('pooler' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        return model
    
    def _get_medcpt(self, medcpt_checkpoint, open_bert_layer=None):
        model = AutoModel.from_pretrained(medcpt_checkpoint)
        
        # freeze
        for name, param in model.named_parameters():
            if 'encoder.layer.' in name and int(name.split('encoder.layer.')[-1].split('.')[0])>open_bert_layer:  # encoder.layer.11.xxx --> 11
                param.requires_grad = True
            elif open_bert_layer < 11 and ('pooler' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        return model
    
    def _get_biolord(self, biolord_checkpoint, open_bert_layer=None):
        model = AutoModel.from_pretrained(biolord_checkpoint)
        
        # freeze
        for name, param in model.named_parameters():
            if 'encoder.layer.' in name and int(name.split('encoder.layer.')[-1].split('.')[0])>open_bert_layer:  # encoder.layer.11.xxx --> 11
                param.requires_grad = True
            elif open_bert_layer < 11 and ('pooler' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        return model

    def lock(self):
        for param in self.parameters():
            param.requires_grad = False
            
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text):
        try:
            text = self.tokenizer.tokenize(text) # (n, max_l)
            text['input_ids'] = text['input_ids'].to(device=torch.cuda.current_device())
            text['attention_mask'] = text['attention_mask'].to(device=torch.cuda.current_device())
        except:
            print(text)
            exit()
            
        if self.biolord:
            output = self.biolord(**text)
            pooler_output = self.mean_pooling(output, text['attention_mask'])
        elif self.pubmedbert:
            output = self.pubmedbert(input_ids = text['input_ids'], attention_mask = text['attention_mask'])
            last_hidden_state, pooler_output, hidden_states = output[0],output[1],output[2]
        elif self.medcpt:
            pooler_output = self.medcpt(**text).last_hidden_state[:, 0, :]
        
        return pooler_output

    
