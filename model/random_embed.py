import os
import json

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel

class Random_Embed(nn.Module):
    """
    """
    def __init__(self, label2index_mapping='/mnt/petrelfs/share_data/wuchaoyi/SAM/Knowledge_Data/label2index.json'):
        super().__init__()
        with open(label2index_mapping, 'r') as f:
            self.label2index_mapping = json.load(f) # label --> x
        
        self.modality_embed = nn.Embedding(4, 768)
        
        self.label_embed = nn.Embedding(500, 768)
    
    def forward(self, input_text, modality_code):
        label_code= []
        for t in input_text:
            label_code.append(self.label2index_mapping[t]) # [1, 4, ....]
        label_code = torch.tensor(label_code).cuda()
                
        query_embed = self.label_embed(label_code)  # b*n, d
        modality_feature = self.modality_embed(modality_code)   # b*n, d
        return modality_feature+query_embed
