import os

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel

class MedCPT(nn.Module):
    """
    wrapper for a single text encoder, to allow text-text contrastive learning
    """
    def __init__(self, cpt_checkpoint=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(cpt_checkpoint)
        self.modality_embed = nn.Embedding(4, 768)
    
    def forward(self, text, modality):
        text_feature = self.model(**text).last_hidden_state[:, 0, :]
        modality_feature = self.modality_embed(modality)
        text_feature += modality_feature
        return text_feature

class MedCPT_Embed(nn.Module):
    """
    wrapper for a single text encoder, to allow text-text contrastive learning
    """
    def __init__(self, cpt_embeddings_dir):
        super().__init__()
        self.embeddings_dict = {}
        for f in os.listdir(cpt_embeddings_dir): # mod_lab.npy / none.npy / lab.npy
            name = f[:-4]
            self.embeddings_dict[name] = np.load(os.path.join(cpt_embeddings_dir, f))    # ct_face.npy / face.npy --> (1, 768)
            
        if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
            print(f"** Query ** Load query embeddings from {cpt_embeddings_dir}.")
        
        self.modality_embed = nn.Embedding(4, 768)
    
    def forward(self, input_text, modality_code):
        query_embed = []
        for t in input_text:
            query_embed.append(self.embeddings_dict[t.lower()])
        queries = torch.tensor(np.concatenate(query_embed, axis=0)) # b*n, d
        modality_feature = self.modality_embed(modality_code)   # b*n, d
        queries = queries.to(modality_feature.device)
        return modality_feature+queries
