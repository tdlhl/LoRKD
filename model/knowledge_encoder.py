import torch.nn as nn
import torch
import numpy as np

class Knowledge_Encoder(nn.Module):
    """
    与VLP的区别 : 少了Projection Layer和Atlas Tower
    """
    def __init__(self,
                 text_tower,
                ):
        super().__init__()
        # LP
        self.text_tower = text_tower
        self.projection_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )
        self.modality_embed = nn.Embedding(4, 768)
    
    def forward(self, text, modality):
        text_feature = self.text_tower(text)
        proj_text_feature = self.projection_layer(text_feature)
        
        modality_feature = self.modality_embed(modality)
        
        text_feature = text_feature + modality_feature
        proj_text_feature = proj_text_feature + modality_feature
        
        return text_feature, proj_text_feature