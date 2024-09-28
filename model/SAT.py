import math
import os
from typing import Tuple, Union, List

import torch.nn as nn
import torch 
from einops import rearrange, repeat, reduce
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D

from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from .transformer_decoder import TransformerDecoder,TransformerDecoderLayer


class SAT(nn.Module):
    def __init__(self, vision_backbone='UNET', input_size=[288, 288, 96], deep_supervision=False):
        """
        SAT

        Args:
            vision_backbone (str, optional): vision encoder and decoder. Defaults to UNET.
            input_size (list, optional): cropped patch size. Defaults to [288, 288, 96].
        """
        super().__init__()
        height, width, depth = input_size
        
        ds_height = ds_width = ds_depth = {
            'UNET' : 32 # 6 layer U-Net
        }[vision_backbone]  
        
        self.backbone = {
            'UNET' : PlainConvUNet(input_channels=3, 
                                   n_stages=6, 
                                   features_per_stage=(64, 64, 128, 256, 512, 768), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                                #    num_classes=4,
                                   n_conv_per_stage_decoder=(2, 2, 2, 2, 2), 
                                   conv_bias=True, 
                                   norm_op=nn.InstanceNorm3d,
                                   norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                   dropout_op=None,
                                   dropout_op_kwargs=None,
                                   nonlin=nn.LeakyReLU, 
                                   nonlin_kwargs=None,
                                   deep_supervision=deep_supervision,
                                   nonlin_first=False
                                   ),
        }[vision_backbone]
        
        if 'UNET' in vision_backbone:
            self.backbone.apply(InitWeights_He(1e-2))
            
        vis_dim = { # dim of latent embedding
            'UNET' : 768,
        }[vision_backbone]
        
        # 256, 256/32=16, 256/32=16, 96//32 = 3
        pos_embedding = PositionalEncoding3D(vis_dim)(torch.zeros(1, (height//ds_height), (width//ds_width), (depth//ds_depth), vis_dim)) # b h/p w/p d/p dim
        self.pos_embedding = rearrange(pos_embedding, 'b h w d c -> (h w d) b c')   # n b dim

        self.avg_pool_ls = [    
            nn.AvgPool3d(32, 32),
            nn.AvgPool3d(16, 16),
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(4, 4),
            nn.AvgPool3d(2, 2),
            ]
            
        self.query_proj_mlp = {
            'UNET' : nn.Identity(),
        }[vision_backbone]
        
        decoder_layer = TransformerDecoderLayer(d_model = vis_dim, nhead = 8, normalize_before=True)
        decoder_norm = nn.LayerNorm(vis_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer = decoder_layer, num_layers = 6, norm=decoder_norm)
        
        self.transformer_decoder_mlp = {
            'UNET' : nn.Sequential(
                        nn.Linear(vis_dim, 256),
                        nn.GELU(),
                        nn.Linear(256, 64),
                        nn.GELU(),
                    ),
        }[vision_backbone]

    def forward(self, queries, image_input):
        
        if isinstance(queries, List):   # inference  
            logits = self.infer(queries, image_input)
            return logits

        B,C,H,W,D = image_input.shape
        _,N,_ = queries.shape    # N is the num of query
        
        # Image Encoder and Pixel Decoder
        latent_embedding, per_pixel_embeddings = self.backbone(image_input)
        per_pixel_embedding = per_pixel_embeddings[0] # output of the last decoder layer
        pos = self.pos_embedding.to(latent_embedding[-1].device)    # (H/P W/P D/P) B Dim
        # By default, attention in torch is not batch_first
        # print('teacher latent emb')
        # for emb1 in latent_embedding:
        #     print(emb1.shape)
        # image_embedding = []
        # for latent_embedding, avg_pool in zip(latent_embedding_ls, self.avg_pool_ls):
        #     tmp = avg_pool(latent_embedding)
        #     image_embedding.append(tmp)   # B ? H/P W/P D/P
        # image_embedding.append(latent_embedding[-1])

        image_embedding = rearrange(latent_embedding[-1], 'b dim h w d -> (h w d) b dim') # (H/P W/P D/P) B Dim
        queries = rearrange(queries, 'b n dim -> n b dim') # N B Dim
        queries = self.query_proj_mlp(queries)
        
        # Transformer Decoder (query image feature with text input
        # print('教师 image_embedding.shape=', image_embedding.shape)
        mask_embedding,_ = self.transformer_decoder(queries, image_embedding, pos = pos) # N B Dim
        mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
        mask_embedding = self.transformer_decoder_mlp(mask_embedding)
        mask_embedding = rearrange(mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        # Dot product
        logits = [torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mask_embedding)] # NOTE: not sigmoid yet
        
        return logits, latent_embedding, image_embedding, per_pixel_embeddings

    def vision_backbone_forward(self, image_input):

        B,C,H,W,D = image_input.shape
        
        # Image Encoder and Pixel Decoder
        latent_embedding, per_pixel_embeddings = self.backbone(image_input)
        per_pixel_embedding = per_pixel_embeddings[0] # output of the last decoder layer
        pos = self.pos_embedding.to(latent_embedding[-1].device)    # (H/P W/P D/P) B Dim
        image_embedding = rearrange(latent_embedding[-1], 'b dim h w d -> (h w d) b dim') # (H/P W/P D/P) B Dim
            
        return latent_embedding, image_embedding, pos, per_pixel_embeddings

    def infer(self, queries, image_input):
        """
        infer batches of queries on a batch of patches
        """
        B,C,H,W,D = image_input.shape
        
        latent_embedding_ls, image_embedding, pos, per_pixel_embedding_ls = self.vision_backbone_forward(image_input)
        
        logits_ls = []
        for q in queries:      
            # Transformer Decoder (query image feature with text input  
            N,_ = q.shape    # N is the num of query
            q = repeat(q, 'n dim -> n b dim', b=B) # N B Dim NOTE:By default, attention in torch is not batch_first
            mask_embedding,_ = self.transformer_decoder(q, image_embedding, pos = pos) # N B Dim
            mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
            mask_embedding = self.transformer_decoder_mlp(mask_embedding)
            mask_embedding = rearrange(mask_embedding, '(b n) dim -> b n dim', b=B, n=N)

            per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
            logits_ls.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mask_embedding)) # bnhwd
        logits = torch.concat(logits_ls, dim=1)  # bNhwd
        
        return logits

if __name__ == '__main__':
    model = Maskformer().cuda()
    image = torch.rand((2, 3, 256, 256, 32)).cuda()
    query = torch.rand((2, 10, 768)).cuda()
    segmentations = model(query, image)
    print(segmentations.shape)