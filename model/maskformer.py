import random
from typing import Tuple, Union, List
import math
import os
import time

import torch.nn as nn
import torch.nn.functional as F
import torch 
from einops import rearrange, repeat, reduce
from positional_encodings.torch_encodings import PositionalEncoding3D
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet, PlainConvUNet_lora, PlainConvUNet_lora_encoder, PlainConvUNet_lora_decoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from .residual_unet import ResidualUNet
from .SwinUNETR import SwinUNETR
from .transformer_decoder import TransformerDecoder,TransformerDecoderLayer
from .position_encoding import PositionEmbeddingLearned3d
# from .umamba_enc import UMambaEnc
# from .umamba_bot import UMambaBot
# from .umamba_mid import UMambaMid


class Maskformer(nn.Module):
    def __init__(self, vision_backbone='UNET', image_size=[288, 288, 96], patch_size=[32, 32, 32], learnable_pe=False, deep_supervision=False):
        """
        Maskformer with UNET as backbone

        Args:
            image_size (list, optional): image size. Defaults to [256, 256, 128].
            patch_size (list, optional): maxium patch size, i.e. the downsample ratio after swin-Transformer encoder. Defaults to [32, 32, 32].
        """
        super().__init__()
        image_height, image_width, frames = image_size
        self.hw_patch_size = patch_size[0] 
        self.frame_patch_size = patch_size[-1]
        self.vision_backbone = vision_backbone
        
        self.deep_supervision = deep_supervision
        
        # backbone can be any multi-scale enc-dec vision backbone
        # the enc outputs multi-scale latent features
        # the dec outputs multi-scale per-pixel features
        self.backbone = {
            'SwinUNETR' : SwinUNETR(
                            img_size=[288, 288, 96],    # 48, 48, 96, 192, 384, 768
                            in_channels=3,
                            feature_size=48,  
                            drop_rate=0.0,
                            attn_drop_rate=0.0,
                            dropout_path_rate=0.0,
                            use_checkpoint=False,
                            ),
            'UNET' : PlainConvUNet(input_channels=3, 
                                   n_stages=6, 
                                   features_per_stage=(64, 64, 128, 256, 512, 768), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
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
            'UNET_nano_lora' : PlainConvUNet_lora(input_channels=3, 
                                   n_stages=6, 
                                   features_per_stage=(64, 64, 128, 256, 512, 768), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
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
            'UNET_lora' : PlainConvUNet_lora(input_channels=3, 
                                   n_stages=5, 
                                   features_per_stage=(32, 64, 128, 256, 512), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2), 
                                   n_conv_per_stage_decoder=(2, 2, 2, 2), 
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
            'UNET_lora_std' : PlainConvUNet(input_channels=3, 
                                   n_stages=5, 
                                   features_per_stage=(32, 64, 128, 256, 512), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2), 
                                   n_conv_per_stage_decoder=(2, 2, 2, 2), 
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
            'UNET-L' : PlainConvUNet(input_channels=3, 
                                   n_stages=6, 
                                   features_per_stage=(128, 128, 256, 512, 1024, 1536), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
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
            'UNET-H' : PlainConvUNet(input_channels=3, 
                                   n_stages=6, 
                                   features_per_stage=(128, 128, 256, 512, 1024, 1536), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(3, 3, 3, 3, 3, 3), 
                                   n_conv_per_stage_decoder=(3, 3, 3, 3, 3), 
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
        
        self.backbone.apply(InitWeights_He(1e-2))
        
        # fixed to text encoder out dim
        query_dim = 768

        # all backbones are 6-depth, thus the first 5 scale latent feature outputs need to be down-sampled
        self.avg_pool_ls = [    
            nn.AvgPool3d(32, 32),
            nn.AvgPool3d(16, 16),
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(4, 4),
            nn.AvgPool3d(2, 2),
            ]
        # multi-scale latent feature are projected to query_dim before query decoder
        self.projection_layer = {
            'SwinUNETR' : nn.Sequential(
                        nn.Linear(1536, 768),
                        nn.GELU(),
                        nn.Linear(768, query_dim),
                        nn.GELU()
                    ),
            'UNET' : nn.Sequential(
                        nn.Linear(1792, 768),
                        nn.GELU(),
                        nn.Linear(768, query_dim),
                        nn.GELU()
                    ),
            'UNET_nano_lora' : nn.Sequential(
                        nn.Linear(1792, 768),
                        nn.GELU(),
                        nn.Linear(768, query_dim),
                        nn.GELU()
                    ),
            'UNET_lora' : nn.Sequential(
                        nn.Linear(992, 768),
                        nn.GELU(),
                        nn.Linear(768, query_dim),
                        nn.GELU()
                    ),
            'UNET_lora_std' : nn.Sequential(
                        nn.Linear(992, 768),
                        nn.GELU(),
                        nn.Linear(768, query_dim),
                        nn.GELU()
                    ),
            'UNET-L' : nn.Sequential(
                        nn.Linear(3584, 1536),
                        nn.GELU(),
                        nn.Linear(1536, query_dim),
                        nn.GELU()
                    ),
            'UNET-H' : nn.Sequential(
                        nn.Linear(3584, 1536),  # 128, 128, 256, 512, 1024, 1536 --> 3584 --> 768
                        nn.GELU(),
                        nn.Linear(1536, query_dim),
                        nn.GELU()
                    ),
        }[vision_backbone]
        
        # positional encoding
        self.learnable_pe = learnable_pe
        if learnable_pe:
            self.pos_embedding = PositionEmbeddingLearned3d(query_dim//3, (image_height//self.hw_patch_size), (image_width//self.hw_patch_size), (frames//self.frame_patch_size))
        else:
            pos_embedding = PositionalEncoding3D(query_dim)(torch.zeros(1, (image_height//self.hw_patch_size), (image_width//self.hw_patch_size), (frames//self.frame_patch_size), query_dim)) # b h/p w/p d/p dim
            self.pos_embedding = rearrange(pos_embedding, 'b h w d c -> (h w d) b c')   # n b dim
        
        # (fused latent embeddings + pe) x query prompts
        decoder_layer = TransformerDecoderLayer(d_model=query_dim, nhead=8, normalize_before=True)
        decoder_norm = nn.LayerNorm(query_dim)
        if vision_backbone=='UNET-E1' or vision_backbone=='UNET-D':
            self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=4, norm=decoder_norm)
        else:
            self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=6, norm=decoder_norm)
        
        # mask embedding are projected to perpixel_dim
        # mid stage output (only consider the last 3 mid layers of decoder, i.e. feature maps with resolution /2 /4 /8)
        if self.deep_supervision:
            feature_per_stage = {
                'SwinUNETR':[48, 96, 192],
                'UNET':[64, 128, 256],
                'UNET_nano_lora':[64, 128, 256],
                'UNET_lora':[64, 128, 256],
                'UNET_lora_std':[64, 128, 256],
                'UNET-L':[128, 256, 512],
                'UNET-H':[128, 256, 512],
                }[vision_backbone]
            mid_dim = {
                'SwinUNETR':[256, 384, 512],
                'UNET':[256, 384, 512],
                'UNET_nano_lora':[256, 384, 512],
                'UNET_lora':[256, 384, 512],
                'UNET_lora_std':[256, 384, 512],
                'UNET-L':[384, 512, 512],
                'UNET-H':[384, 512, 512],
                }[vision_backbone]
            self.mid_mask_embed_proj = []
            for hidden_dim, per_pixel_dim in zip(mid_dim, feature_per_stage):
                self.mid_mask_embed_proj.append(
                    nn.Sequential(
                        nn.Linear(query_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, per_pixel_dim),
                        nn.GELU(),
                        ),
                    )
                self.mid_mask_embed_proj = nn.ModuleList(self.mid_mask_embed_proj)
                
        # largest output        
        mid_dim, per_pixel_dim = {
            'SwinUNETR' : [256, 48],
            'UNET' : [256, 64],
            'UNET_nano_lora' : [256, 64],
            'UNET_lora' : [256, 32],
            'UNET_lora_std' : [256, 32],          
            'UNET-L' : [384, 128],
            'UNET-H' : [384, 128],
        }[vision_backbone]
        self.mask_embed_proj = nn.Sequential(
            nn.Linear(query_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, per_pixel_dim),
            nn.GELU(),
            )
        
        if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
            print(f'** MODEL ** Learnable PE : {learnable_pe}\n** MODEL ** Vision Backbone : {vision_backbone}')    
            
    def vision_backbone_forward(self, image_input, task_labels=None):
        B,C,H,W,D = image_input.shape
        
        # Image Encoder and Pixel Decoder
        if task_labels==None:
            # print('task_labels==None')
            latent_embedding_ls, per_pixel_embedding_ls = self.backbone(image_input)
        else:
            # print('maskformer里task_labels=', task_labels)
            latent_embedding_ls, per_pixel_embedding_ls = self.backbone(image_input, task_labels) # B Dim H/P W/P D/P
        # print('student latent emb')
        # for emb1 in latent_embedding_ls:
        #     print(emb1.shape)
        # avg pooling each multiscale feature to H/P W/P D/P
        image_embedding = []
        # print('len(latent_embedding_ls)', len(latent_embedding_ls), 'len(self.avg_pool_ls)', len(self.avg_pool_ls))
        for latent_embedding, avg_pool in zip(latent_embedding_ls, self.avg_pool_ls):
            # print(latent_embedding.shape)
            tmp = avg_pool(latent_embedding)
            # print(tmp.shape)
            image_embedding.append(tmp)   # B ? H/P W/P D/P
        # print(latent_embedding_ls[-1].shape)
        if self.vision_backbone not in ['UNET_lora_std', 'UNET_lora', 'UNET_lora_encoder', 'UNET_lora_decoder']:
            image_embedding.append(latent_embedding_ls[-1])
        # print(latent_embedding_ls[-1].shape)
    
        # aggregate multiscale features into image embedding (and proj to align with query dim)
        image_embedding = torch.cat(image_embedding, dim=1)
        image_embedding = rearrange(image_embedding, 'b d h w depth -> b h w depth d')
        image_embedding = self.projection_layer(image_embedding)   # B H/P W/P D/P Dim
        image_embedding = rearrange(image_embedding, 'b h w d dim -> (h w d) b dim') # (H/P W/P D/P) B Dim
            
        # add pe to image embedding
        if self.learnable_pe:
            pos = self.pos_embedding(B, H//self.hw_patch_size, W//self.hw_patch_size, D//self.frame_patch_size, latent_embedding_ls[-1]) # B (H/P W/P D/P) D
            pos = rearrange(pos, 'b n dim -> n b dim') # (H/P W/P D/P) B Dim
        else:
            pos = self.pos_embedding.to(latent_embedding_ls[-1].device)   # (H/P W/P D/P) B Dim
            
        return latent_embedding_ls, image_embedding, pos, per_pixel_embedding_ls
    
    def query_decoder_forward(self, queries, image_embedding, pos, per_pixel_embedding_ls):
        """
        infer a batch of queries on on a batch of patches
        """
        _, B, _ = image_embedding.shape
        
        logits_ls = []
        for q in queries:        
            N,_ = q.shape    # N is the num of query
            # By default, attention in torch is not batch_first
            q = repeat(q, 'n dim -> n b dim', b=B) # N B Dim
            # Transformer Decoder (query image feature with text input
            mask_embedding,_ = self.transformer_decoder(q, image_embedding, pos = pos) # N B Dim
            mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
            # Dot product
            mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
            mask_embedding = rearrange(mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
            per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
            logits_ls.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mask_embedding)) # bnhwd
        logits = torch.concat(logits_ls, dim=1)  # bNhwd
        
        return logits
    
    def infer(self, queries, image_input, task_labels=None):
        """
        infer batches of queries on a batch of patches
        """
        B,C,H,W,D = image_input.shape
        
        latent_embedding_ls, image_embedding, pos, per_pixel_embedding_ls = self.vision_backbone_forward(image_input, task_labels)
        
        logits_ls = []
        for q in queries:      
            # Transformer Decoder (query image feature with text input  
            N,_ = q.shape    # N is the num of query
            q = repeat(q, 'n dim -> n b dim', b=B) # N B Dim NOTE:By default, attention in torch is not batch_first
            mask_embedding,_ = self.transformer_decoder(q, image_embedding, pos = pos) # N B Dim
            mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
            # Dot product
            mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
            mask_embedding = rearrange(mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
            per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
            logits_ls.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mask_embedding)) # bnhwd
        logits = torch.concat(logits_ls, dim=1)  # bNhwd
        
        return logits
    
    def forward(self, queries=None, image_input=None, vision_feature_ls=None, task_labels=None):
        # NOTE: queries should be the text encoder output + modality embedding
        
        # Infer / Evaluate Forward ------------------------------------------------------------
        
        # given a list of batched patches (all the patches from a volume), i.e N x bhwd, derive the vision features (of the volume)
        if queries is None:
            vision_feature_ls = []
            for batch in image_input:
                latent_embedding_ls, image_embedding, pos, per_pixel_embedding_ls = self.vision_backbone_forward(batch)
                vision_feature_ls.append([image_embedding, pos, per_pixel_embedding_ls])
            return vision_feature_ls
        
        # given a list of vision features (all the patches of the volume), and a batched queries, derive the segmentations on all these patches
        if vision_feature_ls is not None:
            logits_ls = []
            for image_embedding, pos, per_pixel_embedding_ls in vision_feature_ls:
                logits_ls.append(self.query_decoder_forward([queries], image_embedding, pos, per_pixel_embedding_ls))
            return logits_ls
        
        # given a list of batched queries (usually all the labels to seg) and a batch of patches, derive all the segmentations on these patches
        if isinstance(queries, List):   # inference  
            logits = self.infer(queries, image_input, task_labels)
            return logits
        
        # Train Forward -----------------------------------------------------------------------
        # get vision features
        B,C,H,W,D = image_input.shape
        latent_embedding_ls, image_embedding, pos, per_pixel_embedding_ls = self.vision_backbone_forward(image_input, task_labels)
        # Query Decoder (query image feature with text input
        _,N,_ = queries.shape    # N is the num of query
        queries = rearrange(queries, 'b n dim -> n b dim') # N B Dim NOTE:By default, attention in torch is not batch_first

        mask_embedding,_ = self.transformer_decoder(queries, image_embedding, pos = pos) # N B Dim
        mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
        # Dot product
        last_mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
        last_mask_embedding = rearrange(last_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
        logits = [torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, last_mask_embedding)]
        # Deep supervision
        if self.deep_supervision:
            for mask_embed_proj, per_pixel_embedding in zip(self.mid_mask_embed_proj, per_pixel_embedding_ls[1:]):  # H/2 --> H/16
                mid_mask_embedding = mask_embed_proj(mask_embedding)
                mid_mask_embedding = rearrange(mid_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
                logits.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mid_mask_embedding))
        return logits, latent_embedding_ls, image_embedding, per_pixel_embedding_ls

if __name__ == '__main__':
    model = Maskformer().cuda()    
    image = torch.rand((1, 3, 288, 288, 96)).cuda()
    query = torch.rand((2, 10, 768)).cuda()
    segmentations = model(query, image)
    print(segmentations.shape)