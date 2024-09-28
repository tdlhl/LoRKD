import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from mamba_ssm import Mamba


class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        # seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedResidualBlocks(
                n_blocks = n_conv_per_stage[s-1],
                conv_op = encoder.conv_op,
                input_channels = 2 * input_features_skip,
                output_channels = input_features_skip,
                kernel_size = encoder.kernel_sizes[-(s + 1)],
                initial_stride = 1,
                conv_bias = encoder.conv_bias,
                norm_op = encoder.norm_op,
                norm_op_kwargs = encoder.norm_op_kwargs,
                dropout_op = encoder.dropout_op,
                dropout_op_kwargs = encoder.dropout_op_kwargs,
                nonlin = encoder.nonlin,
                nonlin_kwargs = encoder.nonlin_kwargs,
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            # seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        # self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            seg_outputs.append(x)
            #if self.deep_supervision:
            #    seg_outputs.append(self.seg_layers[s](x))
            #elif s == (len(self.stages) - 1):
            #    seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = [seg_outputs[0]]
        else:
            r = seg_outputs
        return r
    
    
class UMambaBot(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
                                                                
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
                                       
        # layer norm
        self.ln = nn.LayerNorm(features_per_stage[-1])
        self.mamba = Mamba(
                        d_model=features_per_stage[-1],
                        d_state=16,  
                        d_conv=4,    
                        expand=2,   
                    )
        self.decoder = UNetResDecoder(self.encoder, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        # encoder
        skips = self.encoder(x)
        # bottleneck mamba layer
        middle_feature = skips[-1]
        B, C = middle_feature.shape[:2]
        n_tokens = middle_feature.shape[2:].numel()
        img_dims = middle_feature.shape[2:]
        middle_feature_flat = middle_feature.view(B, C, n_tokens).transpose(-1, -2)
        middle_feature_flat = self.ln(middle_feature_flat) 
        out = self.mamba(middle_feature_flat)
        out = out.transpose(-1, -2).view(B, C, *img_dims)
        skips[-1] = out
        # decoder
        outs = self.decoder(skips)
        return skips, outs

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)
    
if __name__ == '__main__':
    import os
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    model = UMambaBot(
                    input_channels=3,
                    n_stages=6,
                    features_per_stage=(64, 64, 128, 256, 512, 768),
                    conv_op=nn.Conv3d,
                    kernel_sizes=3,
                    strides=(1, 2, 2, 2, 2, 2),
                    n_conv_per_stage=(1, 1, 1, 1, 1, 1),
                    n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
                    conv_bias=True,
                    norm_op=nn.InstanceNorm3d,
                    norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nn.LeakyReLU, 
                    nonlin_kwargs=None,
                    deep_supervision=True,
                ).cuda()
    
    image = torch.rand((1, 3, 288, 288, 96)).cuda()
    skips, outs = model(image)
    for s in skips:
        print(s.shape)
    for o in outs:
        print(o.shape)
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** UMambaBot ** {get_parameter_number(model)['Total']/1e6}M parameters")
        
    exit()
        
    encoder = ResidualEncoder(input_channels=3,
                    n_stages=6,
                    features_per_stage=(64, 64, 128, 256, 512, 768),
                    conv_op=nn.Conv3d,
                    kernel_sizes=3,
                    strides=(1, 2, 2, 2, 2, 2),
                    n_blocks_per_stage=(1, 1, 1, 1, 1, 1), 
                    conv_bias=True,
                    norm_op=nn.InstanceNorm3d,
                    norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nn.LeakyReLU, 
                    nonlin_kwargs=None,
                    deep_supervision=True,
                    )
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** Residual Encoder ** {get_parameter_number(encoder)['Total']/1e6}M parameters")
        
    decoder = UNetResDecoder(encoder, (1, 1, 1, 1, 1), deep_supervision=True)
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** Residual Decoder ** {get_parameter_number(decoder)['Total']/1e6}M parameters")
        
    model = ResidualEncoderUNet(input_channels=3, 
                                   n_stages=6, 
                                   features_per_stage=(64, 64, 128, 256, 512, 768), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_blocks_per_stage=(1, 1, 1, 1, 1, 1), 
                                   n_conv_per_stage_decoder=(2, 2, 2, 2, 2), 
                                   conv_bias=True, 
                                   norm_op=nn.InstanceNorm3d,
                                   norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                   dropout_op=None,
                                   dropout_op_kwargs=None,
                                   nonlin=nn.LeakyReLU, 
                                   nonlin_kwargs=None,
                                   deep_supervision=True
                                   )
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** Residual UNet ** {get_parameter_number(model)['Total']/1e6}M parameters")
        
    model = PlainConvUNet(input_channels=3, 
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
                        deep_supervision=True,
                        nonlin_first=False
                        )
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** UNET ** {get_parameter_number(model)['Total']/1e6}M parameters")
        
        
    encoder = PlainConvEncoder(
                    input_channels=3,
                    n_stages=6,
                    features_per_stage=(64, 64, 128, 256, 512, 768),
                    conv_op=nn.Conv3d,
                    kernel_sizes=3,
                    strides=(1, 2, 2, 2, 2, 2),
                    n_conv_per_stage=(2, 2, 2, 2, 2, 2),
                    conv_bias=True,
                    norm_op=nn.InstanceNorm3d,
                    norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nn.LeakyReLU, 
                    nonlin_kwargs=None,
                    return_skips=True
    )
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** UNET Encoder ** {get_parameter_number(encoder)['Total']/1e6}M parameters")
        
    decoder = UNetDecoder(encoder,
                        n_conv_per_stage=(2, 2, 2, 2, 2), 
                        deep_supervision=True)
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** UNET Decoder ** {get_parameter_number(decoder)['Total']/1e6}M parameters")