import torch
import torch.nn as nn
from collections import OrderedDict
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks, StackedConvBlocks_lora, ConvDropoutNormReLU, ConvDropoutNormReLU_lora

class AdapterWrapperUNet(nn.Module):
    def __init__(self, model, adapter_class, num_tasks, gamma, lora_alpha):
        super().__init__()
        self.model = model
        self.unet = model.module.backbone  
        self.add_multi_adapter(adapter_class, num_tasks, gamma, lora_alpha)
        self.model_frozen = False

    def add_multi_adapter(self, adapter_class, num_task, gamma, lora_alpha):
        """
        Add LoRA adapters to both encoder and decoder of the 3D U-Net
        """
        # Apply to encoder stages
        for stage in self.unet.encoder.stages:
            for block in stage:
                if isinstance(block, StackedConvBlocks_lora):
                    for module in block.stages:
                        if isinstance(module, ConvDropoutNormReLU_lora):
                            target_conv = getattr(module, 'conv')
                            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, num_task=num_task)
                            setattr(module, 'conv', adapter)
        
        # Apply to decoder stages
        for block in self.unet.decoder.stages:
            if isinstance(block, StackedConvBlocks_lora):
                for module in block.stages:
                    if isinstance(module, ConvDropoutNormReLU_lora):
                        target_conv = getattr(module, 'conv')
                        adapter = adapter_class(r=gamma*2, lora_alpha=lora_alpha, conv_layer=target_conv, num_task=num_task)
                        setattr(module, 'conv', adapter)

        
    def forward(self, queries=None, image_input=None, vision_feature_ls=None, task_labels=None):
        return self.model(queries=queries, image_input=image_input, task_labels=task_labels)

    def calculate_training_parameter_ratio(self):
        def count_parameters(model, grad):
            return sum(p.numel() for p in model.parameters() if p.requires_grad == grad)

        trainable_param_num = count_parameters(self.resnet, True)
        other_param_num = count_parameters(self.resnet, False)
        print("Non-trainable parameters:", other_param_num)
        print("Trainable parameters:", trainable_param_num)

        ratio = trainable_param_num / other_param_num
        final_ratio = (ratio / (1 - ratio))
        print("Ratio:", final_ratio)

        return final_ratio

    def adapter_state_dict(self):
        """
        Save only adapter parts
        """
        state_dict = self.state_dict()
        adapter_dict = OrderedDict()

        for name, param in state_dict.items():
            if "lora_" in name:
                adapter_dict[name] = param
            elif "bn" in name:
                adapter_dict[name] = param
            elif "bias" in name:
                if "fc" not in name:
                    adapter_dict[name] = param
        return adapter_dict

    def freeze_model(self, freeze=True): # 
        """Freezes all weights of the model."""
        if freeze: 
            # First freeze/ unfreeze all model weights
            for n, p in self.named_parameters():
                if 'lora_' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            for n, p in self.named_parameters():
                if 'bias' in n:
                    if "fc" not in n:
                        p.requires_grad = True
                elif "bn" in n:
                    p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.model_frozen = freeze

class AdapterWrapperUNet_imbalance(nn.Module):
    def __init__(self, model, adapter_class, num_tasks, gamma, lora_alpha):
        super().__init__()
        self.model = model
        self.unet = model.module.backbone  # This should be an instance of your 3D U-Net model
        self.add_multi_adapter(adapter_class, num_tasks, gamma, lora_alpha)
        self.model_frozen = False

    def add_multi_adapter(self, adapter_class, num_task, gamma, lora_alpha):
        """
        Add LoRA adapters to both encoder and decoder of the 3D U-Net
        """
        # Apply to encoder stages
        for stage in self.unet.encoder.stages:
            for block in stage:
                if isinstance(block, StackedConvBlocks_lora):
                    for module in block.stages:
                        if isinstance(module, ConvDropoutNormReLU_lora):
                            target_conv = getattr(module, 'conv')
                            adapter = adapter_class(r_list=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, num_task=num_task)
                            setattr(module, 'conv', adapter)
        
        gamma_decoder = [i*2 for i in gamma]
        # Apply to decoder stages
        for block in self.unet.decoder.stages:
            if isinstance(block, StackedConvBlocks_lora):
                for module in block.stages:
                    if isinstance(module, ConvDropoutNormReLU_lora):
                        target_conv = getattr(module, 'conv')
                        adapter = adapter_class(r_list=gamma_decoder, lora_alpha=lora_alpha, conv_layer=target_conv, num_task=num_task)
                        setattr(module, 'conv', adapter)

    def forward(self, queries=None, image_input=None, vision_feature_ls=None, task_labels=None):
        return self.model(queries=queries, image_input=image_input, task_labels=task_labels)

    def calculate_training_parameter_ratio(self):
        def count_parameters(model, grad):
            return sum(p.numel() for p in model.parameters() if p.requires_grad == grad)

        trainable_param_num = count_parameters(self.resnet, True)
        other_param_num = count_parameters(self.resnet, False)
        print("Non-trainable parameters:", other_param_num)
        print("Trainable parameters:", trainable_param_num)

        ratio = trainable_param_num / other_param_num
        final_ratio = (ratio / (1 - ratio))
        print("Ratio:", final_ratio)

        return final_ratio

    def adapter_state_dict(self):
        """
        Save only adapter parts
        """
        state_dict = self.state_dict()
        adapter_dict = OrderedDict()

        for name, param in state_dict.items():
            if "lora_" in name:
                adapter_dict[name] = param
            elif "bn" in name:
                adapter_dict[name] = param
            elif "bias" in name:
                if "fc" not in name:
                    adapter_dict[name] = param
        return adapter_dict

    def freeze_model(self, freeze=True): # 
        """Freezes all weights of the model."""
        if freeze: 
            # First freeze/ unfreeze all model weights
            for n, p in self.named_parameters():
                if 'lora_' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            for n, p in self.named_parameters():
                if 'bias' in n:
                    if "fc" not in n:
                        p.requires_grad = True
                elif "bn" in n:
                    p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.model_frozen = freeze

