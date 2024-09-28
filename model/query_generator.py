import torch
import numpy as np
import os
import torch.nn as nn

from einops import rearrange, reduce, repeat
from transformers import AutoTokenizer
import wandb

from .tokenizer import MyTokenizer
from .text_tower import Text_Tower
from .knowledge_encoder import Knowledge_Encoder
from .med_cpt import MedCPT, MedCPT_Embed
from .base_bert import BaseBERT, BaseBERT_Embed
from .random_embed import Random_Embed


def compute_average_gradient(module):
    # 初始化梯度总和和参数计数
    total_gradient = 0.0
    total_params = 0
    
    # 遍历module的所有参数
    for param in module.parameters():
        if param.grad is not None:
            # 累加此参数的梯度绝对值
            total_gradient += param.grad.abs().mean().item()
            total_params += 1
    
    # 计算平均梯度
    if total_params > 0:
        average_gradient = total_gradient / total_params
    else:
        average_gradient = None
    
    return average_gradient


class Query_Generator():
    def __init__(self, 
                 # our knowledge encoder
                 knowledge_encoder_checkpoint=None,
                 pubmedbert_checkpoint=None,
                 biolord_checkpoint=None,
                 # medcpt
                 cpt_checkpoint=None, 
                 finetuned_cpt_checkpoint=None, 
                 # basebert
                 basebert_checkpoint=None,
                 finetuned_basebert_checkpoint=None,
                 # random
                 random_embed_label_mapping=None,
                 finetuned_random_embed=None,
                 # other params
                 open_bert_layer=12,
                 open_modality_embed=False,
                 partial_load=False,
                 gpu_id=None,
                 device=None):
        """
        1. 使用our knowledge encoder--> pubmedbert/biolord_checkpoint + knowledge_encoder_checkpoint
        2. 使用cpt + modality embed --> cpt_checkpoint + finetuned_cpt_checkpoint(if resume training)
        3. 使用basebert + modality embed --> basebert_checkpoint + finetuned_basebert_checkpoint(if resume training) + partial_load(if inheriting modality embeddings)
        4. 使用random initialized and learnable embed --> random_embed_label_mapping + finetuned_random_embed(if resume training)
        """
        self.device = device
        
        if knowledge_encoder_checkpoint:
            self.knowledge_encoder = self._load_knowledge_encoder(knowledge_encoder_checkpoint=knowledge_encoder_checkpoint,
                                                                  pubmedbert_checkpoint=pubmedbert_checkpoint,
                                                                  biolord_checkpoint=biolord_checkpoint,
                                                                  open_bert_layer=open_bert_layer,
                                                                  open_modality_embed=open_modality_embed,
                                                                  partial_load=partial_load,
                                                                  gpu_id=gpu_id,
                                                                  device=device)
        else:
            self.knowledge_encoder = False

            
        if cpt_checkpoint:
            # NOTE: set partial_load=True when, for example, cpt_embed -> cpt_encoder, inheriting the modality embeds
            self.cpt_encoder = self._load_cpt_encoder(cpt_checkpoint, 
                                                      finetuned_cpt_checkpoint, 
                                                      open_bert_layer, 
                                                      open_modality_embed, 
                                                      partial_load, 
                                                      gpu_id, 
                                                      device)
            self.tokenizer = AutoTokenizer.from_pretrained(cpt_checkpoint)
        else:
            self.cpt_encoder = False
            
            
        if basebert_checkpoint:
            # NOTE: set partial_load=True when, for example, cpt_embed -> cpt_encoder, inheriting the modality embeds
            self.basebert_encoder = self._load_basebert_encoder(basebert_checkpoint, 
                                                                finetuned_basebert_checkpoint, 
                                                                open_bert_layer, 
                                                                open_modality_embed, 
                                                                partial_load, 
                                                                gpu_id, 
                                                                device)
            self.tokenizer = AutoTokenizer.from_pretrained(basebert_checkpoint)
        else:
            self.basebert_encoder = False
            
            
        if random_embed_label_mapping:
            self.random_embed = self._load_random_embed(random_embed_label_mapping, 
                                                        finetuned_random_embed, 
                                                        open_modality_embed,
                                                        partial_load,
                                                        gpu_id, 
                                                        device)
        else:
            self.random_embed = False
            

    def _load_any_model(self, model, gpu_id, device, checkpoint_path, partial_load=False):
        if "RANK" in os.environ:
            model = model.to(device)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            device = torch.device('cuda')
            model = nn.DataParallel(model)
            model.to(device)
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
         
        if checkpoint_path:       
            # atlas tower are excluded
            checkpoint['model_state_dict'] = {k:v for k,v in checkpoint['model_state_dict'].items() if 'atlas_tower' not in k and 'temperature' not in k}
            if partial_load:
                model_dict =  model.state_dict()
                # check difference
                unexpected_state_dict = [k for k in checkpoint['model_state_dict'].keys() if k not in model_dict.keys()]
                missing_state_dict = [k for k in model_dict.keys() if k not in checkpoint['model_state_dict'].keys()]
                unmatchd_state_dict = [k for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape != model_dict[k].shape]
                # load partial parameters
                state_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
                if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
                    print('The following parameters are unexpected in query generator checkpoint:\n', unexpected_state_dict)
                    print('The following parameters are missing in query generator checkpoint:\n', missing_state_dict)
                    print('The following parameters have different shapes in query generator checkpoint:\n', unmatchd_state_dict)
                    print('The following parameters are loaded in query generator :\n', state_dict.keys())
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                
            if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
                print(f"** QUERY ** Load encoder from {checkpoint_path}.")
                
        return model
    
    def _load_knowledge_encoder(self, 
                                knowledge_encoder_checkpoint,   # must be given
                                pubmedbert_checkpoint,
                                biolord_checkpoint,
                                open_bert_layer, 
                                open_modality_embed, 
                                partial_load,
                                gpu_id, 
                                device):
        
        text_tower = Text_Tower(
            pubmedbert_checkpoint = pubmedbert_checkpoint,
            biolord_checkpoint = biolord_checkpoint,
            embed_dim = 768,
            open_bert_layer = open_bert_layer,
            max_text_length = 64
            )
        
        model = Knowledge_Encoder(
            text_tower=text_tower
            )   
        
        model = self._load_any_model(model, gpu_id, device, knowledge_encoder_checkpoint, partial_load=partial_load)
        
        # open text encoder in training SAT
        for name, param in model.named_parameters():
            if 'encoder.layer.' in name and int(name.split('encoder.layer.')[-1].split('.')[0])>open_bert_layer:  # encoder.layer.11.xxx --> 11
                param.requires_grad = True
            elif open_bert_layer < 11 and ('pooler' in name or 'mlp_embed' in name):
                param.requires_grad = True
            elif open_modality_embed and 'modality_embed' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        return model
    
    def _load_cpt_encoder(self, 
                          cpt_checkpoint, 
                          finetuned_cpt_checkpoint, 
                          open_bert_layer,
                          open_modality_embed, 
                          partial_load, 
                          gpu_id, 
                          device):
        model = MedCPT(cpt_checkpoint)
        
        model = self._load_any_model(model, gpu_id, device, finetuned_cpt_checkpoint, partial_load=partial_load)
                
        # open text encoder in training SAT
        for name, param in model.named_parameters():
            if open_modality_embed and 'modality_embed' in name:
                param.requires_grad = True
            elif 'encoder.layer.' in name and int(name.split('encoder.layer.')[-1].split('.')[0])>open_bert_layer:  # encoder.layer.11.xxx --> 11
                param.requires_grad = True
            elif open_bert_layer < 11 and 'pooler' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        return model
    
    def _load_basebert_encoder(self, 
                               basebert_checkpoint, 
                               finetuned_basebert_checkpoint, 
                               open_bert_layer,
                               open_modality_embed,
                               partial_load,
                               gpu_id, 
                               device):
        model = BaseBERT(basebert_checkpoint)
        
        model = self._load_any_model(model, gpu_id, device, finetuned_basebert_checkpoint, partial_load)
                
        # open text encoder in training SAT
        for name, param in model.named_parameters():
            if open_modality_embed and 'modality_embed' in name:
                param.requires_grad = True
            elif 'encoder.layer.' in name and int(name.split('encoder.layer.')[-1].split('.')[0])>open_bert_layer:  # encoder.layer.11.xxx --> 11
                param.requires_grad = True
            elif open_bert_layer < 11 and 'pooler' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        return model
    
    def _load_random_embed(self, 
                           random_embed_label_mapping, 
                           finetuned_random_embed,
                           open_modality_embed,
                           partial_load,
                           gpu_id, 
                           device):
        
        model = Random_Embed(random_embed_label_mapping)
        
        model = self._load_any_model(model, gpu_id, device, finetuned_random_embed, partial_load=partial_load)
        
        for name, param in model.named_parameters():
            if open_modality_embed and 'modality_embed' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        return model
        
    def get_query(self, label_name_ls, modality_ls):
        """
        Args:
            label_name_ls (List of List / List of Str): B x N / N
            modality_ls (List / Str): B / 1
            
        Return:
            queries (Tensor): B x N / N
        """
        if isinstance(label_name_ls[0], list):
            batch_size = len(label_name_ls)
            num_query = len(label_name_ls[0])
            input_text = [t for t_ls in label_name_ls for t in t_ls]    # BN
            modality = [mod for mod in modality_ls for n in range(num_query)] # repeat each mod for N times -> BN
        else:
            num_query = len(label_name_ls)
            input_text = label_name_ls  # N
            modality = [modality_ls[0] for n in range(num_query)]   # N
            
        # name to code
        modality_code_dict = {
                'ct':0,
                'mri':1,
                'us':2,
                'pet':3,
            }
        modality_code = torch.tensor([modality_code_dict[mod] for mod in modality])   # bn
            
        # if self.knowledge_embeddings_dict:
        #     # retrieval embeddings by name
        #     query_embed = []
        #     for t,m in zip(input_text,modality):
        #         if t == 'none':
        #             query_embed.append(self.knowledge_embeddings_dict['none'])    #  list of (1, 768)
        #         else:
        #             query_embed.append(self.knowledge_embeddings_dict[f'{m}_{t}'])
        #     queries = torch.tensor(np.concatenate(query_embed, axis=0))    # [b*n, 768]
        
        if self.knowledge_encoder:
            text_feature, proj_text_feature = self.knowledge_encoder(text=input_text, modality=modality_code) # b*n, 768  
            queries = proj_text_feature
            
        elif self.cpt_encoder:
            encoded = self.tokenizer(
                input_text, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=64,
            )
            encoded = encoded.to(self.device)
            queries = self.cpt_encoder(encoded, modality_code)
            
        elif self.basebert_encoder:
            encoded = self.tokenizer(
                input_text, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=64,
            )
            encoded = encoded.to(self.device)
            queries = self.basebert_encoder(encoded, modality_code)
            
        elif self.random_embed:
            queries = self.random_embed(input_text, modality_code)
        
        if isinstance(label_name_ls[0], list):
            queries = rearrange(queries, '(b n) d -> b n d', b=batch_size, n=num_query)

        return queries  
    
    def save_any_model(self, step, path):
        # if self.knowledge_embeddings_dict:
        #     return
        
        if self.knowledge_encoder:
            torch.save({'step':step,
                        'model_state_dict': self.knowledge_encoder.state_dict(),
                        }, os.path.join(path))
            
        elif self.cpt_encoder:
            torch.save({'step':step,
                        'model_state_dict': self.cpt_encoder.state_dict(),
                        }, os.path.join(path))
            
        elif self.basebert_encoder:
            torch.save({'step':step,
                        'model_state_dict': self.basebert_encoder.state_dict(),
                        }, os.path.join(path))
            
        elif self.random_embed:
            torch.save({'step':step,
                        'model_state_dict': self.random_embed.state_dict(),
                        }, os.path.join(path))
            
    def parameters(self):
        # if self.knowledge_embeddings_dict:
        #     return None
        
        if self.knowledge_encoder:
            return self.knowledge_encoder.parameters()
            
        elif self.cpt_encoder:
            return self.cpt_encoder.parameters()
            
        elif self.basebert_encoder:
            return self.basebert_encoder.parameters()
        
        elif self.random_embed:
            return self.random_embed.parameters()
        
    def eval(self):
        # if self.knowledge_embeddings_dict:
        #     return
        
        if self.knowledge_encoder:
            self.knowledge_encoder.eval()
            
        elif self.cpt_encoder:
            self.cpt_encoder.eval()
            
        elif self.basebert_encoder:
            self.basebert_encoder.eval()
            
        elif self.random_embed:
            self.random_embed.eval()
            
    def train(self):
        # if self.knowledge_embeddings_dict:
        #     return
        
        if self.knowledge_encoder:
            self.knowledge_encoder.train()
            
        elif self.cpt_encoder:
            self.cpt_encoder.train()
        
        elif self.basebert_encoder:
            self.basebert_encoder.train()
            
        elif self.random_embed:
            self.random_embed.train()
            
    def display_avg_grad(self):
        # if self.knowledge_embeddings_dict:
        #     avg_grad = None
        
        if self.knowledge_encoder:
            avg_grad = compute_average_gradient(self.knowledge_encoder)

        elif self.cpt_encoder:
           avg_grad = compute_average_gradient(self.cpt_encoder)

        elif self.basebert_encoder:
           avg_grad = compute_average_gradient(self.basebert_encoder)
           
        elif self.random_embed:
           avg_grad = compute_average_gradient(self.random_embed)
        
        print(f'Avg Grad Over Trainable Query Encoder Params: {avg_grad}')
    
    def named_parameters(self):
        # if self.knowledge_embeddings_dict:
        #     named_parameters = {}
        
        if self.knowledge_encoder:
            named_parameters = self.knowledge_encoder.named_parameters()

        elif self.cpt_encoder:
           named_parameters = self.cpt_encoder.named_parameters()

        elif self.basebert_encoder:
           named_parameters = self.basebert_encoder.named_parameters()
           
        elif self.random_embed:
           named_parameters = self.random_embed.named_parameters()
           
        return named_parameters
    
    def watch(self, log_freq):
        # if self.knowledge_embeddings_dict:
        #     pass
        
        if self.knowledge_encoder:
            wandb.watch(models=self.knowledge_encoder, log='gradients', log_freq=log_freq)

        elif self.cpt_encoder:
           wandb.watch(models=self.cpt_encoder, log='gradients', log_freq=log_freq)

        elif self.basebert_encoder:
           wandb.watch(models=self.basebert_encoder, log='gradients', log_freq=log_freq)
           
        elif self.random_embed:
           wandb.watch(models=self.random_embed, log='gradients', log_freq=log_freq)
        
           
        
        
        
        