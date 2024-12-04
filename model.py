import cv2
import glob
import json
import matplotlib
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import pdb as pdb
from torch import nn
import torch.optim as optim
import os
import wandb
import numpy as np
import argparse
import sys
import clip
from utils import analysis_embedding

class CLIPTextEncoder(nn.Module):
    def __init__(self, model) :
        super(CLIPTextEncoder, self).__init__()
        #self.model = model
        self.transformer = model.transformer
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.dtype = model.visual.conv1.weight.dtype

        
    def forward(self,text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
        #return self.model.encode_text(text) # 
    
class ImageEncoder(nn.Module):
    def __init__(self, args, clip_model, DinoV2) :
        super(ImageEncoder, self).__init__()
        # self.model = model
        self.args = args
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        self.pretrained_proj = self.visual.proj # (1024, 768)

        self.dinov2 = DinoV2 # Encoder 부분만 가져오기
        self.intermediate_layer_idx = {
            'vitl': list(range(24)), 
        }
        self.encoder = 'vitl'



        if self.args.add_residual_linear_connection:
            self.v_transforms = nn.ModuleList([
                nn.Linear(1024, 1024).to("cuda") for _ in range(len(self.visual.transformer.resblocks))
            ])
            
            # 모든 Linear 레이어 초기화
            for v_transform in self.v_transforms:
                # nn.init.kaiming_normal_(v_transform.weight, nonlinearity='relu')
                nn.init.zeros_(v_transform.weight)
                nn.init.zeros_(v_transform.bias)
        

    def forward(self, image):
        # image : torch.Size([16 (B), 3, 336, 336])
        x = self.visual.conv1(image.type(self.visual.conv1.weight.dtype)) 
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1) 
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND [577, 16 (B), 1024]

        # query, key, value : torch.Size([24, 1, 16, 577, 1024])

        with torch.no_grad():
            _, query, key, value = self.dinov2.get_intermediate_layers(image, self.intermediate_layer_idx[self.encoder], return_class_token=True)
            
            if self.args.test_one_image:
                query = torch.stack([q.detach() for q in query], dim=0) # torch.Size([24 (layer_num), 16 (batch), 577, 1024]) # head 는? 
                key = torch.stack([k.detach() for k in key], dim=0)
                value = torch.stack([v.detach() for v in value], dim=0)
                print(query.shape)
            else:
                query = torch.stack([q.detach() for q in query], dim=0).squeeze(1) # torch.Size([24 (layer_num), 16 (batch), 577, 1024]) # head 는? 
                key = torch.stack([k.detach() for k in key], dim=0).squeeze(1)
                value = torch.stack([v.detach() for v in value], dim=0).squeeze(1)


        for i, block in enumerate(self.visual.transformer.resblocks):
            q_prime, k_prime, v_prime = query[i].permute(1,0,2), key[i].permute(1,0,2), value[i].permute(1,0,2) # torch.Size([16, 577, 1024])

            if self.args.add_residual_linear_connection:
                v_prime = self.v_transforms[i](v_prime)
            else:
                v_prime = None

            x = block(x, q_prime, k_prime, v_prime) # torch.Size([577, 16, 1024]) 
            
        #x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x

        #return self.visual(image.type(self.visual.conv1.weight.dtype))



class MixedCLIP(nn.Module):
    def __init__(self, args, clip_model, DinoV2, width=1024, output_dim=768):
        super(MixedCLIP, self).__init__()

        self.logit_scale = clip_model.logit_scale
        self.text_encoder = CLIPTextEncoder(clip_model)
        self.args = args
        self.image_encoder = ImageEncoder(self.args, clip_model, DinoV2)

        self.text_encoder = torch.nn.DataParallel(self.text_encoder)
        self.image_encoder = torch.nn.DataParallel(self.image_encoder)


    def encode_image(self, images):
        image_embedding = self.image_encoder(images)
        return image_embedding


    def encode_text(self, texts):
        text_embedding = self.text_encoder(texts)
        return text_embedding
    

    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image, logits_per_text = create_logits(image_features, text_features, logit_scale)
        return logits_per_image, logits_per_text



def create_logits(x1,x2,logit_scale):
    x1 = x1.float()
    x2 = x2.float()
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    return logits_per_x1, logits_per_x2

