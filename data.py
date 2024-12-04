import cv2
import glob
import json
import matplotlib
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import pdb as pdb
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torch import nn
import torch.optim as optim
import os
import wandb
import numpy as np
import argparse
import sys
import clip
from torchsummary import summary



class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_caption, list_negated_caption, preprocess, mode="test"):
        # list_image_path = ["data/trainval2017/0000343.jpg", "....".....]
        # list_txt = ["The carrot is in the bowl (True)", "...",....]
        self.caption_tokens = clip.tokenize(list_caption)
        self.negated_caption_tokens = clip.tokenize(list_negated_caption)
        self.mode = mode
        self.preprocess = preprocess

        self.image_path = list_image_path
        self.captions = list_caption
  
    def __len__(self):
        return len(self.caption_tokens)

    def __getitem__(self, idx):
        if self.mode == "train":
            # Image from PIL module
            image = self.preprocess(Image.open(self.image_path[idx]))
            caption_token = self.caption_tokens[idx]
            negated_caption_token = self.negated_caption_tokens[idx]
            return image, caption_token, negated_caption_token
        
        elif self.mode == "test":
            image = self.preprocess(Image.open(self.image_path[idx]))
            caption_token = self.caption_tokens[idx]
            negated_caption_token = self.negated_caption_tokens[idx]

            # Case-Study 를 위함
            img_path = self.image_path[idx]
            caption = self.captions[idx]

            return image, caption_token, negated_caption_token, img_path, caption



def create_dataset(args, json_path, img_path, preprocess, mode="train"):
    data_json = []
    with open(json_path, "r") as f:
        data_json = json.load(f)

    list_image_path = []
    list_caption = []
    list_negated_caption = []

    # 특정 종류의 위치 관계에 대해서만 학습
    if args.train_only_relation and mode =="train":
        print(f"We train just about {args.train_only_relation}")
        for data in data_json:
            if data['relation'] == args.train_only_relation:        
                list_image_path.append(os.path.join(img_path, data['image']))
                caption = data['caption']
                negated_caption = data['structural_negated_caption']
                
                list_caption.append(caption)
                list_negated_caption.append(negated_caption)
        
    # 모든 위치 관계에 대한 데이터 셋 학습
    else:
        for data in data_json:
            list_image_path.append(os.path.join(img_path, data['image']))
            caption = data['caption']
            negated_caption = data['structural_negated_caption']
            
            list_caption.append(caption)
            list_negated_caption.append(negated_caption)
    
    dataset = image_title_dataset(list_image_path, list_caption, list_negated_caption, preprocess, mode)
    return dataset



def create_dataset_per_relation(json_path, img_path, preprocess, rel, mode="test"):
    data_json = []
    with open(json_path, "r") as f:
        data_json = json.load(f)
            
    list_image_path = []
    list_caption = []
    list_negated_caption = []

    for data in data_json:

        if data['relation'] == rel:        
            list_image_path.append(os.path.join(img_path, data['image']))
            caption = data['caption']
            negated_caption = data['structural_negated_caption']
            
            list_caption.append(caption)
            list_negated_caption.append(negated_caption)

    dataset = image_title_dataset(list_image_path, list_caption, list_negated_caption, preprocess, mode)
    return dataset