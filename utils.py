import cv2
import glob
import json
import matplotlib
import torch
import pdb as pdb
from PIL import Image, ImageFile
from torch import nn
import torch.optim as optim
import os
import numpy as np
import argparse
import sys


def save_image(relation, image_path, caption, is_correct):
    correct_dir = f"./validation-CLIP-DAv2/{relation}/correct"
    error_dir = f"./validation-CLIP-DAv2/{relation}/error"
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    img = Image.open(image_path)

    if is_correct:
        save_path = os.path.join(correct_dir, f"{caption}.jpg")
    else:
        save_path = os.path.join(error_dir, f"{caption}.jpg")
    
    img.save(save_path)


# Zero-shot 과 Fine-tuning 결과 비교
def save_image_with_comparison(relation, image_path, caption, is_correct):

    # CLIP False, Fine-tuning True
    pos_dir = f"./validation-with-zeroshot/{relation}/positive"
    # CLIP True, Fine-tuning False
    neg_dir = f"./validation-with-zeroshot/{relation}/negative"
    # CLIP True, Fine-tuning True
    neural_dir = f"./validation-with-zeroshot/{relation}/neural"
    
    os.makedirs(pos_dir, exist_ok = True)
    os.makedirs(neg_dir, exist_ok = True)
    os.makedirs(neural_dir, exist_ok = True)

    # Fine-tuning 결과가 저장되어 있는 폴더 
    correct_dir = f"./validation-CLIP-DAv2/{relation}/correct"
    error_dir = f"./validation-CLIP-DAv2/{relation}/error"

    # Zero-shot 이 맞았는데, Fine-tuning 에서는 틀렸다면 -> negative case 
    if is_correct: 
        file_path = os.path.join(correct_dir, f"{caption}.jpg")
        if os.path.isfile(file_path):
            new_file_path = os.path.join(neural_dir, f"{caption}.jpg")
        else: 
            new_file_path = os.path.join(neg_dir, f"{caption}.jpg")

    # Zero-shot 이 틀렸는데, Fine-tuning 에서는 맞았다면 -> positive case 
    else:
        file_path = os.path.join(error_dir, f"{caption}.jpg")
        if os.path.isfile(file_path):
            new_file_path = os.path.join(neural_dir, f"{caption}.jpg")
        else: 
            new_file_path = os.path.join(pos_dir, f"{caption}.jpg")

    img = Image.open(image_path)
    img.save(new_file_path)



def track_grads(model, save_dir="grads", file_name="train_mlp.json"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    for name, param in model.named_parameters():
        if param.requires_grad: # requires_grad(True)
            if name not in data:
                data[name] = []
            data[name].append(param.grad.norm().item())
        else:
            if name not in data: # requires_grad(False)
                data[name] = []
            data[name].append(param.grad)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def track_weight(model, save_dir="weights", file_name="train_clip_2.json"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    for name, param in model.named_parameters():
        #pdb.set_trace()
        print(name)
        if param.requires_grad: # requires_grad(True)
            if name not in data:
                data[name] = []
            data[name].append(param.tolist())
        elif not param.requires_grad: 
            if name not in data: # requires_grad(False)
                data[name] = []
            data[name].append(param.grad)

        if name == "proj":
            par = param.detach()
            par = par.tolist()
            #pdb.set_trace()

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



def adjust_parameter(args, model):
    # 모든 parameter 활성화 (lr = 2e-3, 2e-5)
    if args.train_module =="All":
        print("Train All")
        mlp_paras = []
        image_encoder_paras = []
        text_encoder_paras = []
        depth_encoder_paras = []
        
        
        for k, v in model.named_parameters():
            if 'depth_encoder' in k:
                depth_encoder_paras.append(v)
            elif 'image_encoder' in k:
                image_encoder_paras.append(v)
            elif 'text_encoder' in k:
                text_encoder_paras.append(v)
            else:
                mlp_paras.append(v)

        optimizer = optim.Adam(
                    [
                        {"params": mlp_paras, "lr": args.mlp_learning_rate},
                        {"params": image_encoder_paras, "lr": args.clip_learning_rate},
                        {"params": depth_encoder_paras, "lr": args.depth_learning_rate},
                        {"params": text_encoder_paras, "lr": args.clip_learning_rate}
                    ],
                    betas=args.betas, 
                    eps=args.eps, 
                    weight_decay=args.weight_decay
                )

    elif args.train_module =="DAv2":
        print("Train DAv2")
        mlp_paras = []
        depth_encoder_paras = []

        for k,v in model.named_parameters():
            v.requires_grad_(False)
        
        for k,v in model.named_parameters():
            if 'depth_encoder' in k:
                v.requires_grad_(True)
                depth_encoder_paras.append(v)
            
            elif k in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']:
                v.requires_grad_(True)
                mlp_paras.append(v)
                    
        optimizer = optim.Adam(
            [
                {"params": mlp_paras, "lr": args.mlp_learning_rate},
                {"params": depth_encoder_paras, "lr": args.depth_learning_rate},
            ],
            betas=args.betas, 
            eps=args.eps, 
            weight_decay=args.weight_decay
        )



    elif args.train_module =="CLIP":
        print("Train CLIP")
        image_encoder_paras = []
        text_encoder_paras = []
        mlp_paras = []
        for k,v in model.named_parameters():
            v.requires_grad_(False)
        
        for k,v in model.named_parameters():
            #print(k)
            if ('image_encoder' in k) and ('q_transform' in k or 'k_transform' in k or 'v_transform' in k): # image encoder 중에서 q, k, v mlp 파라미터
                v.requires_grad_(True)
                mlp_paras.append(v)

            elif 'image_encoder' in k and 'image_encoder.module.dinov2' not in k: # image encoder 중에서 dinov2 제외한 모든 파라미터
                v.requires_grad_(True)
                image_encoder_paras.append(v)

            elif 'text_encoder' in k:
                v.requires_grad_(True)
                text_encoder_paras.append(v)

            # elif k in ["logit_scale", "proj"]: # 중요!!! 
            #     v.requires_grad_(True)
            #     image_encoder_paras.append(v)

        
        optimizer = optim.Adam(
            [
                    {"params": mlp_paras, "lr": args.mlp_learning_rate},
                    {"params": image_encoder_paras, "lr": args.clip_learning_rate},
                    {"params": text_encoder_paras, "lr": args.clip_learning_rate}
            ],
            betas=args.betas, 
            eps=args.eps, 
            weight_decay=args.weight_decay
        )



    elif args.train_module =="mlp":
        print("Train MLP")
        mlp_paras = []
        for k, v in model.named_parameters():
            v.requires_grad_(False)
            if k in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']:
                v.requires_grad_(True)
                mlp_paras.append(v)


        optimizer = optim.Adam(
            [
                {"params": mlp_paras, "lr": args.mlp_learning_rate}
            ],
            betas=args.betas, 
            eps=args.eps, 
            weight_decay=args.weight_decay
        )


    return model, optimizer



from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

def analysis_embedding(image, depth):
    # CPU로 옮기고 reshape
    image = np.array(image.cpu())
    image = image.reshape(64, -1)

    depth = np.array(depth.cpu())
    depth = depth.reshape(64, -1)

    # 결과값을 저장할 리스트
    pearson_corrs = []
    spearmanr_corrs = []
    cosine_sims = []
    pca_results = []
    mean_diffs = []
    std_diffs = []

    # 각 샘플별로 비교
    for i in range(64):
        # 샘플별로 벡터 추출
        image_flat = image[i]
        depth_flat = depth[i]

        # Pearson 상관계수
        pearson_corr, _ = pearsonr(image_flat, depth_flat)
        pearson_corrs.append(pearson_corr)

        # Spearman 상관계수
        spearmanr_corr, _ = spearmanr(image_flat, depth_flat)
        spearmanr_corrs.append(spearmanr_corr)

        # 코사인 유사도
        cos_sim = cosine_similarity(image_flat.reshape(1, -1), depth_flat.reshape(1, -1))[0][0]
        cosine_sims.append(cos_sim)

        # PCA (두 임베딩 간 차원을 줄여 시각적으로 비교 가능)
        combined = np.vstack((image_flat, depth_flat))
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)
        pca_results.append(pca_result)

        # 차이 분석 (평균, 표준편차)
        diff = image_flat - depth_flat
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        mean_diffs.append(mean_diff)
        std_diffs.append(std_diff)

    # 결과를 평균 및 표준편차로 정리
    summary = {
        "Pearson Correlation": {
            "mean": np.mean(pearson_corrs),
            "std": np.std(pearson_corrs)
        },
        "Spearman Correlation": {
            "mean": np.mean(spearmanr_corrs),
            "std": np.std(spearmanr_corrs)
        },
        "Cosine Similarity": {
            "mean": np.mean(cosine_sims),
            "std": np.std(cosine_sims)
        },
        "Mean of Differences": {
            "mean": np.mean(mean_diffs),
            "std": np.std(mean_diffs)
        },
        "Std of Differences": {
            "mean": np.mean(std_diffs),
            "std": np.std(std_diffs)
        }
    }

    # 요약된 결과 출력
    for metric, values in summary.items():
        print(f"{metric}:")
        print(f"  Mean: {values['mean']:.4f}")
        print(f"  Std Dev: {values['std']:.4f}")
        print()