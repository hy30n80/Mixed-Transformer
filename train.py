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

from utils import track_grads, track_weight, count_parameters, count_trainable_parameters, adjust_parameter, save_image, save_image_with_comparison
from model import CLIPTextEncoder, ImageEncoder, MixedCLIP
from data import create_dataset, image_title_dataset, create_dataset_per_relation
from tqdm import tqdm

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def create_logits(x1,x2,logit_scale):
    x1 = x1.float()
    x2 = x2.float()
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    return logits_per_x1, logits_per_x2


def remap_keys(checkpoint, prefix="pretrained."):
    remapped_checkpoint = {}
    for k, v in checkpoint.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # prefix 제거
            remapped_checkpoint[new_key] = v
    return remapped_checkpoint


def train(args, device, model, preprocess, train_dataloader, test_dataloader, train_dataset, test_dataset):
    
    model, optimizer = adjust_parameter(args, model)

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']


    args.all_params = count_parameters(model)
    print("All params", args.all_params)
    args.trainable_params = count_trainable_parameters(model)
    print("Trainable params", args.trainable_params)


    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    wandb_name = "Mixed-transformer-EmbSpatial"

    wandb.init(
        project=wandb_name, 
        entity = "yanghyeonjun",
        config=args)


    total_loss = 0
    best_accuracy = -np.Inf
    best_iter = 0
    checkpoint_path = None
    step_global = 0
    total_epoch = 1 if args.only_evaluate else args.epoch


    for epoch in range(total_epoch):
        wandb_log = {}
        # validate epoch
        validate = (args.only_evaluate or step_global % args.eval_step == 0) and not args.skip_evaluate
        if validate:
            model.eval()
            num_correct = 0
            num_errors = 0
            test_losses = []
            for batch in tqdm(test_dataloader):
                images, texts, negated_captions, _ , _ = batch

                texts = torch.cat((texts, negated_captions), 0)  
                images = images.to(device)
                texts = texts.to(device)


                with torch.no_grad():
                    image_embedding = model.encode_image(images)
                    text_embedding = model.encode_text(texts)

                    logit_scale = model.logit_scale.exp()
                    logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)


                # calculate loss
                len_for_loss = int(logits_per_image.shape[1]/2)
                logits_per_image_for_loss = logits_per_image[:,:len_for_loss] 
                logits_per_text_for_loss = logits_per_text[:len_for_loss,:]
                
                ground_truth = torch.arange(
                    len(images), dtype=torch.long, device=device)

                total_loss = (loss_img(logits_per_image_for_loss, ground_truth) +
                    loss_txt(logits_per_text_for_loss, ground_truth))/2
                test_losses.append(total_loss.item())

                # calculate accuracy
                for i in range(len(logits_per_image)):
                    possitive = logits_per_text[i,i]
                    negative = logits_per_text[i+len(logits_per_image),i]
                    if possitive > negative:
                        num_correct += 1
                    else:
                        num_errors += 1

                        
            test_accuracy = float(num_correct) / float(num_correct + num_errors)
            test_loss = np.mean(test_losses)

            print (f"====== evaluate ======")
            print (f"epoch: {epoch}, global step: {step_global}, test_loss: {test_loss}, test_accuracy: {test_accuracy}")
            print (f"=======================")
            wandb_log["test_loss"] = test_loss
            wandb_log["test_accuracy"] = test_accuracy

            if test_accuracy > best_accuracy:
                best_iter = epoch+1
                best_accuracy = test_accuracy

                checkpoint_dir = os.path.join("checkpoint")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                old_checkpoint_path = checkpoint_path

                checkpoint_path = os.path.join(checkpoint_dir, f'Mixed_Transformer_{epoch:04d}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    }, checkpoint_path) 
                if old_checkpoint_path is not None:
                    os.remove(old_checkpoint_path)
                print (f"===== best model saved! =======")
            
            
            # Accuracy per Relation
            if args.validate_per_relation:
                relation =  ['close', 'far', 'left', 'right', 'above', 'under'] 

                for rel in relation:

                    test_json_path = os.path.join('../EmbSpatial', 'split_sft', 'test.json')
                    img_path = os.path.join('../EmbSpatial', 'Image', 'embspatial_sft')

                    test_dataset_by_rel = create_dataset_per_relation(test_json_path, img_path, preprocess, rel)
                    test_dataloader_by_rel = DataLoader(test_dataset_by_rel, batch_size=64, drop_last=True)

                    num_correct = 0
                    num_errors = 0

                    for batch in tqdm(test_dataloader_by_rel):
                        images, texts, negated_captions, image_paths, naive_captions = batch

                        texts = torch.cat((texts, negated_captions), 0)  # dim=0

                        images = images.to(device)
                        texts = texts.to(device)


                        with torch.no_grad():
                            image_embedding = model.encode_image(images)
                            text_embedding = model.encode_text(texts)

                            logit_scale = model.logit_scale.exp()
                            logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)


                        # calculate loss
                        len_for_loss = int(logits_per_image.shape[1]/2)
                        logits_per_image_for_loss = logits_per_image[:,:len_for_loss]
                        logits_per_text_for_loss = logits_per_text[:len_for_loss,:] 
                
                        # calculate accuracy
                        for i in range(len(logits_per_image)):
                            possitive = logits_per_text[i,i]
                            negative = logits_per_text[i+len(logits_per_image),i]

                            image_path = image_paths[i]
                            caption = naive_captions[i]

                            if possitive > negative:
                                num_correct += 1                     
                                #save_image(rel, image_path, caption, is_correct=True)
                                #save_image_with_comparison(rel, image_path, caption, is_correct=True)
                            else:
                                num_errors += 1
                                #save_image(rel, image_path, caption, is_correct=False)
                                #save_image_with_comparison(rel, image_path, caption, is_correct=False)

                        
                    test_accuracy_by_rel = float(num_correct) / float(num_correct + num_errors)
                    print(f"{rel}_accuracy : {test_accuracy_by_rel}")
                    wandb_log[f"{rel}_accuracy"] = test_accuracy_by_rel
            



        # training epoch
        if not args.only_evaluate:
            model.train()
            optimizer.zero_grad()
            cur_batch_completed = 0
            total_loss = 0.0
            num_loss = 0.0
            num_correct = 0
            num_errors = 0
            def step():
                if device == "cpu":
                    optimizer.step()
                else:
                    convert_models_to_fp32(model.image_encoder.module.visual)
                    convert_models_to_fp32(model.text_encoder)
                    optimizer.step()
                    clip.model.convert_weights(model.image_encoder.module.visual)
                    clip.model.convert_weights(model.text_encoder)



            for batch in tqdm(train_dataloader):
                images, texts, negated_captions = batch

                texts = torch.cat((texts, negated_captions), 0)  # dim=0

                images = images.to(device)
                texts = texts.to(device)

                image_embedding = model.encode_image(images)
                text_embedding = model.encode_text(texts)


                logit_scale = model.logit_scale.exp()
                logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)

                # calculate loss
                len_for_loss = int(logits_per_image.shape[1]/2)
                logits_per_image_for_loss = logits_per_image[:,:len_for_loss] # 아, original caption 에 대해서만 확률값을 구하도록
                logits_per_text_for_loss = logits_per_text[:len_for_loss,:] # ""


                ground_truth = torch.arange(
                    len(images), dtype=torch.long, device=device)

                loss = (loss_img(logits_per_image_for_loss, ground_truth) +
                            loss_txt(logits_per_text_for_loss, ground_truth))/2

                loss.backward()

                total_loss += loss.item()
                num_loss += 1


                # calculate accuracy
                for i in range(len(logits_per_image)):
                    possitive = logits_per_text[i,i]
                    negative = logits_per_text[i+len(logits_per_image),i]
                    if possitive > negative:
                        num_correct += 1
                    else: 
                        num_errors += 1

                cur_batch_completed += len(batch[0])   
                if cur_batch_completed >= args.batch_size:
                    print("Optimization Step")
                    step()
                    cur_batch_completed = 0
                    optimizer.zero_grad()
                    average_loss = total_loss / num_loss
                    total_loss = 0.0
                    num_loss = 0.0

            
            
            if cur_batch_completed > 0:
                step()
                average_loss = total_loss / num_loss


            train_accuracy = float(num_correct) / float(num_correct + num_errors) 
            wandb_log["train_accuracy"] = train_accuracy
            wandb_log["loss"] = average_loss
            wandb_log["lr"] = optimizer.param_groups[0]['lr']
            print("Epoch: {:04d}, Loss: {}".format(epoch, average_loss))
            print(f"Correct : {num_correct},Wrong {num_errors}")
            print("Train Accuracy: ", train_accuracy)
            step_global += 1 
            wandb.log(wandb_log)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--mini_batch_size', type=int, default=64)
    parser.add_argument('--eval_step', type=int, default=10)
    parser.add_argument('--betas', type=float, default=(0.9, 0.98))
    parser.add_argument('--eps', type=float, default=10e-6)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--only_evaluate', type=bool, default=False)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--skip_evaluate', type=bool, default=False)

    parser.add_argument('--clip_base_model', type=str, default="ViT-L/14@336px")
    parser.add_argument('--checkpoint_path', type=str, required=False)
    parser.add_argument('--DAv2_encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--train_module', type=str, default='CLIP', choices=['All', 'DAv2', 'CLIP', 'mlp'])

    parser.add_argument('--mlp_learning_rate', type=float, default=2e-4) #2e-5
    parser.add_argument('--clip_learning_rate', type=float, default=2e-5) #2e-5


    parser.add_argument('--validate_per_relation', type=bool, default=True)
    parser.add_argument('--train_only_relation', type=str, default=None)
    parser.add_argument('--add_residual_linear_connection', type=bool, default=False)

    args = parser.parse_args()
    
    assert args.batch_size % args.mini_batch_size == 0
    assert args.mini_batch_size <= args.batch_size


    torch.manual_seed(args.random_seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Must set jit=False for training
    clip_model, preprocess = clip.load(args.clip_base_model, device=device, jit=False)

    dAv2_model_configs = {'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}}
    
    depth_anything = DepthAnythingV2(**dAv2_model_configs[args.DAv2_encoder])
    depth_checkpoint = torch.load(f'./DAv2_checkpoints/depth_anything_v2_vitl.pth', map_location='cpu')
    DinoV2_checkpoint = remap_keys(depth_checkpoint)

    DinoV2 = depth_anything.pretrained
    DinoV2.load_state_dict(DinoV2_checkpoint)
    DinoV2 = DinoV2.to(device)

    model = MixedCLIP(args, clip_model, DinoV2)


    #pdb.set_trace()
    json_path = os.path.join('../EmbSpatial', 'split_sft', 'train.json') #filtered_sft
    test_json_path = os.path.join('../EmbSpatial', 'split_sft', 'test.json') #filtered_sft

    img_path = os.path.join('../EmbSpatial', 'Image', 'embspatial_sft')
    depth_path = os.path.join('../EmbSpatial', 'Depth', 'embspatial_sft')

    train_dataset = create_dataset(args, json_path, img_path, preprocess, mode="train")
    test_dataset = create_dataset(args, test_json_path, img_path, preprocess, mode="test")

    # Define your own dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.mini_batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, drop_last=True)


    train(args, device, model, preprocess, train_dataloader, test_dataloader, train_dataset, test_dataset)



# CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py --mini_batch_size 64 --batch_size 4096 --clip_base_model ViT-L/14@336px  --eval_step 1 --train_module CLIP --validate_per_relation True


