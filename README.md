# Mixed-Transformer

![poster](poster.png)


### Main Approach 
![Approach](Approach.png)



## Installation

### Environment Setting
```bash
$ git clone https://github.com/hy30n80/Mixed-Transformer.git
$ conda env create -f envi.yml
$ conda activate envi
```


### Installation for Depth Anything

Please Install [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) pre-trained weights (`vitl`). and put them under the `DAv2_checkpoints` directory.





## Usage 


### Dataset Preparation
```bash
$ python dataset-prep.py
```

### Train (CLIP w/ Spatial Attention)
```bash
$ python train.py --mini_batch_size 64 --batch_size 4096 --clip_base_model ViT-L/14@336px --train_module CLIP
```


### Test (CLIP w/ Spatial Attention)

Please Install finetuned weight from [Here](https://drive.google.com/file/d/1vWF0Jj6g8JEoyvRLlDD7_uAxTgMfGgMb/view?usp=drive_link). and put them under the `checkpoints` directory.

```bash
$ python train.py --only_evaluation True --checkpoint_path "your_path" --clip_base_model ViT-L/14@336px
```



## Visualization
![visualization](visualization.png)