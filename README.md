# Neural Photographic Defect Detection

## System prerequisites
- Python 3.6+

## Required python packages and recommended versions (generated by pipreqs)
- numpy==1.15.4
- scipy==1.1.0
- torch==1.1.0
- torchvision==0.3.0
- opencv_python==4.0.0.21
- Pillow==6.1.0
- tensorboardX==1.8
- easydict==1.9

## Dataset preparation
1. Flickr dataset (collected from Flickr Creative Commons 100M dataset)
2. HDR+ dataset
3. AVA, Adobe5K, HDR+ mixed dataset (optional)

## Pretrained model preparation
1. ShuffleNet
2. ShuffleNetV2 (optional)
3. Resnet (optional)

## How to train
`python train.py [--options]`  
For example usages please look at `scripts`, and for the explanation of options
just run `python train.py -h`.

## How to evaluate
`python eval.py [--options]`  
General usage: `python eval.py --model_name [model_name] --epoch 100 --use_averaged_weight --test_spearman --test_objective --test_subjective`
(You may want to set the path to spearman, objective and subjective testing set properly before running the script)

## Problems
If you have any other problems,
please submit an issue or contact [mjust.lkc@gmail.com](mailto:mjust.lkc@gmail.com).
