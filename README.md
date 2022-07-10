# L-CoDe: Language-based Colorization Using Color-object Decoupled Conditions

## Introducation
This is the author's official PyTorch L-CoDe implementation.
In this paper, we propose L-CoDe, a Language-based Colorization network using color-object Decoupled conditions. A predictor for object-color corresponding matrix (OCCM) and a novel attention transfer module (ATM) are introduced to solve the color-object coupling problem. To deal with color-object mismatch that results in incorrect color-object correspondence, we adopt a soft-gated injection module (SIM).

 <img src="./pipeline.png" width = "650" height = "230" alt="图片名称" align=center />

## Prerequisites
* Python 3.6
* PyTorch 1.0
* NVIDIA GPU + CUDA cuDNN

## Installation
Clone this repo

Install PyTorch and dependencies from http://pytorch.org

Install other python requirements

## Datasets
We process the [MSCOCO](https://cocodataset.org/) dataset for evaluation. Specifically, we keep the images whose captions contain adjectives and annotate the correspondence between adjectives and nouns in the caption to produce the ground-truth object-color corresponding matrix (OCCM). Metadata is in ``./resources``.

## Getting Started
Download the [coco2017 images](https://cocodataset.org/) and copy them under IMG_DIR.

Setting the MODEL_DIR as the storage directory for generated experimental results.

These directory parameters could be found in cfg/coco_train.yml and cfg/coco_test.yml. 

### 1) Training
```
python train.py --gm --o2c --train
```
### 2) Testing
```
python test.py --gm --o2c
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

## Citation
If you use this code for your research, please cite our papers [L-CoDe: Language-based Colorization Using Color-object Decoupled Conditions](https://ci.idm.pku.edu.cn/Weng_AAAI22.pdf)
```
@InProceedings{Weng_2022_AAAI,
  author = {Weng, Shuchen and Wu, Hao and Chang, Zheng and Tang, Jiajun and Li ,Si and Shi, Boxin},
  title = {L-CoDe: Language-based Colorization Using Color-object Decoupled Conditions},
  booktitle = {AAAI},
  month = {June},
  year = {2022}
}
```
