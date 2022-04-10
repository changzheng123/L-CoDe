import json
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
from miscc.config import cfg

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1 and m.weight != None:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def load_weight(net, model_dir, model_folder, model_name, ckpt):
    if ckpt == -1:
        return
    net_weight_pth = os.path.join(model_dir, model_folder, model_name + str(ckpt) + '.pth')
    net_state_dict = torch.load(net_weight_pth)
    net.load_state_dict(net_state_dict)

def save_images(img_ls, img_abs, img_names, dir):
    img_labs = torch.cat((img_ls, img_abs.detach()), dim=1).cpu()
    img_labs = (img_labs * 0.5 + 0.5) * 255
    img_labs[img_labs > 255] = 255
    img_labs[img_labs < 0] = 0
    batch_size = img_labs.size(0)
    for i in range(batch_size):
        img_lab_np = np.array(img_labs[i]).astype(np.uint8)
        img_lab_np = np.transpose(img_lab_np, (1, 2, 0))
        img_rgb_np = cv2.cvtColor(img_lab_np, cv2.COLOR_Lab2RGB)
        img_rgb = Image.fromarray(img_rgb_np)
        img_path = os.path.join(dir, img_names[i].replace('jpg', 'png'))
        img_rgb.save(img_path)

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

def build_arc_mat(arc_dict,keys,batch_size,captions,ixtoword):
    arc_mats = []

    for b in range(batch_size):
        caption = captions.cpu()[b].tolist()
        # caption_str = [ixtoword[ix] for ix in caption]
        # print(caption_str)
        parser_list = arc_dict[keys[b]][0]
        # print(parser_list)
        mat = torch.zeros(cfg.TEXT.WORDS_NUM,cfg.TEXT.WORDS_NUM)
        for parser in parser_list:
            try:    
                mat[parser[0]-1][parser[1]-1] = 1
            except:
                continue
        # print(mat)
        arc_mats.append(mat)
    arc_mats = torch.stack(arc_mats)
    return arc_mats.cuda()