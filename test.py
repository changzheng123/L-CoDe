import argparse
import os
import json
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
import cv2
from nltk.util import print_string
import numpy as np

from miscc.config import cfg, cfg_from_file
from model.Gmodel import GNet
from model.Emodel import RNN_ENCODER
from datasets import TextDataset, prepare_data
from nltk.tokenize import RegexpTokenizer
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='test a L-CoDe.')
    parser.add_argument('--model_folder',default='',
                        type = str,
                        help="model_folder, dir name, not path")
    parser.add_argument('--ckpt',default=13,
                        type = int,
                        help="chekpoint")
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/coco_test.yml', type=str)
    parser.add_argument('--wogt', action="store_true",
                        help="without gatemask") 
    parser.add_argument('--gm', action="store_true",
                        help="if use gatemask")
    parser.add_argument('--o2c', action="store_true",
                        help="if use obj2color")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg_from_file(args.cfg_file)
    cfg.MODEL_FOLDER = args.model_folder
    cfg.TRAIN.CKPT = args.ckpt
    cfg.TRAIN.USEMASK = args.gm
    cfg.TRAIN.OBJ2COL = args.o2c

    val_img_dir = os.path.join(cfg.IMG_DIR, 'val2017')
    val_caption_path = cfg.RESOURCE_DIR
    val_transform = transforms.Compose([
        transforms.Resize((cfg.TRAIN.HEIGHT, cfg.TRAIN.WIDTH))])
    val_dataset = TextDataset(val_img_dir,val_caption_path, val_transform,"val")
    word2index = val_dataset.wordtoix
    word_num = len(word2index)
    index2word = val_dataset.ixtoword

    # load model
    word_encoder = RNN_ENCODER(val_dataset.n_words,nhidden=cfg.TEXT.EMBEDDING_DIM)
    word_encoder_weight_path = os.path.join(cfg.MODEL_DIR, cfg.MODEL_FOLDER, 'emb_' + str(cfg.TRAIN.CKPT) + '.pth')
    word_encoder_state_dict = torch.load(word_encoder_weight_path)
    # word_encoder.load_state_dict(word_encoder_state_dict)
    word_encoder.eval()
    word_encoder = word_encoder.cuda()
    word_encoder = nn.DataParallel(word_encoder, device_ids=cfg.GPU_group)
    word_encoder.load_state_dict(word_encoder_state_dict)

    generator = GNet(cfg.TEXT.EMBEDDING_DIM)
    generator_weight_path = os.path.join(cfg.MODEL_DIR, cfg.MODEL_FOLDER, 'generator_' + str(cfg.TRAIN.CKPT) + '.pth')
    generator_state_dict = torch.load(generator_weight_path)
    # generator.load_state_dict(generator_state_dict)
    generator.eval()
    generator = generator.cuda()
    generator = nn.DataParallel(generator, device_ids=cfg.GPU_group)
    generator.load_state_dict(generator_state_dict)

    # prepare data
    save_img_dir = os.path.join(cfg.RESULT_DIR, cfg.RESULT_FOLDER,cfg.MODEL_FOLDER,'val'+str(cfg.TRAIN.CKPT))
    print(save_img_dir)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    test_info = json.load(open(os.path.join(cfg.TEST_DIR, cfg.TEST_INFO), 'r', encoding='utf-8'))

    for img_name, sentence_list in test_info.items():
        for i, sentence in enumerate(sentence_list):
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(sentence)
            # print(tokens)
            cap = []
            for word in tokens:
                cap.append(word2index[word])
            # if len(cap) < cfg.TEXT.WORDS_NUM :
            cap_pad = cap + [0]*(cfg.TEXT.WORDS_NUM-len(cap))
            # else:
                # cap_pad = cap[0:cfg.TEXT.WORDS_NUM]
            # print(cap)
            # print(len(cap))
            cap_tensor = torch.LongTensor(cap_pad).unsqueeze(0)
            cap_len = torch.LongTensor([len(cap)])
            # print(cap_len)
            img_path = os.path.join(cfg.TEST_DIR, img_name)
            img = Image.open(img_path).convert('RGB')
            img_rgb_transform = transforms.Compose([
                transforms.Resize((cfg.TRAIN.HEIGHT, cfg.TRAIN.WIDTH))
            ])
            norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img = img_rgb_transform(img)
            img_lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2Lab)
            img_lab = norm(img_lab)
            img_l = img_lab[[0, ], :, :].unsqueeze(0)

            cap_tensor =cap_tensor.cuda()
            cap_len = cap_len.cuda()
            img_l = img_l.cuda()

            with torch.no_grad():
                object_emb,color_emb,obj2color = word_encoder(cap_tensor, cap_len)

                mask = torch.ones((1, cfg.TEXT.WORDS_NUM), dtype=torch.bool)
                for l in range(cap_len[0]):
                    if l < cfg.TEXT.WORDS_NUM:   
                        mask[0][l] = 0

                fake_img_ab= generator(img_l, color_emb, object_emb, obj2color, mask, cap_len)
                save_img_name = img_name[:-4] + '_' + str(i) + img_name[-4:]

                img_labs = torch.cat((img_l, fake_img_ab.detach()), dim=1).cpu()
                img_labs = (img_labs * 0.5 + 0.5) * 255
                img_labs[img_labs > 255] = 255
                img_labs[img_labs < 0] = 0
                img_lab_np = np.array(img_labs[0]).astype(np.uint8)
                img_lab_np = np.transpose(img_lab_np, (1, 2, 0))
                img_rgb_np = cv2.cvtColor(img_lab_np, cv2.COLOR_Lab2RGB)
                img_rgb = Image.fromarray(img_rgb_np)

                save_path = os.path.join(save_img_dir, save_img_name.replace('jpg', 'png'))
                img_rgb.save(save_path)

