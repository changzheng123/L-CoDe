from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Text

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import cv2
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
from PIL import Image
import numpy.random as np_random
import json
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
def prepare_data(data):
    img_l, img_ab, captions, captions_lens, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    img_l = img_l[sorted_cap_indices]
    img_ab = img_ab[sorted_cap_indices]

    captions = captions[sorted_cap_indices].squeeze()
    keys = [keys[i] for i in sorted_cap_indices]

    return [img_l, img_ab, captions, sorted_cap_lens, keys]

class TextDataset(data.Dataset):
    def __init__(self, img_dir, caption_dir, transform, split='train'):
        self.img_dir = img_dir
        self.caption_path_train = os.path.join(caption_dir,cfg.TRAIN_CAPTION)
        self.caption_path_val = os.path.join(caption_dir,cfg.VAL_CAPTION)
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.filenames, self.captions, self.ixtoword, self.wordtoix, \
            self.n_words= \
            self.load_text_data(cfg.RESOURCE_DIR,split)
    def load_text_data(self, data_dir, split, captions_file='captions.pickle'):
        filepath = os.path.join(data_dir, captions_file)
        if not os.path.isfile(filepath):
            train_json = json.load(open(self.caption_path_train))
            val_json = json.load(open(self.caption_path_val))
            train_names = list(train_json.keys())
            val_names = list(val_json.keys())
            
            train_captions = self.load_captions(train_json,train_names)
            val_captions = self.load_captions(val_json,val_names)
           
            train_captions, val_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, val_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, val_captions,
                             ixtoword, wordtoix,train_names,val_names], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, val_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                train_names,val_names = x[4],x[5]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = val_captions
            filenames = val_names
            
        print('len(filenames)',len(filenames))
        return filenames, captions, ixtoword, wordtoix, n_words
    
    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_captions(self,json_dict,filenames):
        all_captions = []
        for i in range(len(filenames)):
            captions = json_dict[filenames[i]]
            cnt = 0
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                # print('tokens', tokens)
                if len(tokens) == 0:
                    print('cap', cap)
                    continue

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                all_captions.append(tokens_new)
                cnt += 1
                if cnt == self.embeddings_num: 
                    break
            if cnt < self.embeddings_num:
                print('ERROR: the captions for %s less than %d'
                        % (filenames[i], cnt))
        return all_captions

    def get_img(self, img_name):
        img_pth = os.path.join(self.img_dir, img_name)
        img = Image.open(img_pth).convert('RGB')
        img = self.transform(img)
        img_lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2Lab)
        img_lab = self.norm(img_lab)
        img_l = img_lab[0, :, :].unsqueeze(0)
        img_ab = img_lab[1:, :, :]
        return img_l, img_ab

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        img_l, img_ab = self.get_img(key)
        caps, cap_len = self.get_caption(index)

        return img_l, img_ab, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)