import os

import torch
import torch.nn as nn
from torchvision import models

from miscc.config import cfg
from miscc.utils import weights_init
from model.Amodel import  Attention_GATE
from model.Bmodel import conv1x1_relu, conv3x3_bn_relu, conv3x3_tanh
import sys



class GNet(nn.Module):
    def __init__(self, emb_dim):
        super(GNet, self).__init__()
        self.emb_dim = emb_dim

        vgg = models.vgg16_bn()
        self.features = list(vgg.children())[0][:33]

        self.mid1 = conv3x3_bn_relu(512, 512)
        self.attn1 = Attention_GATE(512, emb_dim,cfg.TEXT.WORDS_NUM)
        self.bn1 = nn.BatchNorm2d(512,affine=False)
        
        self.fc_gama1 = nn.Linear(512, 512)
        self.fc_beta1 = nn.Linear(512, 512)
        
        self.mid2 = conv1x1_relu(512, 512)
        self.mid3 = conv3x3_bn_relu(512, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.r1 = nn.ReLU(True)
        self.c1 = conv3x3_bn_relu(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.r2 = nn.ReLU(True)
        self.c2 = conv3x3_bn_relu(128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.r3 = nn.ReLU(True)
        self.c3 = conv3x3_tanh(64, 2)

        self.apply(weights_init)
        self.load_imagenet_weight()

        
    def load_imagenet_weight(self):
        vgg16bn_path = os.path.join(cfg.RESOURCE_DIR, 'vgg16_bn-6c64b313.pth')
        vgg16bn_state_dict = torch.load(vgg16bn_path)
        pretrained_dict = {}
        for i, key in enumerate(vgg16bn_state_dict):
            if i >= 60:
                break
            pretrained_dict[key] = vgg16bn_state_dict[key]
        netG_state_dict = self.state_dict()
        netG_state_dict.update(pretrained_dict)
        self.load_state_dict(netG_state_dict)
        print('Gnet loads weights pretrained on ImageNet')

    def forward(self, img_l, color_emb, object_emb, obj2color, mask):
        """
        :param img_l: batch x 1 x ih x iw
        """
        batch_size = img_l.shape[0]
        x = img_l.repeat(1, 3, 1, 1)
        x = self.features(x)
        # midlevel_features_bn: bs x 512 x 28 x 28
        # c_code : bs x 28*28 x 512
        # gate_mask:  bs x 1 x 28 x 28 
        midlevel_features = self.mid1(x)
        self.attn1.applyMask(mask)
        
        c_code, gate_mask = self.attn1(midlevel_features, color_emb, object_emb, obj2color) # ATM
        
        # bs x 512 x 28 x 28
        midlevel_features_bn = self.bn1(midlevel_features)

        #################### SIM ############################
        gama1 = self.fc_gama1(c_code)
        gama1 = gama1.transpose(1,2).contiguous() # bs x 512 x 28*28
        gama1 = gama1.view(batch_size,512,28,28)  # bs x 512 x 28 x 28
        beta1 = self.fc_beta1(c_code)
        beta1 = beta1.transpose(1,2).contiguous()
        beta1 = beta1.view(batch_size,512,28,28)
        if cfg.TRAIN.USEMASK:    
            beta1_m = beta1 * gate_mask
            gama1_m = gama1 * gate_mask + torch.ones(batch_size,512,28,28).cuda() - gate_mask
        else:
            beta1_m = beta1
            gama1_m = gama1
        
        m_code = midlevel_features_bn * gama1_m + beta1_m
        m_code = m_code + midlevel_features_bn
        ######################################################

        # 512 x 28 x 28
        output = self.mid2(m_code)
        # 512 x 28 x 28
        output = self.mid3(output)
        # 256 x 56 x 56
        output = self.up1(output)
        output = self.r1(output)
        output = self.c1(output)
        # 128 x 112 x 112
        output = self.up2(output)
        output = self.r2(output)
        output = self.c2(output)
        # 2 x 224 x 224
        output = self.up3(output)
        output = self.r3(output)
        output = self.c3(output)

        return output
