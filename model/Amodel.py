import torch
import torch.nn as nn
import sys
from model.Bmodel import conv1x1,conv3x3
from miscc.config import cfg


class Attention_GATE(nn.Module):
    """
    Module ATM: modified from from AttnGAN
    """
    def __init__(self, channel_dim, emb_dim, word_num = None):
        super(Attention_GATE, self).__init__()
        self.conv_context_obj = conv1x1(emb_dim, channel_dim, False)
        self.conv_context_col = conv1x1(emb_dim, channel_dim, False)
        self.sm = nn.Softmax(dim=-1)
        self.mask = None
        self.conv_mask = conv1x1(word_num,1, True)
        
        torch.nn.init.xavier_uniform_(self.conv_mask.weight)
        torch.nn.init.constant_(self.conv_mask.bias,0)

    def applyMask(self, mask):
        self.mask = mask  # batch x word_num

    def forward(self, input, color_emb, object_emb, obj2color):
        """
            input: batch x channel_dim x ih x iw (pixel_num=ihxiw)
            color_emb: batch x emb_dim x word_num
            object_emb: batch x emb_dim x word_num
        """
        ih, iw = input.size(2), input.size(3)
        pixel_num = ih * iw
        batch_size, word_num = object_emb.size(0), object_emb.size(2)

        # --> batch x pixel_num x channel_dim
        target = input.view(batch_size, -1, pixel_num)
        targetT = torch.transpose(target, 1, 2).contiguous()

        # batch x emb_dim x word_num --> batch x emb_dim x word_num x 1
        objectT = object_emb.unsqueeze(3)
        # --> batch x channel_dim x word_num
        # objectT = self.conv_context_obj(objectT).squeeze(3)
        objectT = self.conv_context_obj(objectT).squeeze(3)

        colorT = color_emb.unsqueeze(3)
        # colorT = self.conv_context_col(colorT).squeeze(3)
        # colorT = self.conv_context(colorT).squeeze(3)
        colorT = self.conv_context_col(colorT).squeeze(3)

        # Get attention
        # (batch x pixel_num x channel_dim)(batch x channel_dim x word_num)
        # -->batch x pixel_num x word_num
        attn = torch.bmm(targetT, objectT)

        # obj2color: obj x color
        if cfg.TRAIN.OBJ2COL: 
            attn = torch.bmm(attn, obj2color)

        # --> batch*pixel_num x word_num
        attn = attn.view(batch_size*pixel_num, word_num)
        # mid = attn.clone()
        if self.mask is not None:
            # batch_size x word_num --> batch_size*pixel_num x word_num
            # mask = self.mask.repeat(pixel_num, 1)
            mask = torch.repeat_interleave(self.mask, pixel_num, dim=0).cuda()
            attn.data.masked_fill_(mask.data, -float('inf'))
        # print('attn',attn)
        attn_word = self.sm(attn)
        # --> batch x pixel_num x word_num
        attn_word = attn_word.view(batch_size, pixel_num, word_num)
        
        # --> batch x word_num x pixel_num
        attn_word = torch.transpose(attn_word, 1, 2).contiguous()
        # print('sm_attn_word',attn_word)

        # (batch x channel_dim x word_num)(batch x word_num x pixel_num)
        # --> batch x channel_dim x pixel_num
        # --> out:batch x pixel_dim x channel_num
        weightedContext = torch.bmm(colorT, attn_word)
        weightedContext = torch.transpose(weightedContext,1,2).contiguous()
        
        attn_region = attn_word.view(batch_size*word_num,pixel_num).detach()
        attn_region = self.sm(attn_region)
        attn_region = attn_region.view(batch_size, word_num, ih, iw)
        attn_region = attn_region*28*28 - 1
        gate_mask = self.conv_mask(attn_region)
        gate_mask = torch.sigmoid(gate_mask)
 

        return weightedContext,  gate_mask

