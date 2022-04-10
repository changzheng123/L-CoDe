from __future__ import print_function

import os


from miscc.utils import save_images
from model.Gmodel import GNet
from model.Emodel import RNN_ENCODER

from miscc.config import cfg, cfg_from_file
from miscc.utils import *
from datasets import TextDataset, prepare_data

import time
import argparse

import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_id) for gpu_id in cfg.GPU_group])

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def train(train_dataloader, word_encoder, generator, optimizer, epoch_iterations, log_file, arc_dict,train_dataset):
    generator.train()
    word_encoder.train()
    loss_fn_l1 = nn.SmoothL1Loss()
    total_time = 0
    start_time = time.time()
    for i, data in enumerate(train_dataloader):
        stime = time.time()
        ######################################################
        # (1) Prepare training data and Compute text embeddings
        ######################################################
        for j in range(len(data) - 1):
            data[j] = data[j].cuda()
        img_l, img_ab, caps, cap_len, img_name = prepare_data(data)
        batch_size = img_l.size(0)
        # build ground-truth OCCM 
        arc_mat_gt = build_arc_mat(arc_dict,img_name,batch_size,caps,train_dataset.ixtoword)
        bce_weight = arc_mat_gt * 5 + 1
        
        object_emb,color_emb,obj2color = word_encoder(caps, cap_len)
        mask = torch.ones((batch_size, cfg.TEXT.WORDS_NUM), dtype=torch.bool)
        for b in range(batch_size):
            for p in range(cap_len[b]):
                mask[b][p] = 0
        #######################################################
        # (2) Generate color images
        ######################################################
        fake_img_ab = generator(img_l, color_emb, object_emb, obj2color, mask)
        #######################################################
        # (3) Update weights
        ######################################################
        generator.zero_grad()
        word_encoder.zero_grad()      
        loss_l1 = loss_fn_l1(img_ab, fake_img_ab)
        loss_arc = nn.BCELoss(bce_weight)(obj2color,arc_mat_gt)

        loss = loss_l1 + 0.1 * loss_arc
        
        loss.backward()
        optimizer.step()
        etime = time.time()
        total_time += etime - stime

        if i % cfg.TRAIN.PRINT_FREQUENCY == 0 and i!=0:
            print(str(epoch_iterations) + '-' + str(i) + ':' + 'err_loss:%.4f time:%d s' % (loss.item(), total_time))
            print(str(epoch_iterations) + '-' + str(i) + ':' + 'err_loss:%.4f time:%d s' % (loss.item(), total_time), file=log_file)
            total_time = 0
        # break

    end_time = time.time()
    print('time:%d s' % (end_time - start_time))
    print('time:%d s' % (end_time - start_time), file=log_file)

def evaluate(val_dataloader, word_encoder, generator, val_dataset, epoch_iterations, test=False):
    generator.eval()
    word_encoder.eval()
    if not test:
        modify = args.name
    else:
        modify = 'test_'+args.name
    save_img_dir = os.path.join(cfg.RESULT_DIR, modify, 'val_' + str(epoch_iterations))
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    with torch.no_grad():
        for data in val_dataloader:
            for j in range(len(data) - 1):
                data[j] = data[j].cuda()
            img_l, img_ab, caps, cap_len, img_name = prepare_data(data)
            batch_size = img_l.size(0)
            object_emb,color_emb,obj2color = word_encoder(caps, cap_len)
            mask = torch.ones((batch_size, cfg.TEXT.WORDS_NUM), dtype=torch.bool)
            for b in range(batch_size):
                for p in range(cap_len[b]):
                    mask[b][p] = 0
            fake_img_ab = generator(img_l, color_emb, object_emb, obj2color, mask)
            save_images(img_l, fake_img_ab, img_name, save_img_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a L-CoDe.')
    parser.add_argument('--name',default='experiment_',
                        type = str,
                        help="experiment name")
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/coco_train.yml', type=str)
    parser.add_argument('--train',
                        action="store_true",
                        help="train or not.")
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate")
    parser.add_argument('--bs', type=int, default=16,
                        help="batch size") 
    parser.add_argument('--epoch', type=int, default=13,
                        help="train epoch")
    parser.add_argument('--gm', action="store_true",
                        help="if use soft-gated mask")
    parser.add_argument('--o2c', action="store_true",
                        help="if use obj2color")
    parser.add_argument('--gmthresh', type=float, default=0.0,
                        help="gate mask thresh")                           
    args = parser.parse_args()
    return args

if __name__ =="__main__":
    args = parse_args()
    cfg_from_file(args.cfg_file)
    cfg.TRAIN.LEARNING_RATE = args.lr
    cfg.TRAIN.BATCH_SIZE = args.bs
    cfg.TRAIN.MAX_EPOCH = args.epoch
    cfg.TRAIN.GTTHRESH = args.gmthresh
    cfg.TRAIN.USEMASK = args.gm
    cfg.TRAIN.OBJ2COL = args.o2c
    args.name = args.name + "bs-%d_lr-%7f_epoch-%d_useMask-%s_thresh-%f_o2c-%s"%(cfg.TRAIN.BATCH_SIZE,\
                                            cfg.TRAIN.LEARNING_RATE, cfg.TRAIN.MAX_EPOCH, str(cfg.TRAIN.USEMASK),cfg.TRAIN.GTTHRESH,str(cfg.TRAIN.OBJ2COL))
    print("\n")
    print(args.name)
    print('LR',cfg.TRAIN.LEARNING_RATE)
    print('BS',cfg.TRAIN.BATCH_SIZE)
    print('UseMask',cfg.TRAIN.USEMASK)
    print('o2c',cfg.TRAIN.OBJ2COL)
    print('thresh',cfg.TRAIN.GTTHRESH)

    model_save_path = os.path.join(cfg.MODEL_DIR, args.name)
    if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)

    log_file = open(os.path.join(cfg.MODEL_DIR, args.name, cfg.LOG_FILE), mode='a')

    train_img_dir = os.path.join(cfg.IMG_DIR, 'train2017') 
    train_caption_path = cfg.RESOURCE_DIR  
    train_transform = transforms.Compose([
        transforms.Resize((cfg.TRAIN.HEIGHT, cfg.TRAIN.WIDTH)),
        transforms.RandomHorizontalFlip()])
    train_dataset = TextDataset(train_img_dir,train_caption_path, train_transform, "train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=cfg.WORKERS)

    val_img_dir = os.path.join(cfg.IMG_DIR, 'val2017')
    val_caption_path = cfg.RESOURCE_DIR
    val_transform = transforms.Compose([
        transforms.Resize((cfg.TRAIN.HEIGHT, cfg.TRAIN.WIDTH))])
    val_dataset = TextDataset(val_img_dir,val_caption_path, val_transform,"val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False, shuffle=False, num_workers=cfg.WORKERS)
    
    word_encoder = RNN_ENCODER(train_dataset.n_words,nhidden=cfg.TEXT.EMBEDDING_DIM)
    word_encoder = word_encoder.cuda()

    generator = GNet(cfg.TEXT.EMBEDDING_DIM)
    generator = generator.cuda()

    arc_dict = json.load(open("./resources/obj2col.json"))

    para = []
    for v in word_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    for v in generator.parameters():
        if v.requires_grad:
            para.append(v)

    if args.train:
        word_encoder = nn.DataParallel(word_encoder, device_ids=cfg.GPU_group)
        generator = nn.DataParallel(generator, device_ids=cfg.GPU_group)     
        for epoch_iterations in range(cfg.TRAIN.MAX_EPOCH + 1):
            if epoch_iterations < 10:
                lr = cfg.TRAIN.LEARNING_RATE
            else:
                lr = cfg.TRAIN.LEARNING_RATE * 0.1
            optimizer = optim.Adam(para, lr=lr, betas=(0.9, 0.999))

            train(train_dataloader, word_encoder, generator, optimizer, epoch_iterations, log_file,arc_dict,train_dataset)

            if epoch_iterations % cfg.TRAIN.SAVE_INTERVAL == 0 or epoch_iterations == cfg.TRAIN.MAX_EPOCH:
                generator_weight_save_path = os.path.join(model_save_path,'generator_' + str(epoch_iterations) + '.pth')
                word_encoder_weight_save_path = os.path.join(model_save_path,'emb_' + str(epoch_iterations) + '.pth')
                
                torch.save(generator.state_dict(), generator_weight_save_path)
                torch.save(word_encoder.state_dict(), word_encoder_weight_save_path)
                print('Save models weight.')
            if epoch_iterations % cfg.TRAIN.EVAL_FREQUENCY == 0 or epoch_iterations == cfg.TRAIN.MAX_EPOCH:
                evaluate(val_dataloader, word_encoder, generator, val_dataset, epoch_iterations)

    