import os
import sys
import numpy as np
import numbers
import random
import time
from tqdm import tqdm
import pdb
import argparse
import math
import copy
from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Local
from utils.dataset import UCF101, detection_collate
from utils.video_transform import *
from model.i3d import InceptionI3d
# from utils.augmentations import SSDAugmentation
from utils.CSDSSD.layers.modules import MultiBoxLoss
# from utils.CSDSSD.ssd import build_ssd
from model.videoCSD import build_ssd_con
from utils.config import ucf101 as cfg

ROOT = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/data'
LOCAL_ROOT = ROOT
SAVE_MODEL = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/model/weights/i3d_UCF101/weights_'


# from data import *


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def train(root, save_folder, i3d_weights, vgg_weights='weights/vgg16_reducedfc.pth',  batch_size=2, visdom=False, resume=None):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # setup dataset
    # train_transforms = transforms.Compose([RandomCrop(224),
    #                                        RandomHorizontalFlip(),
    # ])
    # test_transforms = transforms.Compose([CenterCrop(224)])

    if visdom:
        import visdom
        viz = visdom.Visdom()


    # Load model CSD
    ssd_net = build_ssd_con('train', vgg_weights, i3d_weights, cfg['min_dim'], cfg['num_classes'], resume).cuda()
    # ssd_net = orig_build_ssd_con('train', cfg['min_dim']).cuda()
    net = ssd_net
    # net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

    # Set up losses and optimizers
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=5e-4, weight_decay=0.1)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, True)
    conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()
    
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0

    # start iter == 0
    supervised_flag = 1

    print('Loading the dataset...')
    step_index = 0

    if visdom:
        vis_title = 'SSD.PyTorch on UCF101'
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    total_un_iter_num = 0
    supervised_batch =  batch_size

    supervised_dataset = UCF101(train=False, root=root, annot_dir=root, video_transform=None, bounding_boxes=True, semi=True)    
    supervised_data_loader = data.DataLoader(supervised_dataset, supervised_batch,
                                                num_workers=cpu_count(),
                                                shuffle=True, collate_fn=detection_collate,
                                                pin_memory=False, drop_last=True)
    batch_iterator = iter(supervised_data_loader)

    net.train()
    pbar = tqdm(range(0, cfg['max_iter']))
    pbar.set_description('Running')
    for iteration in pbar:
        if visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, 0.1, step_index)

        try:
            images, targets, semis = next(batch_iterator)
        except StopIteration:
            supervised_flag = 0
            supervised_dataset = UCF101(train=False, root=root, annot_dir=root, video_transform=None, bounding_boxes=True, semi=True, min_dim=cfg['min_dim'])    
            supervised_data_loader = data.DataLoader(supervised_dataset, supervised_batch,
                                                        num_workers=cpu_count(),
                                                        shuffle=True, collate_fn=detection_collate,
                                                        pin_memory=True, drop_last=True)
            batch_iterator = iter(supervised_data_loader)
            images, targets, semis = next(batch_iterator)

        try:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        except:
            import pdb; pdb.set_trace()
            print("FAILED")
            continue
        
        
        # forward
        t0 = time.time()

        out, conf, conf_flip, loc, loc_flip = net(images)
        images = None

        sup_image_binary_index = np.zeros([len(semis),1])

        for super_image in range(len(semis)):
            # pdb.set_trace()
            if(int(semis[super_image])==1):
                sup_image_binary_index[super_image] = 1
            else:
                sup_image_binary_index[super_image] = 0

            if(int(semis[len(semis)-1-super_image])==0):
                del targets[len(semis)-1-super_image]

        sup_image_index = np.where(sup_image_binary_index == 1)[0]
        unsup_image_index = np.where(sup_image_binary_index == 0)[0]

        loc_data, conf_data, priors = out

        if (len(sup_image_index) != 0):
            loc_data = loc_data[sup_image_index,:,:]
            conf_data = conf_data[sup_image_index,:,:]
            output = (
                loc_data,
                conf_data,
                priors
            )

        # backprop
        loss = Variable(torch.cuda.FloatTensor([0]))
        loss_l = Variable(torch.cuda.FloatTensor([0]))
        loss_c = Variable(torch.cuda.FloatTensor([0]))

        if(len(sup_image_index)!=0):
            try:
                loss_l, loss_c = criterion(output, targets)
            except:
                print("FAILED")
                import pdb; pdb.set_trace()
                outputs = None
                image = None
                continue
            

        sampling = True
        # sampling = False
        if(sampling is True):
            conf_class = conf[:,:,1:].clone()
            background_score = conf[:, :, 0].clone()
            each_val, each_index = torch.max(conf_class, dim=2)
            mask_val = each_val > background_score
            mask_val = mask_val.data

            mask_conf_index = mask_val.unsqueeze(2).expand_as(conf)
            mask_loc_index = mask_val.unsqueeze(2).expand_as(loc)

            conf_mask_sample = conf.clone()
            loc_mask_sample = loc.clone()

            conf_sampled = conf_mask_sample[mask_conf_index].view(-1, 24)
            loc_sampled = loc_mask_sample[mask_loc_index].view(-1, 4)

            conf_mask_sample_flip = conf_flip.clone()
            loc_mask_sample_flip = loc_flip.clone()
            conf_sampled_flip = conf_mask_sample_flip[mask_conf_index].view(-1, 24)
            loc_sampled_flip = loc_mask_sample_flip[mask_loc_index].view(-1, 4)

        if(mask_val.sum()>0):
            ## JSD !!!!!1
            conf_sampled_flip = conf_sampled_flip + 1e-7
            conf_sampled = conf_sampled + 1e-7
            consistency_conf_loss_a = conf_consistency_criterion(conf_sampled.log(), conf_sampled_flip.detach()).sum(-1).mean()
            consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(), conf_sampled.detach()).sum(-1).mean()
            consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b

            ## LOC LOSS
            consistency_loc_loss_x = torch.mean(torch.pow(loc_sampled[:, 0] + loc_sampled_flip[:, 0], exponent=2))
            consistency_loc_loss_y = torch.mean(torch.pow(loc_sampled[:, 1] - loc_sampled_flip[:, 1], exponent=2))
            consistency_loc_loss_w = torch.mean(torch.pow(loc_sampled[:, 2] - loc_sampled_flip[:, 2], exponent=2))
            consistency_loc_loss_h = torch.mean(torch.pow(loc_sampled[:, 3] - loc_sampled_flip[:, 3], exponent=2))

            consistency_loc_loss = torch.div(
                consistency_loc_loss_x + consistency_loc_loss_y + consistency_loc_loss_w + consistency_loc_loss_h,
                4)

        else:
            consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))

        consistency_loss = torch.div(consistency_conf_loss,2) + consistency_loc_loss

        ramp_weight = rampweight(iteration)
        consistency_loss = torch.mul(consistency_loss, ramp_weight)


        if(supervised_flag ==1):
            loss = loss_l + loss_c + consistency_loss
        else:
            if(len(sup_image_index)==0):
                loss = consistency_loss
            else:
                loss = loss_l + loss_c + consistency_loss


        if(loss.data>0):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t1 = time.time()
        if(len(sup_image_index)==0):
            loss_l.data = Variable(torch.cuda.FloatTensor([0]))
            loss_c.data = Variable(torch.cuda.FloatTensor([0]))
        else:
            loc_loss += loss_l.data  # [0]
            conf_loss += loss_c.data  # [0]


        # if iteration % 10 == 0:
        #     print('timer: %.4f sec.' % (t1 - t0))
        #     print('iter ' + repr(iteration) + ' || Loss: %.4f || consistency_loss : %.4f ||' % (loss.data, consistency_loss.data), end=' ')
        #     print('loss: %.4f , loss_c: %.4f , loss_l: %.4f , loss_con: %.4f, lr : %.4f, super_len : %d\n' % (loss.data, loss_c.data, loss_l.data, consistency_loss.data,float(optimizer.param_groups[0]['lr']),len(sup_image_index)))


        # if(float(loss)>100):
        #     break

        if visdom:
            update_vis_plot(iteration, loss_l.data, loss_c.data,
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and (iteration+1) % 100 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), os.path.join(save_folder, 'ssd300_UCF101_' +
                        repr(iteration+1) + '.pth'))

        # Update Progress Bar
        postfix = '({batch}/{size}) | Total: {total:} | Epoch: {epoch:} | Loc Loss: {loc:.4f} | Cof Loss: {cof:.4f}'.format(
            batch=iteration + 1,
            total=(iteration+1) * batch_size,
            size=len(supervised_dataset),
            epoch=epoch,
            loc=loc_loss, 
            cof=conf_loss
        )
        pbar.set_postfix_str(postfix)

        

def rampweight(iteration):
    ramp_up_end = 32000
    ramp_down_start = 100000

    if(iteration<ramp_up_end):
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end),2))
    elif(iteration>ramp_down_start):
        ramp_weight = math.exp(-12.5 * math.pow((1 - (120000 - iteration) / 20000),2)) 
    else:
        ramp_weight = 1 


    if(iteration==0):
        ramp_weight = 0

    return ramp_weight




def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = 1e-3 * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def xavier(param):
#     init.xavier_uniform(param)


# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         xavier(m.weight.data)
#         m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


def train_i3d(init_lr=0.1, epochs=500, mode='rgb', root=ROOT, train=True, batch_size=8*5, pretrained_pth='models/weights/rgb_imagenet.pt', save_model=SAVE_MODEL):

    # setup dataset
    train_transforms = transforms.Compose([RandomCrop(224),
                                           RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([CenterCrop(224)])

    dataset = UCF101(train=True, root=root, annot_dir=root, video_transform=train_transforms, bounding_boxes=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    val_dataset = UCF101(train=False, root=root, annot_dir=root, video_transform=test_transforms, bounding_boxes=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    i3d = InceptionI3d(num_classes=400, in_channels=3)
    i3d.load_state_dict(torch.load(pretrained_pth))
    
    i3d.replace_logits(24)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 4 # accum gradient
    steps = 0
    for epoch in range(epochs):

        # Each epoch has a training and validation phase        
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            pbar = tqdm(dataloaders[phase])
            pbar.set_description(phase)
            for batch_idx, (inputs, labels, vid) in enumerate(pbar):
                num_iter += 1

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())
                pdb.set_trace() 
                try:
                    per_frame_logits = i3d(inputs.float())
                    # upsample to input size
                    # per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
                    per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear')
                    # compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                    tot_loc_loss += loc_loss.data  # [0]

                    # compute classification loss (with max-pooling along time B x C x T)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                    tot_cls_loss += cls_loss.data  # [0]
                except Exception as e:
                    print("ERROR WITH {}: {}".format(vid, e))

                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.data  # [0]
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()

                    # Update Progress Bar
                    postfix = '({batch}/{size}) | Total: {total:} | Epoch: {epoch:} | Grad Steps: {step: } Loc Loss: {loc:.4f} | Cls Loss: {cls:.4f} | Tot Loss: {tot:.4f}'.format(
                        batch=batch_idx + 1,
                        total=(batch_idx+1) * inputs.shape[0],
                        size=len(dataloader),
                        epoch=epoch,
                        step=steps,
                        loc=tot_loc_loss/(10*num_steps_per_update), 
                        cls=tot_cls_loss/(10*num_steps_per_update),
                        tot=tot_loss/10
                    )
                    pbar.set_postfix_str(postfix)

                    if steps % 10 == 0:
                        # print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, , , ))
                        # save model
                        torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                # print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))
                # Update Progress Bar
                postfix = '({batch}/{size}) | Total: {total:} | Epoch: {epoch:} | Grad Steps: {step: } Loc Loss: {loc:.4f} | Cls Loss: {cls:.4f} | Tot Loss: {tot:.4f}'.format(
                    batch=batch_idx + 1,
                    total=(batch_idx+1) * inputs.shape[0],
                    size=len(dataloader),
                    epoch=epoch,
                    step=steps,
                    loc=tot_loc_loss/(10*num_steps_per_update), 
                    cls=tot_cls_loss/(10*num_steps_per_update),
                    tot=tot_loss/10
                )
                pbar.set_postfix_str(postfix)
    

def test_dataset():
    test_transforms = transforms.Compose([CenterCrop(224)])

    dataset = UCF101(train=True, video_transform=test_transforms, bounding_boxes=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=32, pin_memory=True)    
    test_inputs = list()
    test_targets = list()
    # try:
    for idx, (inputs, targets) in enumerate(tqdm(dataset)):
        continue

def get_sample_data(root=ROOT, annot_dir= os.path.join(ROOT, 'annotations')):
    test_transforms = transforms.Compose([CenterCrop(224)])
    train_transforms = transforms.Compose([RandomCrop(224),
                                           RandomHorizontalFlip(),
    ])
    dataset = UCF101(train=True, video_transform=train_transforms, bounding_boxes=True, root=root, annot_dir=annot_dir, semi=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=32, pin_memory=True)    
    for idx, (inputs, targets, vid) in enumerate(dataset):
        break
    return inputs, targets


if __name__ == '__main__':
    # need to add argparse
    root = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/data'
    vgg_weights = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/utils/CSDSSD/weights/vgg16_reducedfc.pth'
    i3d_weights = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/model/weights/i3d_UCF101/weights_005710.pt'
    save_folder = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/model/weights'
    train(root, save_folder, i3d_weights, vgg_weights=vgg_weights,  batch_size=1, visdom=False)