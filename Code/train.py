import os
import sys
import numpy as np
import numbers
import random
from tqdm import tqdm
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
# Local
from utils.dataset import UCF101
from utils.video_transform import *
from model.i3d import InceptionI3d


ROOT = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/data'
SAVE_MODEL = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/model/weights/i3d_UCF101/weights_'

def run(init_lr=0.1, epochs=500, mode='rgb', root=ROOT, train=True, batch_size=8*5, save_model=SAVE_MODEL):

    # setup dataset
    train_transforms = transforms.Compose([RandomCrop(224),
                                           RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([CenterCrop(224)])

    dataset = UCF101(train=True, video_transform=train_transforms, bounding_boxes=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    val_dataset = UCF101(train=False, video_transform=test_transforms, bounding_boxes=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    i3d = InceptionI3d(num_classes=400, in_channels=3)
    i3d.load_state_dict(torch.load('model/weights/rgb_imagenet.pt'))
    
    i3d.replace_logits(25)
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
            for batch_idx, (inputs, labels) in enumerate(pbar):
                num_iter += 1

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs.float())
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data  # [0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data  # [0]
            
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
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        continue
    # except Exception as e:
    #     error = e
    #     import pdb; pdb.set_trace()
        
if __name__ == '__main__':
    # need to add argparse
    run(batch_size=1, epochs=10)
    # test_dataset()