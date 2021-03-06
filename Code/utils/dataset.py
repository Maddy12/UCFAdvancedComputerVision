import os
import sys
import pickle
import glob
import pandas as pd
import numpy as np
import cv2
import pdb
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from utils.video_transform import *
from utils.CSDSSD.utils import augmentations
from pandas.errors import ParserWarning
import warnings
warnings.filterwarnings('ignore', category=ParserWarning)

# PATH = '/home/cap6412.student5/UCF-101'
# ROOT = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/data'
# ANNOTATION_DIR = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/data/annotations'
ROOT = '/home/cap6412.student24/UCFAdvancedComputerVision/Code/data'
ANNOTATION_DIR = '/home/cap6412.student24/UCFAdvancedComputerVision/Code/data/annotations'

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
MEANS = (104, 117, 123)


class UCF101(Dataset):
    """
    The annotation pickle file is a dictionary in which the keys are the `action/video_name`. 
    Each object has the following keys: annotations, numf, and label. 
    The annotations are a list of dictionaries. Each element has the following keys: label, ef, boxes, sf
        * label: the class of the action at this frame
        * ef: the end frame of the action
        * boxes: The bounding box of the action
        * sf: the start frame of the action

   
    """
    def __init__(self, root=ROOT, min_dim=240, means=MEANS, annot_dir=ANNOTATION_DIR, video_transform=None, bounding_boxes=True, train=True, semi=False):
        self.root_dir = os.path.join(root, 'train') if train else os.path.join(root, 'test')
        # annot_dir = os.path.join(root, 'annotations')

        # The labels are offset by 1 where 0 is background and then the list 
        self.action_classes = ['VolleyballSpiking', 'SoccerJuggling', 'SalsaSpin', 'RopeClimbing', 'FloorGymnastics', 'Surfing', 
                            'Fencing', 'Skiing', 'WalkingWithDog', 'HorseRiding', 'LongJump', 'TrampolineJumping', 'Skijet', 
                            'IceDancing', 'SkateBoarding', 'CliffDiving', 'Biking', 'PoleVault', 'Basketball', 'CricketBowling', 
                            'Diving', 'GolfSwing', 'BasketballDunk', 'TennisSwing']
        self.action_classes.sort()
        # self.action_classes.insert(0, 'Background')

        self.semi = semi
        self.num_classes = 24
        self.n_frames_per_segment = 3
        self.n_frames_per_video = 64
        self.bounding_boxes = bounding_boxes
        self.n_frames_per_segment = 1
        self.n_frames_per_video = 64

        self.annot = pickle.load(open(os.path.join(annot_dir, 'pyannot.pkl'), 'rb'))
        self.videos  = ['/'.join(pth.split('.')[-2].split('/')[-2:]) for pth in glob.glob(self.root_dir + '/*/*') if pth.endswith('.avi')]
        self.vids = [vid for vid in list(self.annot.keys()) if vid in self.videos]

        # Unnecessary but adding for visual
        if semi:
            self.unlabeled  = [vid for vid in self.videos if vid not in list(self.annot.keys())]
            self.vids = np.concatenate([self.vids, self.unlabeled])
            random.shuffle(self.vids)
        else:
            self.unlabeled = list()
        # self.data = self.__get_labels()

        # Transforms, the first is associated with the paper and the other is general I3D transforms
        self.transform = augmentations.SSDAugmentation(min_dim, means)
        self.video_transform = video_transform

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        """
        Returns frames which are (240, 320, 3).
        """
        # vid, num_frames, sf, ef, sampled_frame_idxs, labels = self.data[index]
        vid = self.vids[index]
        if vid in self.unlabeled:
            unlabeled = True
            semi = 0
        else:
            unlabeled = False
            semi = 1

        # Filter for 64 frames from video at default 30 fps
        pth = os.path.join(self.root_dir, vid + '.avi')
        video = cv2.VideoCapture(pth)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        v_frames = np.array([i for i in range(num_frames)])
        sampled_frame_idxs = _uniform_sample_frames_per_video_for_i3d(v_frames, self.n_frames_per_segment, self.n_frames_per_video)[0]
        video.release()

        # Load frames using indexes
        frames = load_rgb_frames(pth, sampled_frame_idxs)
        if self.video_transform is not None:
            frames = self.video_transform(frames)

        # Get labels and transform frames 
        labels, frames = self.__get_labels(vid, frames, sampled_frame_idxs, unlabeled)
        labels = np.array(labels)
        if video_to_tensor(np.array(frames)).float().shape[1] != 64: 
            pdb.set_trace()
        return video_to_tensor(np.array(frames)).float(), torch.from_numpy(labels).float(), semi
        # return torch.from_numpy(np.array(frames)).permute(2, 0, 1).float(), torch.from_numpy(labels).float(), semi

    def __get_labels(self, vid, frames, sampled_frame_idxs, unlabeled):
        """
        Returns the following:
            * the video ID
            * a label with dimensions (Num classes, Num frames, Bounding Boxes)
            * number of frames for the given video
        """
        # Create labeling for each frame, want (num_frames, [xmin, ymin, xmax, ymax, label_idx])
        if self.bounding_boxes:
            label = np.zeros((64, 5))  
        else: 
            # Otherwise create a binary classification for each possible class
            label = np.zeros((self.num_classes, self.n_frames_per_video), np.float32)
        new_frames = list()
        # Iterate through the annotations
        if not unlabeled:
            for ann in self.annot[vid]['annotations']:
                sf = ann['sf']
                ef = ann['ef']
                for idx, fr in enumerate(sampled_frame_idxs):
                    
                    # We only want an action label when an action is present, othewise it is just Background
                    if fr - sf < 0 or fr >= ef or np.sum(ann['boxes'][fr-sf]) == 0:
                        continue
                    else:
                        if self.bounding_boxes:
                            # box = np.concatenate([ann['boxes'][fr-sf], [ann['label']+1]])
                            img, boxes, labels = self.transform(frames[idx], np.expand_dims(ann['boxes'][fr-sf], axis=0).astype(np.float32), np.array([[ann['label']+1]]))
                            # label[idx] = np.concatenate([boxes[0], labels[0]])
                            label[idx] = np.concatenate([ann['boxes'][len(ann['boxes'])//2], labels[0]])
                            new_frames.append(img)
                        else:
                            label[ann['label']+1, idx] = 1
                            new_frames.append(frames[idx])
        else:
            for idx, fr in enumerate(sampled_frame_idxs):
                if self.bounding_boxes:
                    # box = np.concatenate([ann['boxes'][fr-sf], [ann['label']+1]])
                    img, boxes, labels = self.transform(frames[idx], np.expand_dims([0.,0.,0.,0.], axis=0).astype(np.float32), np.array([[0.]]))
                    label[idx] = np.concatenate([boxes[0], labels[0]])
                    new_frames.append(img)
        return label, new_frames


def _uniform_sample_frames_per_video_for_i3d(v_frames, n_frames_per_segment, n_frames_per_video):
    sampled_frame_pathes = []

    # i3d model accepts sequence of 8 frames
    n_frames = len(v_frames)
    n_segments = int(n_frames_per_video / n_frames_per_segment)

    if n_frames < n_frames_per_video:
        # padding = int(np.ceil((n_segments - n_frames) / 2))
        padding = int(n_segments - n_frames)
        idx = np.arange(len(v_frames))
        for i in range(padding):
            if i % 2 == 0:
                idx = np.insert(idx, 0, 0, axis=0)
            else:
                idx = np.insert(idx, -1, idx[-1], axis=0)
    elif n_frames == n_frames_per_video:
        idx = np.arange(n_frames_per_video)
    else:
        step = n_frames / float(n_segments)
        idces_start = np.arange(0, n_frames, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + n_frames_per_segment, dtype=np.int).tolist()

    v_sampled = v_frames[idx]
    sampled_frame_pathes.append(v_sampled)
    if len(sampled_frame_pathes[0]) < 64:
        print(sampled_frame_pathes[0])
    return sampled_frame_pathes


def load_rgb_frames(pth, sampled_frame_idxs):
    """
    Args:
        pth (str): Path to avi file to load.
    Returns:
        frames (list): List of all the frames in the video.
    """
    frames = []
    cap = cv2.VideoCapture(pth)
    ret = True
    i = 0
    while ret:
        ret, img = cap.read()
        if ret:
            while i in sampled_frame_idxs:
                w, h, c = img.shape
                if w < 226 or h < 226:
                    d = 226. - min(w, h)
                    sc = 1 + d / min(w, h)
                    img = cv2.resize(img, dsize=(0,0), fx=sc, fy=sc)
                img = (img/255.) * 2 - 1
                frames.append(img)
                sampled_frame_idxs = np.delete(sampled_frame_idxs, 0)
            i += 1
    # special case, too tired to fix, prepending until reach 64
    if len(frames) < 64:
        needed = 64 - len(frames)
        cap = cv2.VideoCapture(pth)
        for i in range(needed):
            ret, img = cap.read()
            frames = np.concatenate([[img], frames])
    # return np.asarray(frames)
    return frames


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    ### changed when semi-supervised
    targets = []
    imgs = []
    semis = []
    for sample in batch:
        imgs.append(sample[0])
        try:
            targets.append(torch.FloatTensor(sample[1]))
        except:
            print(sample[2])
            exit()
        if(len(sample)==3):
            # semis.append(torch.FloatTensor(sample[2]))
            semis.append(sample[2])
    if(len(sample)==2):
        return torch.stack(imgs, 0), targets
    else:
        return torch.stack(imgs, 0), targets, semis
    # return torch.stack(imgs, 0), targets


def video_to_tensor(pic):
    """
    From https://github.com/piergiaj/pytorch-i3d/blob/master/charades_dataset.py
    
    Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    try:
        return torch.from_numpy(pic.transpose([3,0,1,2]))
    except:
        pdb.set_trace()


def test(root=ROOT, annot_dir=ANNOTATION_DIR):
    train_transforms = transforms.Compose([RandomCrop(224),
                                           RandomHorizontalFlip(),
    ])
    root = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/data'
    annot_dir = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/Code/data/annotations'
    dataset = UCF101(root=root, annot_dir=annot_dir, video_transform=None, semi=True, bounding_boxes=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=False)
    for inputs, targets, sims in tqdm(dataset):
        if inputs.shape[1] != 64:
            pdb.set_trace()
        continue


if __name__ == '__main__':
    test()
