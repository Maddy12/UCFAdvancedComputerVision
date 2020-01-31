import os, re
import glob
import pdb
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch

class TinyImagenet(Dataset):
    """
    Images are (3 x 64 x 64).
    This object, when called, will return the loaded image pixel values and the class ID and intiger label.
    """
    def __init__(self, perc_per_class=100, base_dir='../data/tiny-imagenet-200', split='train', transform=None):
        assert split == 'train' or split == 'val', "Provide 'train' or 'val' as split"
        self.split = split

        # Get ground truth label matching
        df = pd.read_csv(os.path.join(base_dir, 'labels.txt'), delim_whitespace=True, header=None)
        df = df.rename(columns={0: 'id', 1: 'orig_label', 2: 'class'})
        classes = os.listdir(os.path.join(base_dir, 'train'))
        self.labels = df[df['id'].isin(classes)]
        self.labels['label'] = range(0, len(classes))
        # A list of the image paths to load
        per_class = 500 * (perc_per_class/100.0)
        
        if split == 'train':
            self.images = glob.glob(os.path.join(base_dir, split, '*', 'images', '*'))
            image_names = [image.split('/')[-1] for image in self.images]
            image_classes = [image.split('_')[0] for image in image_names]
            train_annotations = pd.DataFrame(data={'image_name': self.images, 'id': image_classes}).sort_values(by='id')
            self.images = train_annotations.groupby(['id']).head(per_class)['image_name'].values
        else:
            self.__val_classes  = pd.read_csv(os.path.join(base_dir, split, 'val_annotations.txt'), header=None, delimiter='\t')
            self.__val_classes = self.__val_classes .rename(columns={1: 'id', 0:'image_name'})
            self.images = self.__val_classes.groupby(['id']).head(per_class)['image_name'].values
            self.images = [os.path.join(base_dir, split, 'images', im) for im in self.images]

        # Transformations for images
        if transform is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        if self.split == 'train':
            # Extract the class ID from the image path
            class_label = re.findall('n\d+(?!\/)', image_path)[-1]
        else:
            # Match the image path to the assigned class ID from the val_annotations.txt dataframe
            image_name =  image_path.split('/')[-1]
            class_label = self.__val_classes[self.__val_classes['image_name'] == image_name]['id'].values[0]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Use class ID to get label information
        labels = self.labels[self.labels['id'] == class_label].to_dict(orient='records')[0]
        return image, labels


class SVHN(Dataset):
    """
    Images are (3 x 64 x 64).
    This object, when called, will return the loaded image pixel values and the class ID and intiger label.
    """
    def __init__(self, perc_per_class=100, base_dir='../data/svhn', split='train', transform=None):
        # Load data
        path = os.path.join(base_dir, split+'_32x32.mat')
        data = loadmat(path)
        self.images = data['X'].reshape(data['X'].shape[-1], data['X'].shape[2], data['X'].shape[0], data['X'].shape[1])
        self.targets = data['y']
        del data

        # Transformations for images
        if transform is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image
        image = self.images[index]
        image = Image.frombytes('RGB', (32, 32), image)
        image = self.transform(image)

        # Use class ID to get label information
        label = self.targets[index][0]
        if label == 10:
            label = 0
        labels = {'label': label}
        return image, labels


def test_functionality_imagenet():
    test = TinyImagenet(split='train', base_dir='/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/data/tiny-imagenet-200')
    print(len(test))
    test_loader = DataLoader(test, shuffle=False, batch_size=128) 
    for inputs, targets in test_loader:
        print(inputs)
        print(targets)
        break
    return inputs, targets


def test_functionality_svhn():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    resize = (64, 64)
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), normalize]) 
    test = SVHN(split='train', transform=transform, base_dir='/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/data/svhn')
    print(len(test))
    test_loader = DataLoader(test, shuffle=False, batch_size=1) 
    for inputs, targets in test_loader:
        print(inputs)
        print(targets)
        break
    return inputs, targets

if __name__ == '__main__':
    test_functionality_svhn()