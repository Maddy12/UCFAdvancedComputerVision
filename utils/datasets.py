import os, re
import glob
import pdb
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class TinyImagenet(Dataset):
    """
    Images are (3 x 64 x 64).
    This object, when called, will return the loaded image pixel values and the class ID and intiger label.
    """
    def __init__(self, base_dir='../data/tiny-imagenet-200', split='train', transform=None):
        assert split == 'train' or split == 'val', "Provide 'train' or 'val' as split"
        self.split = split

        # Get ground truth label matching
        df = pd.read_csv(os.path.join(base_dir, 'labels.txt'), delim_whitespace=True, header=None)
        df = df.rename(columns={0: 'id', 1: 'label', 2: 'class'})

        # A list of the image paths to load
        if split == 'train':
            classes = os.listdir(os.path.join(base_dir, split))
            self.labels = df[df['id'].isin(classes)]
            self.images = glob.glob(os.path.join(base_dir, split, '*', 'images', '*'))
        else:
            classes = pd.read_csv(os.path.join(base_dir, split, 'val_annotations.txt'), header=None, delimiter='\t')
            self.__val_classes = classes.rename(columns={1: 'id', 0:'image_name'})
            classes = df['id'].values
            self.labels = df[df['id'].isin(classes)]
            self.images = glob.glob(os.path.join(base_dir, split, 'images', '*'))

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


def test_functionality():
    test = TinyImagenet(split='train')
    test_loader = DataLoader(test, shuffle=False, batch_size=1) 
    for inputs, targets in test_loader:
        print(inputs)
        print(targets)
        break
    return inputs, targets

if __name__ == '__main__':
    test_functionality()