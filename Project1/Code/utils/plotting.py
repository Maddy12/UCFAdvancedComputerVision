import matplotlib.pyplot as pyplot
import os, pdb
from multiprocessing import cpu_count
from torch import nn, cuda, device, load
from torch.utils.data import DataLoader
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt

# Local
import sys
from utils.datasets import TinyImagenet, SVHN
from utils.model import MyModel, AlexNetFine
from main import run_model


class Results(object):
    def __init__(self, models_dir, data_path='../data'):
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
       
        self.models = os.listdir(models_dir)
        self.models_dir = models_dir
        self.convolution_experiments = ['finetune_imagenet_alexnet', 'convs3_perc100', 'convs4_perc100', 'convs5_perc100']
        self.datasize_experiments = ['convs3_perc25', 'convs3_perc50', 'convs3_perc75', 'convs3_perc100']
        self.finetune_experiments = ['finetune_svhn_alexnet', 'finetune_imagenet_alexnet']
        
        self.criterion = nn.CrossEntropyLoss()

        self.datasets = dict()
        resize = (64, 64)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), normalize]) 
        data_paths = ['tiny-imagenet-200', 'svhn']
        for idx, data_obj in enumerate([TinyImagenet, SVHN]):
            val_dataset = data_obj(perc_per_class=100, base_dir=os.path.join(data_path, data_paths[idx]), split='val', transform=transform)
            dataloader = dict()
            dataloader['testing'] = DataLoader(dataset=val_dataset, batch_size=128, num_workers=cpu_count(), shuffle=True)
            dataloader['testing_length'] = len(val_dataset)
            self.datasets[data_obj.__name__] = dataloader
        
        self.results = dict()
        self._get_validation_results()
        pdb.set_trace()
    def _get_validation_results(self):
        for model_name in self.models:
            if model_name.startswith('convs'):
                convs = int(model_name.split('_')[0][-1])
                model = MyModel(convs)
                num_classes = 200
                dataname = 'TinyImagenet'
                if '_'.join(model_name.split('_')[:2]) in self.convolution_experiments:
                    experiment = 1
                else:
                    experiment = 2
            else:
                if 'imagenet' in model_name:
                    num_classes = 200
                    dataname = 'TinyImagenet'
                else: 
                    num_classes = 10
                    dataname = 'SVHN'
                experiment = 3
                model = AlexNetFine(num_classes)
            print("Running evaluation for " + str(model_name))
            model.load_state_dict(load(os.path.join(self.models_dir, model_name, 'model_best.pth.tar'))['state_dict'])
            model = model.to(self.device).eval()
            dataloader = self.datasets[dataname]
            avg_test_loss, avg_test_top1_prec = run_model(0, model, self.criterion, None, dataloader['testing'], dataloader['testing_length'], 
                                                                train=False, device=self.device, num_classes=num_classes)     
            self.results[model_name] = {'experiment': experiment, 'Average Loss': avg_test_loss, 'Average Accuracy': avg_test_top1_prec}

def plot(results, experiment, outdir, query):
    """
    {'convs5_perc100_2020131_checkpoints': {'experiment': 1, 'avg_test_loss': 5.298324664306641, 'avg_test_top1_prec': 0.0049}, 'finetune_svhn_alexnet_2020131_checkpoints': {'experiment': 3, 'avg_test_loss': nan, 'avg_test_top1_prec': 0.06699446836496763}, 'convs3_perc50_2020131_checkpoints': {'experiment': 2, 'avg_test_loss': 5.299220902252197, 'avg_test_top1_prec': 0.0052}, 'finetune_imagenet_alexnet_202023_checkpoints': {'experiment': 3, 'avg_test_loss': 4.892291538238525, 'avg_test_top1_prec': 0.0259}, 'convs3_perc75_2020131_checkpoints': {'experiment': 2, 'avg_test_loss': 5.310231043243408, 'avg_test_top1_prec': 0.0046}, 'convs4_perc100_2020131_checkpoints': {'experiment': 1, 'avg_test_loss': 5.299722052764893, 'avg_test_top1_prec': 0.005}, 'convs3_perc100_2020131_checkpoints': {'experiment': 1, 'avg_test_loss': 5.294576371765137, 'avg_test_top1_prec': 0.0045}, 'convs3_perc25_2020131_checkpoints': {'experiment': 2, 'avg_test_loss': 5.298298305511475, 'avg_test_top1_prec': 0.0045}}
    Args: 
        experiment (int): which experiment to plot. Options 1, 2, 3
    """
    y = [vals[query] for key, vals in results.items() if vals['experiment'] == experiment]
    x = [key for key, vals in results.items() if vals['experiment'] == experiment]
    ax = sns.barplot(x, y)
    plt.title(query+" for experiment "+str(experiment))
    if 'Loss' in query:
        ax.set_ylim([0,10])
    plt.ylabel(query)
    # plt.xticks(rotation=15)
    plt.savefig(os.path.join(outdir, 'experiment'+str(experiment)+'_'+'_'.join(query.split(' '))+'.jpg'))
    plt.close()

if __name__=='__main__':
    results = Results('/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/models')
    outdir = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/plots'
    # Experiment 1
    results.plot(1, outdir, loss=False)
    results.plot(1, outdir, loss=True)

    # Experiment 2
    results.plot(2, outdir, loss=False)
    results.plot(2, outdir, loss=True)

    # Experiment 3
    results.plot(3, outdir, loss=False)
    results.plot(3, outdir, loss=True)