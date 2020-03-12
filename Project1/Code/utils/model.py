import pdb
# Local
from utils.layers import BasicConv2d as conv_block
from utils import datasets

import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead
from sklearn.metrics.pairwise import euclidean_distances as eucd  # '0.19.1'
from torch_geometric.nn import GCNConv, TopKPooling,  EdgePooling, global_mean_pool
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class MyModel(nn.Module):
    def __init__(self, num_convs=3, fine_tune=False):
        super(MyModel, self).__init__()
        assert 8 >= num_convs > 1, "Cannot have less than 1 or greater than 8 convolutional+pooling layers."
        self.num_convs = num_convs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Simple Segmentation Model
        self.backbone = models.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.backbone = IntermediateLayerGetter(self.backbone, return_layers={'layer4': 'out'})
        if not fine_tune:
            for p in self.backbone.parameters():
                p.requires_grad = False
        in_channels = 1028
        classes = 200  # actually classes

        # Graph convolution layers and graph pooling layers
        self.classifier = FCNHead(in_channels, classes)
        self.convolutions = dict()
        for conv in range(num_convs):
            self.convolutions['conv'+str(conv+1)] = GCNConv(2048, 2048).to(self.device)
            self.convolutions['pool'+str(conv+1)] = TopKPooling(2048, ratio=.6).to(self.device)

        #Final Output
        self.lin1 = torch.nn.Linear(2048, 1024).to(self.device)
        self.lin2 = torch.nn.Linear(1024, 1024).to(self.device)
        self.lin3 = torch.nn.Linear(1024, classes).to(self.device)
        self.act1 = torch.nn.ReLU().to(self.device)
        self.act2 = torch.nn.ReLU().to(self.device)

    def forward(self, x):
        """
        x represents node features.
        """
        # [N, 3, 64, 64]
        x = self.backbone(x)["out"]  
        # [N, 2048, 8, 8]
        x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1])     
        # [N, 8, 8, 2048]
        x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        # [N, 64, 2048]
        batches = list()
        for batch, node_features in enumerate(x):
            node_features = x[batch]
            _x = self._per_batch(node_features).unsqueeze(0)
            if batch == 0:
                batches = _x
            else:
                batches = torch.cat([batches, _x])
        x = batches.to(self.device)
        # x = self.classifier(x)
        # (N, 1, 2048)
        x = self.lin1(x)
        x = self.act1(x)
        # (N, 1, 1024)
        x = self.lin2(x)
        x = self.act2(x)
        # (N, 1, 1024)
        x = F.dropout(x, p=.5, training=self.training)
        x = torch.sigmoid(self.lin3(x)).squeeze(1)
        # (N, 200)
        return x

    def _per_batch(self, node_features):
        """
        Performs graph convolutions and topK poolling after each iteration. 
        The numbers of iteration are determined by the num_convs parameter passed as a hyperparameter.
        After the iterations are complete, a global_mean_pool is taken of the node features. 
        """
        edge_index = self._generate_edges(node_features)
        for idx in range(self.num_convs):
            # print(idx+1)
            conv = self.convolutions['conv'+str(int(idx+1))]
            pool = self.convolutions['pool'+str(int(idx+1))]
            node_features = F.relu(conv(node_features, edge_index))
            node_features, edge_index,  _, batch, _, _ = pool(node_features, edge_index)
        node_features = global_mean_pool(node_features, batch)
        # Node Features (1, 2048)
        return node_features

    def _generate_edges(self, x):
        """
        Returns an array of (Node Pair, Number of relationships) so (2, N)
        Args: 
            x (torch.tensor): Node features of shape (N, 64, 2048)
        """
        x = x.cpu().detach().numpy()
        distances = eucd(x)
        # Get one half of the symmetric distance matrix and turn the other half into nans
        distances = np.tril(distances)
        distances[distances == 0.] = np.nan 
        # Normalize distance metrics
        distances = (distances - np.nanmin(distances, axis=0)) / (np.nanmax(distances, axis=0) - np.nanmin(distances, axis=0))
        # mean_distance = np.nanmean(distances) 
        edges = np.argwhere(distances < np.nanmean(distances))  # (E, 2)
        edges = torch.tensor(edges.reshape(edges.shape[1], edges.shape[0]))  # (2, E)
        edges = edges.to(self.device)
        return edges

class AlexNetFine(nn.Module):
    def __init__(self, classes=10):
        super(AlexNetFine, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.alexnet = IntermediateLayerGetter(alexnet, return_layers={'avgpool':'out'})
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.alexnet(x)['out']
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def test_functionality_imagenet():
    inputs, targets = datasets.test_functionality_imagenet()
    model = MyModel()
    node_features, edge_list = model(inputs)
    in_channels = node_features.shape[-1]
    out_channels = 200  # number of classes
    model = MyModel()
    model(inputs)
    return node_features, edge_list


def test_functionality_svhn():
    inputs, targets = datasets.test_functionality_svhn()
    model = AlexNetFine()
    x = model(inputs)
    return 

if __name__ == '__main__':
    test_functionality_svhn()