import pdb
# Local
from layers import BasicConv2d as conv_block
from layers import SAGEConv
import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead
from sklearn.metrics.pairwise import euclidean_distances as eucd  # '0.19.1'
from torch_geometric.nn import GCNConv, TopKPooling,  EdgePooling, global_mean_pool
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, fine_tune=False):
        super(MyModel, self).__init__(num_convs=3)
        assert 8 >= num_convs > 1, "Cannot have less than 1 or greater than 8 convolutional+pooling layers."
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
        convolutions = dict()
        for conv in range(num_convs):
            convolutions['conv'+str(conv+1)] = GCNConv(2048, 2048)
            convolutions['pool'+str(conv+1)] = TopKPooling(2048, ratio=.6)
        # self.gconv1 = GCNConv(2048, 2048)
        # self.pool1 = TopKPooling(2048, ratio=.6)
        # self.gconv2 = GCNConv(2048, 2048)
        # self.pool2 = TopKPooling(2048, ratio=.6)
        # self.gconv3 = GCNConv(2048, 2048)
        # self.pool3 = TopKPooling(2048, ratio=.6)
        # self.gconv4 = GCNConv(2048, 2048)

        # Final Output
        self.lin1 = torch.nn.Linear(2048, 1024)
        self.lin2 = torch.nn.Linear(1024, 1024)
        self.lin3 = torch.nn.Linear(1024, classes)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

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
        edge_list = self._generate_edges(x)
        # [2, R]
        batches = list()
        for batch, (node_features, edge_index) in enumerate(zip(x, edge_list)):
            node_features = x[batch]
            edge_index = edge_list[batch]
            x = self._per_batch(node_features, edge_index).unsqueeze(0)
            if batch == 0:
                batches = x
            else:
                batches = torch.cat([batches, x])
        x = batches
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

    def _per_batch(self, node_features, edge_index):
        # Node Features (64, 2048)
        node_features = F.relu(self.gconv1(node_features, edge_index))
        # Node Features (64, 2048)
        node_features, edge_index, _, _, _, _ = self.pool1(node_features, edge_index)
        # Node Features (39, 2048) Edge Index (2, Pooled # of relationships)
        node_features = F.relu(self.gconv2(node_features, edge_index))
        # Node Features (39, 2048)
        node_features, edge_index, _, _, _, _ = self.pool2(node_features, edge_index)
        # Node Features (24, 2048) Edge Index (2, Pooled # of relationships)
        node_features = F.relu(self.gconv3(node_features, edge_index))
        # Node Features (25, 2048)
        node_features, edge_index,  _, batch, _, _ = self.pool3(node_features, edge_index)
        # Node Features (15, 2048) Edge Index (2, Pooled # of relationships)
        node_features = F.relu(self.gconv4(node_features, edge_index))
        node_features = global_mean_pool(node_features, batch)
        # Node Features (1, 2048)
        return node_features

    @staticmethod
    def _generate_edges(x):
        """
        Returns an array of (Node Pair, Number of relationships) so (2, N)
        Args: 
            x (torch.tensor): Node features of shape (N, 64, 2048)
        """
        x = x.detach().numpy()
        edge_list = list()
        for idx, batch in enumerate(range(x.shape[0])):
            distances = eucd(x[batch])
            # Get one half of the symmetric distance matrix and turn the other half into nans
            distances = np.tril(distances)
            distances[distances == 0.] = np.nan 
            # Normalize distance metrics
            distances = (distances - np.nanmin(distances, axis=0)) / (np.nanmax(distances, axis=0) - np.nanmin(distances, axis=0))
            # mean_distance = np.nanmean(distances) 
            edges = np.argwhere(distances < np.nanmean(distances))  # (E, 2)
            edges = torch.tensor(edges.reshape(edges.shape[1], edges.shape[0]))  # (2, E)
            edges = edges.unsqueeze(0)
            if idx ==  0:
                edge_list = edges
            else:
                edge_list = torch.cat([edge_list, edges])  # torch.cat appends to the end of the list
        return edge_list

def test_functionality():
    import datasets
    inputs, targets = datasets.test_functionality()
    model = MyModel()
    node_features, edge_list = model(inputs)
    in_channels = node_features.shape[-1]
    out_channels = 200  # number of classes
    layer = layers.SAGEConv(in_channels=in_channels, out_channels=out_channels)
    return node_features, edge_list