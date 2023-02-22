import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, feature, hidden, layer):
        super(GCN, self).__init__()
        self.layer = layer
        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(GCNConv(feature, hidden)) 
        for _ in range(self.layer - 1):
            self.gnn_layers.append(GCNConv(hidden, hidden)) 


    def forward(self, x, edge_index):
        for i in range(self.layer):
            x = self.gnn_layers[i](x, edge_index.coalesce().indices())
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        return x

class GAT(torch.nn.Module):
    def __init__(self, feature, hidden, layer=1, heads=1):
        super(GAT,self).__init__()
        self.gat1 = GATConv(feature, hidden, heads=heads)
        self.gat2 = GATConv(hidden*heads, hidden)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index.coalesce().indices())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index.coalesce().indices())
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden, layer=1):
        super(GraphSAGE, self).__init__()
        self.layer = layer
        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(SAGEConv(feature, hidden))
        for _ in range(self.layer - 1):
            self.gnn_layers.append(SAGEConv(hidden, hidden)) 
        

    def forward(self, x, edge_index):
        for i in range(self.layer):
            x = self.gnn_layers[i](x, edge_index.coalesce().indices())
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)
