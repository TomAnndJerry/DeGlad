import torch
from torch import nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
class GINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_adv=None):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_weight=edge_adv))
        return out

    def message(self, x_j, edge_weight=None):
        if edge_weight is not None:
            mess = F.relu(x_j * edge_weight)
        else:
            mess = F.relu(x_j)
        return mess

    def update(self, aggr_out):
        return aggr_out

class CGA_model(torch.nn.Module):
    def __init__(self, in_dim, args, dropout_rate=0.5):
        super(CGA_model, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.noise_alpha = args.noise_alpha
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(self.num_layer - 1)])
        self.batch_norm1 = nn.BatchNorm1d(self.emb_dim)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.emb_dim) for _ in range(self.num_layer - 1)])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(self.num_layer - 1)])
        self.conv1 = GINConv(in_dim, self.emb_dim)
        self.convs = nn.ModuleList([GINConv(self.emb_dim, self.emb_dim) for _ in range(self.num_layer - 1)])
        self.node_att_mlp = nn.Linear(self.emb_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.conv1)
        reset(self.convs)
        reset(self.node_att_mlp)

    def get_score(self, x, edge_index, node_adv, edge_adv, batch):
        x = x * node_adv
        node_rep = self.batch_norm1(self.conv1(x, edge_index, edge_adv=edge_adv))
        if self.num_layer > 1:
            noise = torch.randn_like(node_rep) * self.noise_alpha
            node_rep = self.relu1(node_rep + noise)
            node_rep = self.dropout1(node_rep)
        for i, (conv, batch_norm, relu, dropout) in enumerate(zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            node_rep = batch_norm(conv(node_rep, edge_index, edge_adv))
            if i != len(self.convs) - 1:
                node_rep = relu(node_rep)
            node_rep = dropout(node_rep)
        node_key = torch.sigmoid(self.node_att_mlp(node_rep))
        edge_key = self.lift_node_score_to_edge_score(node_key, edge_index)
        return node_key, edge_key

    def forward(self, data, node_imp, edge_imp):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_adv = node_imp
        edge_adv = edge_imp
        node_key, edge_key = self.get_score(x, edge_index, node_adv, edge_adv, batch)
        output = {"aug_node_score": node_key, "aug_edge_score": edge_key}
        return output

    def lift_node_score_to_edge_score(self, node_score, edge_index):
        src_lifted_att = node_score[edge_index[0]]
        dst_lifted_att = node_score[edge_index[1]]
        edge_score = src_lifted_att * dst_lifted_att
        return edge_score