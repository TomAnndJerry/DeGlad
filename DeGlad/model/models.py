import torch
import torch.nn.functional as F
import torch_scatter
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, HypergraphConv, global_add_pool, global_max_pool, global_mean_pool
import torch.nn as nn
from torch_scatter import scatter

scalar = 20
eps = 1e-10

class MSIB(torch.nn.Module):
    def __init__(self, input_dim, input_dim_edge, args, device):
        super(MSIB, self).__init__()
        self.device = device
        self.embedding_dim = args.hidden_dim
        if args.readout == 'concat':
            self.embedding_dim *= args.encoder_layers

        if args.extractor_model == 'mlp':
            self.extractor = MSIB_Extractor_MLP(input_dim, args.extractor_hidden_dim, args.extractor_layers)
        else:
            self.extractor = MSIB_Extractor_GIN(input_dim, args.extractor_hidden_dim, args.extractor_layers, args.extractor_readout)

        self.encoder = MSIB_GIN(input_dim, args.hidden_dim, args.encoder_layers, args.pooling, args.readout)
        self.encoder_hyper = MSIB_HyperGNN(input_dim, input_dim_edge, args.hidden_dim, args.encoder_layers, args.pooling, args.readout)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_hyper = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                             nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def forward(self, data):
        node_imp = self.extractor(data.x, data.edge_index, data.batch)
        edge_imp = self.lift_node_score_to_edge_score(node_imp, data.edge_index)
        y, y_node = self.encoder(data.x, data.edge_index, data.batch, node_imp)
        y_hyper, y_hyper_node = self.encoder_hyper(data.x, data.edge_index, data.edge_attr, data.batch, edge_imp)
        raw_y = self.proj_head(y)
        raw_y_hyper = self.proj_head_hyper(y_hyper)
        return raw_y, raw_y_hyper, y_node, y_hyper_node

    def get_mix_out_dim(self, y_node, y_hyper_node, gy_simil_matrix, gyh_simil_matrix, output, batch, edge_batch, args):
        cgc_ys = torch.tensor([]).to(self.device)
        cgc_y_hypers = torch.tensor([]).to(self.device)
        core_node_ids, core_edge_ids, env_node_ids, env_edge_ids = output["core_node_ids"], output["core_edge_ids"], output["env_node_ids"], output["env_edge_ids"]
        core_node_batch = batch[core_node_ids]
        env_node_batch = batch[env_node_ids]
        core_edge_batch = edge_batch[core_edge_ids]
        env_edge_batch = edge_batch[env_edge_ids]
        core_node_embedding = scatter(y_node[core_node_ids], core_node_batch, dim=0, reduce='sum')
        env_node_embedding = scatter(y_node[env_node_ids], env_node_batch, dim=0, reduce='sum')
        core_edge_embedding = scatter(y_hyper_node[core_edge_ids], core_edge_batch, dim=0, reduce='sum')
        env_edge_embedding = scatter(y_hyper_node[env_edge_ids], env_edge_batch, dim=0, reduce='sum')

        simil_matrix = gy_simil_matrix * 0.5 + gyh_simil_matrix * 0.5
        _, indices = torch.sort(simil_matrix, dim=1, descending=True)
        top_0 = indices[:, 0]
        sim_top_k_id = indices[:, 1:]
        alpha = args.threshold[0]
        lambda_ratio = (torch.rand(top_0.shape[0], 1) * (args.threshold[1] - alpha) + alpha).to(self.device)
        cgc_g_y = core_node_embedding[top_0] + (env_node_embedding[top_0] * lambda_ratio + env_node_embedding[sim_top_k_id[:, 0]] * (1 - lambda_ratio))
        cgc_g_yh = core_edge_embedding[top_0] + (env_edge_embedding[top_0] * lambda_ratio + env_edge_embedding[sim_top_k_id[:, 0]] * (1 - lambda_ratio))
        cgc_ys = torch.cat((cgc_ys, cgc_g_y), dim=0)
        cgc_y_hypers = torch.cat((cgc_y_hypers, cgc_g_yh), dim=0)
        cgc_ys = self.proj_head(cgc_ys)
        cgc_y_hypers = self.proj_head_hyper(cgc_y_hypers)
        return cgc_ys, cgc_y_hypers

    def get_code_dim(self, data, node_imp, edge_imp, output, batch, edge_batch):
        _, y_node = self.encoder(data.x, data.edge_index, data.batch, node_imp)
        _, y_hyper_node = self.encoder_hyper(data.x, data.edge_index, data.edge_attr, data.batch, edge_imp)
        core_node_ids, core_edge_ids = output["core_node_ids"], output["core_edge_ids"]
        core_node_batch = batch[core_node_ids]
        core_edge_batch = edge_batch[core_edge_ids]
        core_node_embedding = scatter(y_node[core_node_ids], core_node_batch, dim=0, reduce='sum')
        core_edge_embedding = scatter(y_hyper_node[core_edge_ids], core_edge_batch, dim=0, reduce='sum')
        raw_y = self.proj_head(core_node_embedding)
        raw_y_hyper = self.proj_head_hyper(core_edge_embedding)
        return raw_y, raw_y_hyper, y_node, y_hyper_node

    def get_out_dim(self, data, node_imp, edge_imp):
        y, y_node = self.encoder(data.x, data.edge_index, data.batch, node_imp)
        y_hyper, y_hyper_node = self.encoder_hyper(data.x, data.edge_index, data.edge_attr, data.batch, edge_imp)
        raw_y = self.proj_head(y)
        raw_y_hyper = self.proj_head_hyper(y_hyper)
        return raw_y, raw_y_hyper, y_node, y_hyper_node

    def lift_node_score_to_edge_score(self, node_score, edge_index):
        src_lifted_att = node_score[edge_index[0]]
        dst_lifted_att = node_score[edge_index[1]]
        edge_score = src_lifted_att * dst_lifted_att
        return edge_score

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    @staticmethod
    def loss_nce(x1, x2, temperature=0.2):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-10)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-10)

        loss_0 = - torch.log(loss_0 + 1e-10)
        loss_1 = - torch.log(loss_1 + 1e-10)
        loss = (loss_0 + loss_1) / 2.0
        return loss

def MSIB_DHT(edge_index, batch, add_loops=True):
    num_edge = edge_index.size(1)
    device = edge_index.device

    edge_to_node_index = torch.arange(0, num_edge, 1, device=device).repeat_interleave(2).view(1, -1)
    hyperedge_index = edge_index.T.reshape(1, -1)
    hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long()

    edge_batch = hyperedge_index[1, :].reshape(-1, 2)[:, 0]
    edge_batch = torch.index_select(batch, 0, edge_batch)

    if add_loops:
        bincount = hyperedge_index[1].bincount()
        mask = bincount[hyperedge_index[1]] != 1
        max_edge = hyperedge_index[1].max()
        loops = torch.cat([torch.arange(0, num_edge, 1, device=device).view(1, -1),
                           torch.arange(max_edge + 1, max_edge + num_edge + 1, 1, device=device).view(1, -1)],
                          dim=0)

        hyperedge_index = torch.cat([hyperedge_index[:, mask], loops], dim=1)

    return hyperedge_index, edge_batch


class MSIB_Extractor_MLP(torch.nn.Module):
    def __init__(self, num_features, dim, n_layers):
        super(MSIB_Extractor_MLP, self).__init__()

        self.n_layers = n_layers
        self.mlps = torch.nn.ModuleList()
        self.sigmoid = torch.nn.Sigmoid()

        for i in range(n_layers):
            if i:
                nn = Sequential(Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim))
            self.mlps.append(nn)
        self.final_mlp = Linear(dim, 1)

    def forward(self, x, edge_index, batch):

        for i in range(self.n_layers):
            x = self.mlps[i](x)
            x = F.relu(x)
        node_prob = self.final_mlp(x)
        node_prob = self.sigmoid(node_prob)
        return node_prob

class MSIB_Extractor_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, readout):
        super(MSIB_Extractor_GIN, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.readout = readout
        self.convs = torch.nn.ModuleList()
        self.sigmoid = torch.nn.Sigmoid()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            self.convs.append(conv)

        if self.readout == 'concat':
            self.mlp = Linear(dim * num_gc_layers, 1)
        else:
            self.mlp = Linear(dim, 1)

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            if i != self.num_gc_layers - 1:
                x = self.convs[i](x, edge_index)
                x = F.relu(x)
            else:
                x = self.convs[i](x, edge_index)
            xs.append(x)

        if self.readout == 'last':
            node_prob = xs[-1]
        elif self.readout == 'concat':
            node_prob = torch.cat([x for x in xs], 1)
        elif self.readout == 'add':
            node_prob = 0
            for x in xs:
                node_prob += x
        node_prob = self.mlp(node_prob)
        res = self.sigmoid(node_prob)
        return res

class MSIB_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, pooling, readout):
        super(MSIB_GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.pooling = pooling
        self.readout = readout

        self.convs = torch.nn.ModuleList()
        self.dim = dim
        self.pool = self.get_pool()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)

            self.convs.append(conv)

    def forward(self, x, edge_index, batch, node_imp):

        if node_imp is not None:
            out, _ = torch_scatter.scatter_max(torch.reshape(node_imp.detach(), (1, -1)), batch)
            out = out.reshape(-1, 1)
            out = out[batch]
            node_imp = node_imp / (out + eps)
            node_imp = (2 * node_imp - 1)/(2 * scalar) + 1
            x = x * node_imp

        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], batch)
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, batch) for x in xs], 1)
        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, batch)

        return graph_emb, torch.cat(xs, 1)

    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        elif self.pooling == 'mean':
            pool = global_mean_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))
        return pool


class MSIB_HyperGNN(torch.nn.Module):

    def __init__(self, input_dim, input_dim_edge, hidden_dim, num_gc_layers, pooling, readout):
        super(MSIB_HyperGNN, self).__init__()

        self.num_node_features = input_dim
        if input_dim_edge:
            self.num_edge_features = input_dim_edge
            self.use_edge_attr = True
        else:
            self.num_edge_features = input_dim
            self.use_edge_attr = False
        self.nhid = hidden_dim
        self.enhid = hidden_dim
        self.num_convs = num_gc_layers
        self.pooling = pooling
        self.readout = readout
        self.convs = self.get_convs()
        self.pool = self.get_pool()


    def forward(self, x, edge_index, edge_attr, batch, edge_imp):
        if not self.use_edge_attr:
            a_, b_ = x[edge_index[0]], x[edge_index[1]]
            edge_attr = (a_ + b_) / 2

        hyperedge_index, edge_batch = MSIB_DHT(edge_index, batch)

        if edge_imp is not None:
            out, _ = torch_scatter.scatter_max(torch.reshape(edge_imp, (1, -1)), edge_batch)
            out = out.reshape(-1, 1)
            out = out[edge_batch]
            edge_imp = edge_imp / (out + eps)
            edge_imp = (2 * edge_imp - 1)/(2 * scalar) + 1
            edge_attr = edge_attr * edge_imp

        xs = []

        for _ in range(self.num_convs):
            edge_attr = F.relu(self.convs[_](edge_attr, hyperedge_index))
            xs.append(edge_attr)

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], edge_batch)
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, edge_batch) for x in xs], 1)
        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, edge_batch)

        return graph_emb, torch.cat(xs, 1)

    def get_convs(self):
        convs = torch.nn.ModuleList()
        for i in range(self.num_convs):
            if i == 0:
                conv = HypergraphConv(self.num_edge_features, self.nhid)
            else:
                conv = HypergraphConv(self.nhid, self.nhid)
            convs.append(conv)

        return convs

    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        elif self.pooling == 'mean':
            pool = global_mean_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))

        return pool
