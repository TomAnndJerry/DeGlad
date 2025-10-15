import os
import torch.nn.functional as F
import torch
from torch_scatter import scatter

def makeFileDirs(path):
    if not os.path.exists(path):
        oldmask = os.umask(000)
        os.makedirs(path, mode=0o777)
        os.umask(oldmask)

def get_node_id(core_id, env_id):
    if core_id.shape[0] == 0:
        return env_id.clone()
    else:
        return core_id

def get_ce_id(node_imp, edge_imp, data, edge_batch):
    core_node_ids = torch.tensor([]).to(data.x.device).long()
    core_edge_ids = torch.tensor([]).to(data.x.device).long()
    env_node_ids = torch.tensor([]).to(data.x.device).long()
    env_edge_ids = torch.tensor([]).to(data.x.device).long()
    node_means = scatter(node_imp, data.batch, dim=0, reduce='mean')
    edge_means = scatter(edge_imp, edge_batch, dim=0, reduce='mean')

    N_start = 0
    E_start = 0
    unique = torch.unique(data.batch)
    for b in unique:
        node_batch = torch.nonzero(data.batch == b)
        min_id = node_batch.min().item()
        max_id = node_batch.max().item()
        N = max_id - min_id + 1
        edge_batch = torch.nonzero((data.edge_index[0] >= min_id) & (data.edge_index[0] <= max_id), as_tuple=False).view(-1)
        node_mean = node_means[b]
        edge_mean = edge_means[b]
        node_score = node_imp[node_batch].view(-1)
        edge_score = edge_imp[edge_batch].view(-1)

        core_node_id = torch.nonzero(node_score > node_mean)
        env_node_id = torch.nonzero(node_score <= node_mean)
        core_edge_id = torch.nonzero(edge_score > edge_mean)
        env_edge_id = torch.nonzero(edge_score <= edge_mean)

        core_node_id = get_node_id(core_node_id, env_node_id)
        env_node_id = get_node_id(env_node_id, core_node_id)
        core_edge_id = get_node_id(core_edge_id, env_edge_id)
        env_edge_id = get_node_id(env_edge_id, core_edge_id)

        core_node_ids = torch.cat([core_node_ids, N_start + core_node_id], dim=0)
        core_edge_ids = torch.cat([core_edge_ids, E_start + core_edge_id], dim=0)
        env_node_ids = torch.cat([env_node_ids, N_start + env_node_id], dim=0)
        env_edge_ids = torch.cat([env_edge_ids, E_start + env_edge_id], dim=0)
        N_start += N
        E_start += edge_batch.shape[0]
    output = {"core_node_ids": core_node_ids.view(-1), "core_edge_ids": core_edge_ids.view(-1),
              "env_node_ids": env_node_ids.view(-1), "env_edge_ids": env_edge_ids.view(-1)}
    return output

def get_simil_matrix(gy_dim, gyh_dim, pooly_dim, poolyh_dim):
    gy_dim = F.normalize(gy_dim, p=2, dim=-1)
    gyh_dim = F.normalize(gyh_dim, p=2, dim=-1)
    pooly_dim = F.normalize(pooly_dim, p=2, dim=-1)
    poolyh_dim = F.normalize(poolyh_dim, p=2, dim=-1)
    gy_cos_sim = torch.matmul(gy_dim, pooly_dim.T)
    gyh_cos_sim = torch.matmul(gyh_dim, poolyh_dim.T)
    return gy_cos_sim, gyh_cos_sim

def kl_div(node_imp, pos_node_mask, output, batch):
    core_node_ids, env_node_ids = output["core_node_ids"], output["env_node_ids"]
    node_mask = pos_node_mask
    raw_core_dis = node_imp[core_node_ids].view(-1) + 1e-10
    raw_env_dis = node_imp[env_node_ids].view(-1) + 1e-10
    pos_core_dis = node_mask[core_node_ids].view(-1) + 1e-10
    pos_env_dis = node_mask[env_node_ids].view(-1) + 1e-10
    loss_func = torch.nn.KLDivLoss(reduction='none')
    pos_core_kl_loss = loss_func(pos_core_dis.log(), raw_core_dis)
    pos_env_kl_loss = loss_func(pos_env_dis.log(), raw_env_dis)
    pos_core_kl_loss = scatter(pos_core_kl_loss, batch[core_node_ids], dim=0, reduce='mean')
    pos_env_kl_loss = scatter(pos_env_kl_loss, batch[env_node_ids], dim=0, reduce='mean')
    node_imp_mean = scatter(raw_env_dis, batch[env_node_ids], dim=0, reduce='mean')
    env_imp_mean = scatter(pos_env_dis, batch[env_node_ids], dim=0, reduce='mean')
    env_mean_kl_loss = loss_func(env_imp_mean.log(), node_imp_mean)
    return torch.mean(pos_core_kl_loss), torch.mean(pos_env_kl_loss), torch.mean(env_mean_kl_loss)