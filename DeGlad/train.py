import os

import torch
import numpy as np
import torch_geometric
from sklearn.metrics import roc_auc_score

from arguments import arg_parse
from get_data_loaders_tuad import get_ad_split_TU, get_data_loaders_TU
import random
import warnings

from model.cgc_models import CGA_model
from model.models import MSIB, MSIB_DHT
from utils.tools import makeFileDirs, kl_div, get_ce_id, get_simil_matrix

warnings.filterwarnings("ignore")
# torch.autograd.set_detect_anomaly(True)
logger = None


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch_geometric.seed_everything(seed)


def eval_model(model, test_loader, device):
    model.eval()
    all_ad_true = []
    all_ad_score = []
    for data in test_loader:
        all_ad_true.append(data.y.cpu())
        data = data.to(device)
        with torch.no_grad():
            raw_y, raw_y_hyper, _, _ = model(data)
            ano_score = model.loss_nce(raw_y, raw_y_hyper)
        all_ad_score.append(ano_score.cpu())
    ad_true = torch.cat(all_ad_true)
    ad_score = torch.cat(all_ad_score)
    ad_auc = roc_auc_score(ad_true, ad_score)
    info_test = '\t\t[test]:\t【auc {:.4f}】'.format(ad_auc)
    print(info_test)
    return ad_auc


def run(args, seed, split=None):
    print("=" * 10 + "[ " + args.dataset + " ](" + str(seed) + ")" + "=" * 10)
    set_seed(seed)
    loaders, meta = get_data_loaders_TU(args, split)
    n_feat = meta['num_feat']
    n_edge_feat = meta['num_edge_feat']

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    msib_model = MSIB(n_feat, n_edge_feat, args, device).to(device)
    cga_model = CGA_model(n_feat, args).to(device)
    optimizer = torch.optim.Adam(msib_model.parameters(), lr=args.lr)
    cga_optimizer = torch.optim.Adam(cga_model.parameters(), lr=args.cga_net_lr, weight_decay=5e-6)

    train_loader = loaders['train']
    test_loader = loaders['test']
    train_anomalyNum, train_nomalyNum = 0, 0
    test_anomalyNum, test_nomalyNum = 0, 0
    for data in train_loader:
        train_anomalyNum += sum(data.y == 1)
        train_nomalyNum += sum(data.y == 0)
    for data in test_loader:
        test_anomalyNum += sum(data.y == 1)
        test_nomalyNum += sum(data.y == 0)
    print("="*100)
    print("Node attribute dimension:{}".format(n_feat))
    print("Edge attribute dimension:{}".format(n_edge_feat))
    print("The number of abnormal cases in the training set is:{},and the number of normal cases is:{}".format(train_anomalyNum, train_nomalyNum))
    print("The number of abnormal cases in the test set is:{},and the number of normal cases is:{}".format(test_anomalyNum, test_nomalyNum))
    print("Training set: Ratio of abnormal cases to normal cases:{:.2f}/{:.2f}".format(train_anomalyNum / len(train_loader.dataset),
                                                               train_nomalyNum / len(train_loader.dataset)))
    print("Test set: Ratio of abnormal cases to normal cases:{:.2f}/{:.2f}".format(test_anomalyNum / len(test_loader.dataset),
                                                               test_nomalyNum / len(test_loader.dataset)))
    print("=" * 100)

    best_score = -1
    min_loss = -1
    for epoch in range(args.epochs):
        msib_model.train()
        cga_model.train()
        loss_all = 0
        dis_loss_all = 0
        core_kl_all = 0
        env_kl_all = 0
        env_mean_kl_all = 0
        num_sample = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            raw_y, raw_y_hyper, _, _ = msib_model(data)
            loss = msib_model.loss_nce(raw_y, raw_y_hyper).mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                node_imp = msib_model.extractor(data.x, data.edge_index, data.batch)
                edge_imp = msib_model.lift_node_score_to_edge_score(node_imp, data.edge_index)
                _, edge_batch = MSIB_DHT(data.edge_index, data.batch)
                output = get_ce_id(node_imp, edge_imp, data, edge_batch)

            with torch.no_grad():
                raw_y, raw_y_hyper, _, _ = msib_model.get_out_dim(data, node_imp, edge_imp)
                gy_simil_matrix, gyh_simil_matrix = get_simil_matrix(raw_y, raw_y_hyper, raw_y, raw_y_hyper)
            optimizer.zero_grad()
            _, _, y_node, y_hyper_node = msib_model(data)
            cgc_y, cgc_y_hyper = msib_model.get_mix_out_dim(y_node, y_hyper_node, gy_simil_matrix, gyh_simil_matrix,
                                                            output, data.batch, edge_batch, args)
            loss = msib_model.loss_nce(cgc_y, cgc_y_hyper).mean()
            loss.backward()
            optimizer.step()

            cga_optimizer.zero_grad()
            res = cga_model(data, node_imp, edge_imp)
            aug_node_score = res["aug_node_score"]
            core_kl_node_loss, env_kl_node_loss, env_mean_kl_node_loss = kl_div(node_imp, aug_node_score, output,
                                                                                data.batch)
            dis_loss = core_kl_node_loss * args.core_kl_alpha + env_mean_kl_node_loss * args.env_mean_kl_alpha - env_kl_node_loss * args.env_kl_alpha
            cga_loss = dis_loss
            cga_loss.backward()
            cga_optimizer.step()
            core_kl_all = (core_kl_node_loss).item()
            env_kl_all = (env_kl_node_loss).item()
            env_mean_kl_all = (env_mean_kl_node_loss).item()
            dis_loss_all = dis_loss.item()

            optimizer.zero_grad()
            res = cga_model(data, node_imp, edge_imp)
            aug_node_score, aug_edge_score = res["aug_node_score"], res["aug_edge_score"]
            cgc_y, cgc_y_hyper, _, _ = msib_model.get_out_dim(data, aug_node_score.detach(), aug_edge_score.detach())
            loss = msib_model.loss_nce(cgc_y, cgc_y_hyper).mean()
            loss.backward()
            optimizer.step()

            loss_all += loss.item() * data.num_graphs
            num_sample += data.num_graphs
        loss_all = loss_all / num_sample
        info_train = 'Ep{:1d}_{:1d}: 【loss {:.4f}】【dis_loss_all {:.4f}】【core_kl_all {:.4f}】【env_kl_all {:.4f}】【env_mean_kl_all {:.4f}】【best_s {:.4f}】'.format(
            epoch, seed, loss_all, dis_loss_all, core_kl_all, env_kl_all, env_mean_kl_all, best_score)
        print(info_train)

        if min_loss == -1 or loss_all <= min_loss:
            min_loss = loss_all
            best_score = eval_model(msib_model, test_loader, device)
        if epoch % args.log_interval == 0:
            eval_model(msib_model, test_loader, device)
        if epoch % args.lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr_decay_factor * param_group['lr']
    return best_score

if __name__ == '__main__':
    datasets = ["COX2", "DHFR", "BZR", "AIDS", "PROTEINS_full", "ENZYMES", "DD", "NCI1", "IMDB-BINARY", "REDDIT-BINARY"]
    args = arg_parse()
    args.device = 0
    if args.threshold is not None:
        args.threshold = [float(num) for num in args.threshold.split(',')]
    args.num_trials = 5
    print(args)

    ad_aucs = []
    splits = get_ad_split_TU(args, fold=5)
    for trial in range(args.num_trials):
        results = run(args, seed=trial, split=splits[trial])
        ad_auc = results
        ad_aucs.append(ad_auc)
    results = 'AUC: {:.2f}±{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print('[FINAL RESULTS] ' + results)
