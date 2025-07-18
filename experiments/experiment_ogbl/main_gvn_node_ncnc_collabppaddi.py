import sys
sys.path.append("..") 
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from baseline_models.NCN.model import predictor_dict, convdict, GCN, DropEdge, VSF_GCN
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from local_visual_integrater import Resnet50VE, Resnet50VE_Tuned, AttentionModule, generate_one_hop_node_subgraph, generate_subgraphs, parallel_generate_subgraphs_cpu, integrate_image_features, preprocess_image
import time
from copy import deepcopy
from tqdm import tqdm

from utils import Logger
from typing import Iterable
import random


def set_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

class PermIterator:

    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret
    
def loaddataset(name, use_val_edges_as_input, load=None):
   
    dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    edge_index = data.edge_index
    data.edge_weight = None 
    print(data.num_nodes, edge_index.max())
    # if data.edge_weight is None else data.edge_weight.view(-1).to(torch.float)
    # data = T.ToSparseTensor()(data)
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    if name == "ppa":
        data.x = torch.argmax(data.x, dim=-1)
        data.max_x = torch.max(data.x).item()
    elif name == "ddi":
        #data.x = torch.zeros((data.num_nodes, 1))
        data.x = torch.arange(data.num_nodes)
        data.max_x = data.num_nodes
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])


    # Use training + validation edges for inference on test set.
    if use_val_edges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data, split_edge

def train(attention_module, model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None):
    def penalty(posout, negout):
        scale = torch.ones_like(posout[[0]]).requires_grad_()
        loss = -F.logsigmoid(posout*scale).mean()-F.logsigmoid(-negout*scale).mean()
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(torch.square(grad))
    
    if alpha is not None:
        predictor.setalpha(alpha)
    attention_module.train()  
    model.train()
    predictor.train()
    
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm in tqdm(PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    )):
        optimizer.zero_grad()
        # collab
        if not data.max_x >= 0: integrated_features = attention_module(data.x, data.image_features)
        
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        
        if data.max_x >= 0: h = model(data.x, data.image_features, adj)
        else: h = model(integrated_features, adj)

        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        
        # 清理中间变量
        del h, pos_outs, neg_outs, pos_losss, neg_losss, loss
        if not data.max_x >= 0: del integrated_features
        # torch.cuda.synchronize()  # 确保所有 CUDA 操作完成
    total_loss = np.average(total_loss)
    return total_loss


def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    # test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    
    valid_auc = round(valid_auc, 4)
    # test_auc = round(test_auc, 4)

    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    # test_ap = average_precision_score(test_true, test_pred)
    
    valid_ap = round(valid_ap, 4)
    # test_ap = round(test_ap, 4)
    
    results['AP'] = valid_ap


    return results


@torch.no_grad()
def test(attention_module, model, predictor, data, split_edge, evaluator, batch_size,
         use_val_edges_as_input):
    split = '_use_val' if use_val_edges_as_input else ''
    attention_module.eval()
    model.eval()
    predictor.eval()

    # pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    
    # integrated_features = torch.zeros((data.num_nodes, data.num_features)).to(data.x.device)       
    # for node in range(data.num_nodes):
    #     integrated_features[node] = attention_module(data.x[node], data.image_features[node])
    if data.max_x >= 0:
        h = model(data.x, data.image_features, adj)
    else:
        integrated_features = attention_module(data.x, data.image_features)
        h = model(integrated_features, adj)

    '''
    pos_train_pred = torch.cat([
        predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    '''

    pos_valid_pred = torch.cat([
        predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    neg_valid_pred = torch.cat([
        predictor(h, adj, neg_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    if use_val_edges_as_input:
        adj = data.full_adj_t
        if data.max_x >= 0: h = model(data.x, data.image_features, adj)
        else: h = model(integrated_features, adj)
        # h = model(data.x, adj)

    pos_test_pred = torch.cat([
        predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    neg_test_pred = torch.cat([
        predictor(h, adj, neg_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K

        train_hits = 0
        '''
        evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        '''

        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
    

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    train_auc = 0
    val_pred = torch.cat([pos_valid_pred, neg_valid_pred])
    val_true = torch.cat([torch.ones(pos_valid_pred.size(0), dtype=int), 
                            torch.zeros(neg_valid_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])
    
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    results['AUC'] = (train_auc, result_auc_val['AUC'], result_auc_test['AUC'])
    results['AP'] = (train_auc, result_auc_val['AP'], result_auc_test['AP'])

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(),  h.cpu()]


    return results, h.cpu(), score_emb

def save_emb(score_emb, save_path):

    pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x= score_emb
    state = {
    'pos_valid_score': pos_valid_pred,
    'neg_valid_score': neg_valid_pred,
    'pos_test_score': pos_test_pred,
    'neg_test_score': neg_test_pred,
    'node_emb': x
    }

    torch.save(state, save_path)



def parseargs():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--use_val_edges_as_input', action='store_true')
    parser.add_argument('--mplayers', type=int, default=1)
    parser.add_argument('--nnlayers', type=int, default=3)
    parser.add_argument('--ln', action="store_true")
    parser.add_argument('--lnnn', action="store_true")
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--jk', action="store_true")
    parser.add_argument('--maskinput', action="store_true")
    parser.add_argument('--hiddim', type=int, default=32)
    parser.add_argument('--gnndp', type=float, default=0.3)
    parser.add_argument('--xdp', type=float, default=0.3)
    parser.add_argument('--tdp', type=float, default=0.3)
    parser.add_argument('--gnnedp', type=float, default=0.3)
    parser.add_argument('--predp', type=float, default=0.3)
    parser.add_argument('--preedp', type=float, default=0.3)
    parser.add_argument('--splitsize', type=int, default=-1)
    parser.add_argument('--gnnlr', type=float, default=0.0003)
    parser.add_argument('--prelr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--testbs', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--trndeg', type=int, default=-1)
    parser.add_argument('--tstdeg', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default="collab")
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument('--model', choices=convdict.keys())
    parser.add_argument('--cndeg', type=int, default=-1)
    parser.add_argument('--save_gemb', action="store_true")
    parser.add_argument('--load', type=str)
    parser.add_argument('--cnprob', type=float, default=0)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--increasealpha", action="store_true")
    parser.add_argument("--savex", action="store_true")
    parser.add_argument("--loadx", action="store_true")
    parser.add_argument("--loadmod", action="store_true")
    parser.add_argument("--savemod", action="store_true")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=30,    type=int,       help='early stopping')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--save', action="store_true")

    # add for GVN
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--attn_lr', type=float, default=0.001)
    parser.add_argument('--VE', type=str, default='resnet50')
    parser.add_argument('--hop_num', type=int, default=2)
    parser.add_argument('--feat', type=int, default=5)
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--nolabel", action="store_true")
    parser.add_argument("--color_center", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    print(args, flush=True)
    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    # writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    # writer.add_text("hyperparams", hpstr)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
        evaluator = Evaluator(name=f'ogbl-ppa')
    else:
        evaluator = Evaluator(name=f'ogbl-{args.dataset}')

    data, split_edge = loaddataset(args.dataset, args.use_val_edges_as_input, args.load)

    data = data.to(device)

    predfn = predictor_dict[args.predictor]

    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    ret = []
    ret_auc = []

    if args.dataset =='collab':
        eval_metric = 'Hits@50'
    elif args.dataset =='ddi':
        eval_metric = 'Hits@20'

    elif args.dataset =='ppa':
        eval_metric = 'Hits@100'
    
    elif args.dataset =='citation2':
        eval_metric = 'MRR'

    loggers = {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'AUC': Logger(args.runs),
        'AP': Logger(args.runs)
      
    }

   

    for run in range(0, args.runs):
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        set_seed(seed)

        save_path = args.output_dir+'/lr'+str(args.gnnlr) + '_drop' + str(args.gnndp) +  '_numlayer' + str(args.mplayers)+ '_dim'+str(args.hiddim) + '_'+ 'best_run_'+str(seed)
        
        if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            data, split_edge = loaddataset(args.dataset, args.use_val_edges_as_input, args.load)
            data = data.to(device)
        bestscore = None
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        if args.dataset in ['ppa', 'ddi']:
            model = VSF_GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        
        if args.loadx:
            with torch.no_grad():
                model.xemb[0].weight.copy_(torch.load(f"gemb/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"))
            model.xemb[0].weight.requires_grad_(False)
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        
        if args.dataset in ['collab']: generate_subgraphs(data=data, dataname=args.dataset, hop_num=args.hop_num, feat=args.feat, relabel=args.relabel, use_val=args.use_val_edges_as_input, color_center = args.color_center, nolabel = args.nolabel, tmp_image=True)
        elif args.dataset in ['ppa', 'ddi']: 
            if args.hop_num == 1:
                generate_one_hop_node_subgraph(data=data, dataname=args.dataset)
            else:
                parallel_generate_subgraphs_cpu(data=data, dataname=args.dataset, hop_num=args.hop_num, feat=args.feat, relabel=args.relabel, use_val=args.use_val_edges_as_input, color_center = args.color_center, nolabel = args.nolabel, tmp_image=True, num_workers=20)
        # Initialize VE and fusion module
        image_feature_dim_dict = {'resnet50':2048}
        VE_dict = {'resnet50': Resnet50VE}
        image_feature_dim = image_feature_dim_dict[args.VE]
        feature_extractor = VE_dict[args.VE]().to(device)
        attention_module = AttentionModule(data.num_node_features, image_feature_dim, args.attention_dim).to(device)
        
        # Generate Local node-centered subgraph images and extract VSF for node-centered subgraph
        data.image_features = torch.zeros((data.num_nodes, attention_module.image_feature_dim)).to(data.x.device)
        split_flag = "_useval" if args.use_val_edges_as_input else ""
        label_flag = ''
        if args.relabel: label_flag = 'relabel_'
        if args.nolabel: label_flag = 'nolabel_'
        hop_label = 'onehop' if args.hop_num == 1 else "mulhop"
        VSF_tensor_path = f'dataset/{args.dataset}/{label_flag}_VSF_{hop_label}.pt' 
        if os.path.exists(VSF_tensor_path): data.image_features = torch.load(VSF_tensor_path).to(device)
        
        if args.loadmod:
            keys = model.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
            keys = predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pre.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
            keys = attention_module.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.attn.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
            
        if not data.max_x >= 0:
            optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
            {'params': attention_module.parameters(), 'lr': args.attn_lr},
            {'params': predictor.parameters(), 'lr': args.prelr}])
        else:
            attention_params = list(model.attn_module.parameters())
            gnn_params = [param for name, param in model.named_parameters() if "attn_module" not in name]
            optimizer = torch.optim.Adam([{'params': gnn_params, "lr": args.gnnlr}, 
            {'params': attention_params, 'lr': args.attn_lr},
            {'params': predictor.parameters(), 'lr': args.prelr}])
        kill_cnt = 0
        best_valid = 0
        for epoch in range(1, args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(attention_module, model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha)
            print(f"trn time {time.time()-t1:.2f} s", flush=True)
            if True:
                t1 = time.time()
                results, h, score_emb = test(attention_module, model, predictor, data, split_edge, evaluator,
                               args.testbs, args.use_val_edges_as_input)
                print(f"test time {time.time()-t1:.2f} s")
                if bestscore is None:
                    bestscore = {key: list(results[key]) for key in results}

                if True:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        loggers[key].add_result(run, result)

                        if key == eval_metric:
                            current_valid_eval = valid_hits

                        if valid_hits > bestscore[key][1]:
                            bestscore[key] = list(result)
                            if args.save_gemb:
                                torch.save(h, f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
                            if args.savex:
                                torch.save(model.xemb[0].weight.detach(), f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                            if args.savemod:
                                torch.save(model.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                                torch.save(predictor.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt")
                                torch.save(attention_module.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.attn.pt")
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                        
                    print(f'best {eval_metric}, '
                            f': {bestscore[eval_metric][2]:.4f}%, ')
                    key_AUC='AUC'
                    print(f'best auc, '
                            f': {bestscore[key_AUC][2].item():.4f}%, ')
                        
        
                    print('---', flush=True)

                if bestscore[eval_metric][1] >   best_valid:
                    kill_cnt = 0
                    best_valid =  bestscore[eval_metric][1]
                    if args.save:
                        save_emb(score_emb, save_path)
                

                else:
                    kill_cnt += 1
                   
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")

                        break

                
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

   
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

    

if __name__ == "__main__":
    main()