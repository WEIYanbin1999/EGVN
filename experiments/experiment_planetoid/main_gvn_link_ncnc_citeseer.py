import argparse
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
sys.path.append("..") 
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from baseline_models.NCN.model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from baseline_models.NCN.util import PermIterator
import time
from copy import deepcopy
# from ogbdataset import loaddataset
from typing import Iterable
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from utils import init_seed, Logger, save_emb, get_logger
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from local_visual_integrater import seed_torch
from visual_integrater import Resnet50VE, Resnet50VE_Tuned, FusionAttentionModule, FusionMOE, GraphVisualizer, FusionModule, mlp_vision_decoder
from visual_integrater import preprocess_batch_image
from torch_geometric.utils import k_hop_subgraph, negative_sampling
from utils import *
from tqdm import tqdm
from collections import defaultdict
log_print = get_logger('testrun', 'log', get_config_dir())

seed_torch(11)
def generate_graph_images(edge_list, pos_edge_flag, args, graph_visualizer, data):
    neg_edge_to_graph_index = defaultdict()
    graph_index = 0
    if args.tqdm: edge_list = tqdm(edge_list)
    for edge in edge_list:
        src_node, dst_node = edge
        hop_num_label = "" if args.hop_num == 1 else f'hop_{args.hop_num}/'
        # use_val_label = "useval/" if args.use_val_edges_as_input else ''
        use_val_label = ''
        if args.no_label:
            no_label_label = 'link_nolabel/' 
        elif args.relabel:
            no_label_label = 'link_relabel/'
        else:
            no_label_label = ''
        store_config = "store" if pos_edge_flag else "vary"
        if pos_edge_flag:
            image_path = f'./dataset/{args.dataset}/{hop_num_label}{use_val_label}{no_label_label}{src_node}_{dst_node}.png'
        else:
            image_path = f'./dataset/{args.dataset}/{hop_num_label}{use_val_label}{no_label_label}tmp_neg_{graph_index}_{args.id}.png'
            neg_edge_to_graph_index['_'.join((str(edge[0]),str(edge[1])))] = graph_index
            graph_index += 1
            
        tmp_visible_graph = deepcopy(data)
        if not pos_edge_flag: 
            new_edge = torch.tensor([[src_node], [dst_node]], dtype=torch.long).to(data.x.device)
            tmp_visible_graph.edge_index = torch.cat([tmp_visible_graph.edge_index, new_edge], dim=1)
        
        subgraph_nodes, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
            [src_node, dst_node], num_hops=args.hop_num, edge_index=tmp_visible_graph.edge_index, relabel_nodes=True
        )
        # Get the node index in the subgraph
        subgraph_node_indices = subgraph_nodes.cpu().numpy()

        # Find the new index of the original "src node" and "dst node" in the subgraph
        src_node_subgraph_index = (subgraph_node_indices == src_node).nonzero()[0].item()
        dst_node_subgraph_index = (subgraph_node_indices == dst_node).nonzero()[0].item()
        
        mask = ((subgraph_edge_index[0] == src_node_subgraph_index) &
                (subgraph_edge_index[1] == dst_node_subgraph_index))
        subgraph_edge_index = subgraph_edge_index[:, ~mask]

        # if not pos_edge_flag: print(image_path)
        graph_visualizer.convert_graph_to_image(
            src_node_subgraph_index, dst_node_subgraph_index, subgraph_edge_index,
            image_path, store_config, layout_aug=args.layout_aug
        )
    return neg_edge_to_graph_index

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 3, 10, 20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [1, 3, 10, 20, 50, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])


    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )

    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
   
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])

    
    return result
def train(args, feature_extractor, graph_visualizer,
          model,
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
    
    feature_extractor.train()
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    # print(pos_train_edge.shape)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm, is_last in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        optimizer.zero_grad()
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
        h = model(data.x, adj)
        
        
        edge = pos_train_edge[:, perm]

        train_pos_perm_list = edge.t().tolist()
        
        # Extract VSF
        if args.use_vsf:
            prepocessed_images = preprocess_batch_image(args, train_pos_perm_list, args.dataset)
            image_features = feature_extractor(prepocessed_images, pos_train_edge.device)
        else:
            image_features = 1
        pos_outs = predictor.multidomainforward(image_features, h, adj, edge, cndropprobs=cnprobs, outer_forward=True)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        
        
        edge = negedge[:, perm]
        if args.use_vsf:
            train_neg_perm_list = edge.t().tolist()
            # Generate Subgraph Vision for random train negedges
            neg_edge_to_graph_index = generate_graph_images(train_neg_perm_list, False, args, graph_visualizer,  data)
            
            # Extract VSF
            prepocessed_images = preprocess_batch_image(args, train_neg_perm_list, args.dataset, False, neg_edge_to_graph_index, args.id)
            image_features = feature_extractor(prepocessed_images, pos_train_edge.device)
        
        neg_outs = predictor.multidomainforward(image_features, h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        del image_features, h, pos_outs, neg_outs, pos_losss, neg_losss, loss

    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss


@torch.no_grad()
def test(args, feature_extractor, model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr, batch_size,
         use_val_edges_as_input):
    feature_extractor.eval()
    model.eval()
    predictor.eval()

    # pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)

    '''
    pos_train_pred = torch.cat([
        predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    '''
    image_features = 1
    predictions = []
    for perm, last in PermIterator(pos_valid_edge.device, pos_valid_edge.shape[0], batch_size, False):
        edge_perm_list = pos_valid_edge[perm].tolist()
        # Extract VSF
        if args.use_vsf:
            prepocessed_images = preprocess_batch_image(args, edge_perm_list, args.dataset)
            image_features = feature_extractor(prepocessed_images, pos_valid_edge.device)
        pred = predictor(image_features, h, adj, pos_valid_edge[perm].t(), True).squeeze().cpu()
        predictions.append(pred)
    pos_valid_pred = torch.cat(predictions, dim=0)
        
    predictions = []
    for perm, last in PermIterator(neg_valid_edge.device, neg_valid_edge.shape[0], batch_size, False):
        if args.use_vsf:
            edge_perm_list = neg_valid_edge[perm].tolist()
            # Extract VSF
            prepocessed_images = preprocess_batch_image(args, edge_perm_list, args.dataset)
            image_features = feature_extractor(prepocessed_images, neg_valid_edge.device)
        pred = predictor(image_features, h, adj, neg_valid_edge[perm].t(),True).squeeze().cpu()
        predictions.append(pred)
    neg_valid_pred = torch.cat(predictions, dim=0)
        
    if use_val_edges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)


    predictions = []
    # inference_time = []
    for perm, last in PermIterator(pos_test_edge.device, pos_test_edge.shape[0], batch_size, False):
        # t1= time.time()
        edge_perm_list = pos_test_edge[perm].tolist()
        if args.use_vsf:
            # Extract VSF
            prepocessed_images = preprocess_batch_image(args, edge_perm_list, args.dataset)
            image_features = feature_extractor(prepocessed_images, pos_test_edge.device)
        pred = predictor(image_features, h, adj, pos_test_edge[perm].t(), True).squeeze().cpu()
        predictions.append(pred)
    #     inference_time.append(time.time()-t1)
    # print("inference time:", sum(inference_time)/len(inference_time))
    pos_test_pred = torch.cat(predictions, dim=0)

    predictions = []
    for perm, last in PermIterator(neg_test_edge.device, neg_test_edge.shape[0], batch_size, False):
        edge_perm_list = neg_test_edge[perm].tolist()
        if args.use_vsf:
            # Extract VSF
            prepocessed_images = preprocess_batch_image(args, edge_perm_list, args.dataset)
            image_features = feature_extractor(prepocessed_images, neg_test_edge.device)
        pred = predictor(image_features, h, adj, neg_test_edge[perm].t(), True).squeeze().cpu()
        predictions.append(pred)
    neg_test_pred = torch.cat(predictions, dim=0)

    
    print('train valid_pos valid_neg test_pos test_neg', pos_valid_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_valid_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]

    return result, score_emb


def parseargs():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--use_val_edges_as_input', type=bool, default=True)
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
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--loadmod", action="store_true")
    parser.add_argument("--savemod", action="store_true")
    parser.add_argument("--edge_split", type=str, default="7:1:2")
    ### add for GVN
    parser.add_argument("--velr", type=float, default=0.01)
    parser.add_argument("--fslr", type=float, default=0.01)
    parser.add_argument("--hop_num", type=int, default=2)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--feat", type=int, default=10)
    parser.add_argument("--VE", type=str, default='resnet50_tuned')
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--layout_aug", action="store_true")
    parser.add_argument("--et", action="store_true")
    parser.add_argument("--clip_score", action="store_true")
    parser.add_argument("--no_label", action="store_true")
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--simple', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='Hits@100')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--attention_dim', type=int, default=128)
    parser.add_argument('--tqdm', action='store_true', default=False)
    parser.add_argument('--use_vsf', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=10000, type=int, help='early stopping')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--eval_steps', type=int, default=5)
    
    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    print(args, flush=True)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs)
    }


    data, split_edge = loaddataset(args.dataset, args.edge_split, args.use_val_edges_as_input, args.load)

    data = data.to(device)

    predfn = predictor_dict[args.predictor]

    train_pos = split_edge['train']['edge'].to(data.x.device)
    test_pos = split_edge['test']['edge'].to(data.x.device)
    test_neg = split_edge['test']['edge_neg'].to(data.x.device)
    valid_pos = split_edge['valid']['edge'].to(data.x.device)
    valid_neg = split_edge['valid']['edge_neg'].to(data.x.device)
    
    
    if args.simple:
        if args.predictor != "cn0":
            predfn = partial(predfn, cndeg=args.cndeg)
        if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
            predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
        if args.predictor in ["incn1cn1"]:
            predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
        if args.predictor in ["gvnlincn1cn1v1", "gvnlincn1cn1v2"]:
            predfn = partial(predfn, attn_dim=args.attention_dim, image_dim=2048)
        ret = []
    else:
        if args.predictor != "cn0":
            predfn = partial(predfn, cndeg=args.cndeg)
        if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1", "gvnlincn1cn1v1"]:
            predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
        if args.predictor in ["incn1cn1", "gvnlincn1cn1v1"]:
            predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
        if args.predictor in ["gvnlincn1cn1v1", "gvnlincn1cn1v2"]:
            predfn = partial(predfn, attn_dim=args.attention_dim, image_dim=2048)
        ret = []
    
    for run in range(0, args.runs):
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run + args.seed
        print('seed: ', seed)

        init_seed(seed)

        save_path = args.output_dir+'/lr'+str(args.gnnlr) + '_drop' + str(args.gnndp) + '_l2'+ str(args.l2) + '_numlayer' + str(args.mplayers)+ '_numPredlay' + str(args.nnlayers) +'_dim'+str(args.hiddim) + '_'+ 'best_run_'+str(seed)


        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
       
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
       
        # GVN module components
        image_feature_dim_dict = {
            'resnet50':2048,
            'resnet50_tuned':2048
            }
        VE_dict = {
            'resnet50': Resnet50VE, 
            'resnet50_tuned':Resnet50VE_Tuned
            }
        image_feature_dim = image_feature_dim_dict[args.VE]
        feature_extractor = VE_dict[args.VE]().to(device)
        graph_visualizer = GraphVisualizer()
        
        if args.predictor == 'gvnlincn1cn1v1':
            args.use_vsf = True
            fusion_params = list(predictor.attention_module.parameters())
            predictor_params = [param for name, param in predictor.named_parameters() if "attention_module" not in name]
        if args.predictor == "gvnlincn1cn1v2":
            fusion_params = list(predictor.vsflin.parameters())
            predictor_params = [param for name, param in predictor.named_parameters() if "vsflin" not in name]
        
        
        
        
        if args.use_vsf:
           optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor_params, 'lr': args.prelr},
           {'params': fusion_params, 'lr': args.fslr},
           {'params': feature_extractor.parameters(), 'lr': args.velr}],  weight_decay=args.l2)
        else:
            optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr},
           {'params': feature_extractor.parameters(), 'lr': args.velr}],  weight_decay=args.l2)
        
        # Visualization link-centered subgraphs
        
        combined_list = train_pos.tolist() + test_pos.tolist() + test_neg.tolist() + valid_pos.tolist() + valid_neg.tolist()
        print(len(combined_list))
        generate_graph_images(combined_list, True, args, graph_visualizer,  data)
        
        # Add
        # test(args, feature_extractor, model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr,
        #                         args.testbs, args.use_val_edges_as_input)
        
        
        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(args, feature_extractor, graph_visualizer, model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha)
            print(f"trn time {time.time()-t1:.2f} s", flush=True)
            
            t1 = time.time()
            if epoch % args.eval_steps == 0:
                results, score_emb = test(args, feature_extractor, model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr,
                                args.testbs, args.use_val_edges_as_input)
                print(f"test time {time.time()-t1:.2f} s")
            
                
                for key, result in results.items():
                    _, valid_hits, test_hits = result

                   
                    loggers[key].add_result(run, result)
                        
                    print(key)
                    log_print.info(
                        f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                    if key == eval_metric: 
                        metric_current = f'{100 * test_hits:.2f}'
                print('---', flush=True)

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max().item()

                if best_valid_current > best_valid or best_valid_current == torch.tensor(loggers[eval_metric].results[run])[:, 1][-1].item():
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save:

                        save_emb(score_emb, save_path)
                    print(f'Best Current: {metric_current}_{epoch}')
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
    
    result_all_run = {}
    for key in loggers.keys():
        print(key)
        
        best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()

        if key == eval_metric:
            best_metric_valid_str = best_metric
            best_valid_mean_metric = best_valid_mean
            
        if key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        
    
    print(best_metric_valid_str +' ' +best_auc_valid_str)

    return best_valid_mean_metric, best_auc_metric, result_all_run

 

if __name__ == "__main__":
    main()
