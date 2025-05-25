import argparse
import numpy as np
import torch
import sys

sys.path.append("..") 
from gnn_model import *
from utils import *
from scoring import mlp_score
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from baseline_models.NCN.util import PermIterator
import time
# from ogbdataset import loaddataset
from typing import Iterable
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from utils import init_seed, Logger, save_emb, get_logger
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from local_visual_integrater import seed_torch, LinearProjector, Resnet50VE, Resnet50VE_Tuned, AttentionModule, generate_one_hop_node_subgraph, generate_subgraphs, parallel_generate_subgraphs_cpu, integrate_image_features, preprocess_image
from utils import *
from copy import deepcopy
from tqdm import tqdm

seed_torch(11)
log_print = get_logger('testrun', 'log', get_config_dir())

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 3, 10, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [1, 3, 10, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])


    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )
    
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    # for K in [1,3,10, 100]:
    #     result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

   
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

def train(args, attention_module, feature_extractor, model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None):
    
    # if alpha is not None:
    #     predictor.setalpha(alpha)
    
    attention_module.train()
    feature_extractor.train()
    model.train()
    predictor.train()

    
    # duplicate_count = 0
    # vsf_set = set()
    
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm, is_last_batch in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        need_retain_graph = False if is_last_batch else True
        optimizer.zero_grad()
        
        # Vision 
        projected_image_features = feature_extractor(data.image_features) if args.fp_dim > 0 else data.image_features
        integrated_features = attention_module(data.x, projected_image_features)
        
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
        h = model(integrated_features, adj)
        edge = pos_train_edge[:, perm]
        
        # load_vsfs = torch.cat((h[edge[0]],h[edge[1]]), dim=-1).tolist()
        # for vsf in load_vsfs:
        #     vsf_str = str(vsf)  # Convert tensor to string for hashing
        #     if vsf_str in vsf_set:
        #         duplicate_count += 1
        #     else:
        #         vsf_set.add(vsf_str)
        
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_losss = -torch.log(pos_out + 1e-15).mean()
        edge = negedge[:, perm]
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_losss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = neg_losss + pos_losss
        loss.backward(retain_graph=need_retain_graph)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), 1.0)
        
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    # print(f"Number of elements with at least one duplicate in load_trained_vsf: {duplicate_count/ len(pos_train_edge.t())}")
    return total_loss

# python main_gvn_node.py --nolabel --VE resnet50 --hop_num 2 --color_center --dataset cora  --gnnlr 0.01 --prelr 0.01 --l2 5e-4  --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 1 --hiddim 128 --testbs 512 --epochs 9999 --eval_steps 5  --batch_size 1024     --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4    --probscale 4.3 --proboffset 2.8 --alpha 1.0    --ln --lnnn --predictor incn1cn1 --model puregcn   --maskinput  --jk  --use_xlin  --tailact --attn_lr 1e-6 --velr 0.01 --runs 3 --kill_cnt 1000 --fp_dim 0 --attention_dim 512
@torch.no_grad()
def test(args, attention_module, feature_extractor, model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr, batch_size,
         use_val_edges_as_input):
    split = 'train_val' if use_val_edges_as_input else 'train'
    attention_module.eval()
    feature_extractor.eval()
    model.eval()
    predictor.eval()

    # pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    
    # Vision 
    projected_image_features = feature_extractor(data.image_features) if args.fp_dim > 0 else data.image_features
    integrated_features = attention_module(data.x, projected_image_features)
    h_val = model(integrated_features, adj)

    '''
    pos_train_pred = torch.cat([
        predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
        for perm, is_last in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    '''

    pos_valid_pred = torch.cat([
        predictor(h_val[pos_valid_edge[perm].t()[0]], h_val[pos_valid_edge[perm].t()[1]]).squeeze().cpu()
        for perm, is_last in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    neg_valid_pred = torch.cat([
        predictor(h_val[neg_valid_edge[perm].t()[0]], h_val[neg_valid_edge[perm].t()[1]]).squeeze().cpu()
        for perm, is_last in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)

    adj = data.full_adj_t if use_val_edges_as_input else data.adj_t
    h_test = model(integrated_features, adj)

    # t1 = time.time()
    pos_test_pred = torch.cat([
        predictor(h_test[pos_test_edge[perm].t()[0]], h_test[pos_test_edge[perm].t()[1]]).squeeze().cpu()
        for perm, is_last in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)
    # print("inference time", ((time.time()-t1)/len(pos_test_edge)*batch_size))
    

    
    neg_test_pred = torch.cat([
        predictor(h_test[neg_test_edge[perm].t()[0]], h_test[neg_test_edge[perm].t()[1]]).squeeze().cpu()
        for perm, is_last in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    
    print('train valid_pos valid_neg test_pos test_neg', pos_valid_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_valid_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb





def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--testbs', type=int, default=512)
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--use_val_edges_as_input', type=bool, default=True)
    
    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)

    # add for consistency
    parser.add_argument("--savex", action="store_true")
    parser.add_argument("--loadx", action="store_true")
    parser.add_argument("--loadmod", action="store_true")
    parser.add_argument("--savemod",  action="store_true")
    parser.add_argument('--load', type=str)
    parser.add_argument("--increasealpha", action="store_true")
    parser.add_argument('--maskinput',  action="store_true")
    
    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--begin_seed', type=int, default=0)
    parser.add_argument('--gnnlr', type=float, default=0.001)
    parser.add_argument('--prelr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=10000, type=int, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=5)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='Hits@100')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument("--edge_split", type=str, default="7:1:2")
    ### add for GVN
    parser.add_argument("--resample_steps", type=int, default=50)
    parser.add_argument("--velr", type=float, default=0.0)
    parser.add_argument("--attn_lr", type=float, default=0.0001)
    parser.add_argument("--VE", type=str, default='resnet50')
    parser.add_argument("--attention_dim", type=int, default=512)
    parser.add_argument("--fp_dim", type=int, default=0)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no_attn", action="store_true")
    # for graph image generation
    parser.add_argument('--hop_num', type=int, default=2)
    parser.add_argument('--feat', type=int, default=5)
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--nolabel", action="store_true")
    parser.add_argument("--color_center", action="store_true")
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ###### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    
    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print(args, flush=True)

    init_seed(args.seed)
  
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs)
    }


    data, split_edge = loaddataset(args.dataset, args.edge_split, args.use_val_edges_as_input, args.load)

    data = data.to(device)
    node_num = data['x'].size(0)

    # ones_tensor = torch.ones_like(data['x'])
    # # Replace data.x with the ones tensor
    # data['x'] = ones_tensor
    
    x = data['x']

    if args.cat_n2v_feat:
        print('cat n2v embedding!!')
        n2v_emb = torch.load(os.path.join(get_root_dir(), 'dataset', args.data_name+'-n2v-embedding.pt'))
        x = torch.cat((x, n2v_emb), dim=-1)

    x = x.to(device)

    input_channel = x.size(1)
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, node_num, args.cat_node_feat_mf).to(device)
    
    predictor = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
    
    
    for run in range(0, args.runs):
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run + args.seed
        print('seed: ', seed)

        init_seed(seed)

        relabel_flag = '_relabel' if args.relabel else ''
        relabel_flag = '_relabel' if args.relabel else ''
        nolabel_flag = '_nolabel' if args.nolabel else ''
        save_path = f'{args.output_dir}/gvn_node_gcn_{args.dataset}'+str(args.resample_steps) + '_attnlr' + str(args.attn_lr) + '_velr'+ str(args.velr) + '_hopn' + str(args.hop_num)+ '_feat' + str(args.feat) + relabel_flag + nolabel_flag + '_'+ 'best_run_'+str(seed)


        model.reset_parameters()
        predictor.reset_parameters()
        
        if args.hop_num == 1:
            generate_one_hop_node_subgraph(data=data, dataname=args.dataset)
        else:
            generate_subgraphs(dataname=args.dataset, data=data, hop_num=args.hop_num, feat=args.feat, relabel=args.relabel, use_val=args.use_val_edges_as_input, color_center = args.color_center, nolabel = args.nolabel, tmp_image=True)
    
        
        image_feature_dim_dict = {'resnet50':2048, 'resnet50_tuned':2048}
        VE_dict = {'resnet50': Resnet50VE, 'resnet50_tuned':Resnet50VE_Tuned}
        image_feature_dim = image_feature_dim_dict[args.VE]
        feature_extractor = VE_dict[args.VE]().to(device)
        if args.fp_dim > 0: 
            feature_extractor = LinearProjector(image_feature_dim, args.fp_dim).to(device)
            image_feature_dim = args.fp_dim
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
        
        
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr},
           {'params': feature_extractor.parameters(), 'lr': args.velr},
           {'params': attention_module.parameters(), 'lr': args.attn_lr},
           {'params': predictor.parameters(), 'lr': args.prelr}],  weight_decay=args.l2)
        
        
        if args.loadmod:
            all_state_dict = torch.load(f'{save_path}.pth', map_location=device)
            # 加载各个模块的状态字典
            model.load_state_dict(all_state_dict['model'])
            optimizer.load_state_dict(all_state_dict['optimizer'])
            attention_module.load_state_dict(all_state_dict['attention_module'])
            feature_extractor.load_state_dict(all_state_dict['feature_extractor'])
            predictor.load_state_dict(all_state_dict['predictor'])
            print("Loaded saved model parameters.")
            results, score_emb = test(args, attention_module, feature_extractor, model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr,
                                args.testbs, args.use_val_edges_as_input)
            for key, result in results.items():
                _, valid_hits, test_hits = result

                
                loggers[key].add_result(run, result)
                    
                print(key)
                log_print.info(
                    f'Loaded Result: '
                        f'Valid: {100 * valid_hits:.2f}%, '
                        f'Test: {100 * test_hits:.2f}%')
            print('---', flush=True)
            save_emb(score_emb, save_path)
            if args.test: exit()
        else:
            print("No saved model parameters found. Starting training from scratch.")
        
        
        # test(args, attention_module, feature_extractor, model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr,
        #                         args.testbs, args.use_val_edges_as_input)
        # exit()
        
        best_test = 0
        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(args, attention_module, feature_extractor, model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha)
            # print(f"trn time {time.time()-t1:.2f} s", flush=True)
            
            t1 = time.time()
            if epoch % args.eval_steps == 0:
                
                results, score_emb = test(args, attention_module, feature_extractor, model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr,
                                args.testbs, args.use_val_edges_as_input)
                # print(f"test time {time.time()-t1:.2f} s")
            
                metric_current = ''
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
                
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.savemod:
                        # save_emb(score_emb, save_path)
                        all_state_dict = {
                            'feature_extractor':feature_extractor.state_dict(),
                        }
                        torch.save(all_state_dict, f'{save_path}_{metric_current}_{epoch}.pth')
                    best_test = metric_current
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
                print(f'Best Current: {best_test}_{epoch}')
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
            
        # if key == 'Hits@100':
        #     best_auc_valid_str = best_metric
        #     best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        
    
    print(result_all_run[eval_metric])

    return
    # return best_valid_mean_metric, best_auc_metric, result_all_run

 

if __name__ == "__main__":
    main()

