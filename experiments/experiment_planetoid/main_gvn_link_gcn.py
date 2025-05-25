import sys
sys.path.append("..") 
from collections import defaultdict
import torch
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
from scoring import mlp_score
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from visual_integrater import Resnet50VE, Resnet50VE_Tuned, FusionAttentionModule, FusionMOE, GraphVisualizer, FusionModule, mlp_vision_decoder
from visual_integrater import seed_torch, preprocess_batch_image
from torch_geometric.utils import k_hop_subgraph, negative_sampling
dir_path  = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())
import time
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
        store_config = "store" if pos_edge_flag else "vary"
        if pos_edge_flag:
            image_path = f'./dataset/{args.data_name}/{hop_num_label}{use_val_label}{src_node}_{dst_node}.png'
        else:
            image_path = f'./dataset/{args.data_name}/{hop_num_label}{use_val_label}tmp_neg_{graph_index}_{args.id}.png'
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

        graph_visualizer.convert_graph_to_image(
            src_node_subgraph_index, dst_node_subgraph_index, subgraph_edge_index,
            image_path, store_config, layout_aug=args.layout_aug
        )
    return neg_edge_to_graph_index

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

        

def train(args, feature_extractor, graph_visualizer, fusion_module, vision_score, gnn_score,
          data, 
          model, 
          train_pos, 
          optimizer, 
          batch_size):
    fusion_module.train()
    feature_extractor.train()
    model.train()
    vision_score.train()
    gnn_score.train()

    total_loss = total_examples = 0

    # duplicate_count = 0
    # vsf_set = set()

    dataloader_length = len(DataLoader(range(train_pos.size(0)), batch_size))
    for index, perm in enumerate(DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True)):
        # need_retain_graph = False if index == dataloader_length - 1 else True
        optimizer.zero_grad()


        num_nodes = data.num_nodes

        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
    
        train_edge_mask = train_pos[mask].transpose(1,0)

        
        # train_edge_mask = to_undirected(train_edge_mask)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
        # edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
            
        ###################
        # print(adj)

        h = model(data.x, adj)

        edge = train_pos[perm].t()
        train_pos_perm_list = train_pos[perm].tolist()

        
        # Extract VSF
        prepocessed_images = preprocess_batch_image(args, train_pos_perm_list, args.data_name)
        image_features = feature_extractor(prepocessed_images, train_pos.device)
        
        # Fusion VSF
        if args.modal_fusion == 'vision_only':
            pos_out = vision_score(image_features)
        elif args.modal_fusion == 'moe_fusion': 
            pos_out_vision_score = vision_score(image_features)
            pos_out_gnn_score = gnn_score(h[edge[0]], h[edge[1]])
            pos_out = fusion_module(pos_out_vision_score, pos_out_gnn_score)
        elif args.modal_fusion == 'attn_fusion': 
            fusioned_h_src, fusioned_h_dst = fusion_module(h[edge[0]], h[edge[1]], image_features)
            pos_out = gnn_score(fusioned_h_src, fusioned_h_dst)
        
        # load_vsfs = torch.cat((h[edge[0]],h[edge[1]],image_features), dim=-1).tolist()
        # for vsf in load_vsfs:
        #     vsf_str = str(vsf)  # Convert tensor to string for hashing
        #     if vsf_str in vsf_set:
        #         duplicate_count += 1
        #     else:
        #         vsf_set.add(vsf_str)
        
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        train_neg_perm_list = edge.t().tolist()
        
        # Generate Subgraph Vision for random train negedges
        neg_edge_to_graph_index = generate_graph_images(train_neg_perm_list, False, args, graph_visualizer,  data)
        
        # Extract VSF
        prepocessed_images = preprocess_batch_image(args, train_neg_perm_list, args.data_name, False, neg_edge_to_graph_index, args.id)
        image_features = feature_extractor(prepocessed_images, train_pos.device)
        
        # Fusion VSF
        if args.modal_fusion == 'vision_only':
            neg_out = vision_score(image_features)
        elif args.modal_fusion == 'moe_fusion': 
            neg_out_vision_score = vision_score(image_features)
            neg_out_gnn_score = gnn_score(h[edge[0]], h[edge[1]])
            neg_out = fusion_module(neg_out_vision_score, neg_out_gnn_score)
        elif args.modal_fusion == 'attn_fusion': 
            fusioned_h_src, fusioned_h_dst = fusion_module(h[edge[0]], h[edge[1]], image_features)
            neg_out = gnn_score(fusioned_h_src, fusioned_h_dst)
        
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(fusion_module.parameters(), 1.0)
        if args.clip_score:
            torch.nn.utils.clip_grad_norm_(vision_score.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(gnn_score.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    # print(f"Number of elements with at least one duplicate in load_trained_vsf: {duplicate_count/ len(train_pos.tolist())}")
    return total_loss / total_examples



@torch.no_grad()
def test_edge(feature_extractor, fusion_module, vision_score, gnn_score, input_data, args, h, batch_size, device):
    fusion_module.eval()
    feature_extractor.eval()
    vision_score.eval()
    gnn_score.eval()

    preds = []
    # inference_time = []
    for perm in DataLoader(range(input_data.size(0)), batch_size):
        # t1= time.time()
        edge = input_data[perm].t()
        edge_perm_list = input_data[perm].tolist()
        
        # Extract VSF
        prepocessed_images = preprocess_batch_image(args, edge_perm_list, args.data_name)
        image_features = feature_extractor(prepocessed_images, device)
        
        if args.modal_fusion == 'vision_only':
            preds += [vision_score(image_features).cpu()]
        elif args.modal_fusion == 'moe_fusion': 
            out_vision_score = vision_score(image_features)
            out_gnn_score = gnn_score(h[edge[0]], h[edge[1]])
            final_score = fusion_module(out_vision_score, out_gnn_score)
            preds += [final_score.cpu()]
        elif args.modal_fusion == 'attn_fusion': 
            fusioned_h_src, fusioned_h_dst = fusion_module(h[edge[0]], h[edge[1]], image_features)
            final_score = gnn_score(fusioned_h_src, fusioned_h_dst)
            preds += [final_score.cpu()]
        else:
            preds += [fusion_module(h[edge[0]], h[edge[1]], image_features).cpu()]
        # inference_time.append((time.time()-t1))
    # print('Inference time', sum(inference_time)/len(inference_time))
    pred_all = torch.cat(preds, dim=0)

    return pred_all


@torch.no_grad()
def test(args, split_edge, feature_extractor, fusion_module, vision_score, gnn_score, model, data, evaluator_hit, evaluator_mrr, batch_size,
         use_val_edges_as_input):
    split = 'train_val' if use_val_edges_as_input else 'train'
    fusion_module.eval()
    feature_extractor.eval()
    vision_score.eval()
    gnn_score.eval()
    model.eval()

    adj = data.adj_t
    h_val = model(data.x, adj)

    pos_valid_pred = test_edge(feature_extractor, fusion_module, vision_score, gnn_score, split_edge['valid']['edge'], args, h_val, batch_size, data.x.device)
    
    neg_valid_pred = test_edge(feature_extractor, fusion_module, vision_score, gnn_score, split_edge['valid']['edge_neg'], args, h_val, batch_size, data.x.device)

    adj = data.full_adj_t if use_val_edges_as_input else data.adj_t
    h_test = model(data.x, adj)
    
    pos_test_pred = test_edge(feature_extractor, fusion_module,  vision_score, gnn_score, split_edge['test']['edge'], args, h_test, batch_size, data.x.device)

    neg_test_pred = test_edge(feature_extractor, fusion_module,  vision_score, gnn_score, split_edge['test']['edge_neg'], args, h_test, batch_size, data.x.device)

    # pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)


    print('train valid_pos valid_neg test_pos test_neg', pos_valid_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_valid_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb




def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--use_val_edges_as_input', action="store_true")
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    

    parser.add_argument('--load', type=str)
    parser.add_argument("--loadmod", action="store_true")
    parser.add_argument("--savemod",  action="store_true")
    parser.add_argument("--tqdm",  action="store_true")
    
    
    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--kill_cnt',  dest='kill_cnt', default=10000, type=int, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument("--edge_split", type=str, default="7:1:2")
    
    ### add for GVN
    parser.add_argument("--velr", type=float, default=0.01)
    parser.add_argument("--fslr", type=float, default=0.01)
    parser.add_argument("--hop_num", type=int, default=1)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--feat", type=int, default=10)
    parser.add_argument("--VE", type=str, default='resnet50_tuned')
    parser.add_argument("--no_label", action='store_true')
    parser.add_argument("--relabel", action='store_true')
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--layout_aug", action="store_true")
    parser.add_argument("--et", action="store_true")
    parser.add_argument("--clip_score", action="store_true")
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='Hits@100')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--modal_fusion', type=str, default='concate')
    parser.add_argument('--attn_fusion', action='store_true', default=False)
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
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    data, split_edge = loaddataset(args.data_name, args.edge_split, args.use_val_edges_as_input, args.load)
    data = data.to(device)
    node_num = data.num_nodes
    
    # ones_tensor = torch.ones_like(data.x)
    # # Replace data.x with the ones tensor
    # data.x= ones_tensor
    x = data.x

    if args.cat_n2v_feat:
        print('cat n2v embedding!!')
        n2v_emb = torch.load(os.path.join(get_root_dir(), 'dataset', args.data_name+'-n2v-embedding.pt'))
        x = torch.cat((x, n2v_emb), dim=-1)

    train_pos = split_edge['train']['edge'].to(data.x.device)
    test_pos = split_edge['test']['edge'].to(data.x.device)
    test_neg = split_edge['test']['edge_neg'].to(data.x.device)
    valid_pos = split_edge['valid']['edge'].to(data.x.device)
    valid_neg = split_edge['valid']['edge_neg'].to(data.x.device)

    input_channel = x.size(1)
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, node_num, args.cat_node_feat_mf).to(device)
   
    
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

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run + args.seed
        print('seed: ', seed)

        init_seed(seed)
        
        
        save_path = args.output_dir+ f'/{args.data_name}'+ '_fslr' + str(args.fslr) + '_velr'+ str(args.velr) + 'run_'+str(seed)

        model.reset_parameters()
        # vision_score.reset_parameters()
        
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
        
        
        
        if args.modal_fusion == 'attn_fusion': fusion_module = FusionAttentionModule(args.hidden_channels, image_feature_dim, args.hidden_channels).to(device)
        elif args.modal_fusion == 'moe_fusion': fusion_module = FusionMOE().to(device)
        else: fusion_module = FusionModule(args.hidden_channels*2 + image_feature_dim, args.hidden_channels).to(device)
            
        vision_score = mlp_vision_decoder(image_feature_dim, args.hidden_channels,
                1, args.num_layers_predictor, args.dropout).to(device)

        gnn_score = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                1, args.num_layers_predictor, args.dropout).to(device)
            
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.lr},
           {'params': feature_extractor.parameters(), 'lr': args.velr},
           {'params': fusion_module.parameters(), 'lr': args.fslr},
           {'params': vision_score.parameters(), "lr": args.lr},
           {'params': gnn_score.parameters(), "lr": args.lr},
           ],            
            weight_decay=args.l2)
        
        if args.loadmod:
            all_state_dict = torch.load(f'{save_path}.pth', map_location=device)
            # 加载各个模块的状态字典
            model.load_state_dict(all_state_dict['model'])
            optimizer.load_state_dict(all_state_dict['optimizer'])
            # attention_module.load_state_dict(all_state_dict['attention_module'])
            feature_extractor.load_state_dict(all_state_dict['feature_extractor'])
            vision_score.load_state_dict(all_state_dict['vision_score'])
            gnn_score.load_state_dict(all_state_dict['gnn_score'])
            fusion_module.load_state_dict(all_state_dict['fusion_module'])
            print("Loaded saved model parameters.")
            
        if args.loadmod or args.test:
            results, score_emb = test(args, split_edge, feature_extractor, fusion_module, vision_score, gnn_score, model, data, evaluator_hit, evaluator_mrr, 
                                args.batch_size, args.use_val_edges_as_input)
            for key, result in results.items():
                _, valid_hits, test_hits = result

                
                loggers[key].add_result(run, result)
                    
                print(key)
                log_print.info(
                    f'Loaded Result: '
                        f'Valid: {100 * valid_hits:.2f}%, '
                        f'Test: {100 * test_hits:.2f}%')
            print('---', flush=True)
            # save_emb(score_emb, save_path)
            if args.test: exit()
        else:
            print("No saved model parameters found. Starting training from scratch.")
        
        # Visualization link-centered subgraphs
        
        combined_list = train_pos.tolist() + test_pos.tolist() + test_neg.tolist() + valid_pos.tolist() + valid_neg.tolist()
        generate_graph_images(combined_list, True, args, graph_visualizer,  data)
        
        # test(args, split_edge, feature_extractor, fusion_module, vision_score, gnn_score, model, data, evaluator_hit, evaluator_mrr, 
        #                         args.batch_size, args.use_val_edges_as_input)
        
        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(args, feature_extractor, graph_visualizer, fusion_module, vision_score, gnn_score, data, model, train_pos, optimizer, args.batch_size)
            # print(model.convs[0].att_src[0][0][:10])
            
            if args.et and epoch % args.eval_steps != 0: 
                results_rank, score_emb = test(args, split_edge, feature_extractor, fusion_module, vision_score, gnn_score, model, data, evaluator_hit, evaluator_mrr, 
                                args.batch_size, args.use_val_edges_as_input)
            elif epoch % args.eval_steps == 0:
                results_rank, score_emb = test(args, split_edge, feature_extractor, fusion_module, vision_score, gnn_score, model, data, evaluator_hit, evaluator_mrr, 
                                args.batch_size, args.use_val_edges_as_input)

                metric_current = ''
                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result


                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                        if key == eval_metric: 
                            metric_current = f'{100 * test_hits:.2f}'
                    print('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0

                    if args.savemod:
                        # save_emb(score_emb, save_path)
                        all_state_dict = {
                            # 'model': model.state_dict(),
                            # 'optimizer':optimizer.state_dict(),
                            # 'fusion_module':fusion_module.state_dict(),
                            # 'vision_score': vision_score.state_dict(),
                            # 'gnn_score': gnn_score.state_dict(),
                            'feature_extractor':feature_extractor.state_dict()
                        }
                        torch.save(all_state_dict, f'{save_path}_{metric_current}_{epoch}.pth')
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


            
        if key == 'Hits@100':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        
    
    print(best_metric_valid_str +' ' +best_auc_valid_str)

    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()

